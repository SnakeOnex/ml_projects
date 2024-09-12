import argparse, time, tqdm, PIL, wandb, numpy as np
import torch, torch.nn as nn, torchvision
from pathlib import Path
from vqgan import VQGAN, VQGANConfig
from model_configs import dataset_loaders
from gpt import GPTLanguageModel, GPTConfig
from utils import get_free_gpu, denormalize
from dataclasses import dataclass, field

device = torch.device(get_free_gpu())
print("selected device: ", device)

@dataclass
class TrainGPTConfig:
    gpt_config: GPTConfig
    vqgan_path: str
    vqgan_config: VQGANConfig = field(default_factory=lambda: VQGANConfig(K=1024, D=256))
    dataset: str = "flower"
    batch_size: int = 16
    epochs: int = 1000
    lr: float = 4e-5
    p_keep: float = 0.9
    betas: tuple = (0.5, 0.9)
    log_interval: int = 10
    eval_interval: int = 500

class TrainGPT:
    def __init__(self, config: TrainGPTConfig):
        self.config = config

        # 1. init models
        self.vqgan = VQGAN(config.vqgan_config).to(device).eval()
        self.vqgan.load_state_dict(torch.load(self.config.vqgan_path, map_location=device))
        self.gpt = GPTLanguageModel(config.gpt_config).to(device)

        # 2. optimizers
        self.optim = self.configure_optimizers()

        # 3. dataset
        self.train_loader, self.test_loader = dataset_loaders[self.config.dataset](self.config.batch_size)
        self.train_dataset, self.test_dataset = self.train_loader.dataset, self.test_loader.dataset

        # 4. run folder
        run_name = f"gpt-{self.config.dataset}-{time.time():.0f}"
        self.run_folder = Path("runs_gpt") / run_name
        self.run_folder.mkdir(exist_ok=True, parents=True)
        wandb.init(project="gpt-vqgan", name=run_name, config={**self.config.__dict__})

        # 4. running variables
        self.steps = 0
        self.best_loss = float("inf")

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.gpt.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"): no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules): decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules): no_decay.add(fpn)
        no_decay.add("position_embedding_table.weight")
        param_dict = {pn: p for pn, p in self.gpt.named_parameters()}
        optim_groups = [{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}]
        return torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=self.config.betas)

    @torch.no_grad()
    def evaluate(self):
        val_loss = 0
        bar = tqdm.tqdm(self.test_loader, desc="eval")
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            _, quantized, _ = self.vqgan(x)
            quantized = quantized.view(x.shape[0], -1)

            image_tokens = torch.cat([(y.view((-1,1))*0)+self.config.vqgan_config.K, quantized], dim=1)
            tokens_x = image_tokens[:,:self.config.gpt_config.block_size]
            tokens_y = image_tokens[:,1:self.config.gpt_config.block_size+1]
            _, loss = self.gpt(tokens_x, tokens_y)
            bar.set_postfix(loss=f"{loss.item():.4f}")
            val_loss += loss.item()
        val_loss /= len(self.test_loader)
        wandb.log({"val/loss": val_loss})

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            # torch.save(self.gpt.state_dict(), self.run_folder / "best.pth")

    def train(self):
        for epoch in range(self.config.epochs):
            bar = tqdm.tqdm(self.train_loader)
            for x, y in bar:
                x, y = x.to(device), y.to(device)

                # 1. get VQ-GAN quantized tokens
                with torch.no_grad(): 
                    _, quantized, _ = self.vqgan(x)
                quantized = quantized.view(x.shape[0], -1)

                # 2. add sos / class token
                image_tokens = torch.cat([(y.view((-1,1))*0)+self.config.vqgan_config.K, quantized], dim=1)

                # 3. input & target tokens
                tokens_x = image_tokens[:,:self.config.gpt_config.block_size]
                tokens_y = image_tokens[:,1:self.config.gpt_config.block_size+1]

                # 4. mask input tokens
                mask = torch.bernoulli(self.config.p_keep * torch.ones(tokens_x.shape, device=device)).to(dtype=torch.int64)
                random_indices = torch.randint_like(tokens_x, self.config.gpt_config.vocab_size)
                tokens_x = mask * tokens_x + (1 - mask) * random_indices

                # 5. train
                _, loss = self.gpt(tokens_x, tokens_y)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                # 6. log
                bar.set_postfix(loss=f"{loss.item():.4f}")

                if self.steps % self.config.log_interval == 0:
                    wandb.log({"train/loss": loss.item()})

                if self.steps % self.config.eval_interval == 0:
                    self.evaluate()
                    self.generate_samples(self.run_folder / f"{self.steps}.jpg")
                    wandb.log({"Generated samples": [wandb.Image(str(self.run_folder / f"{self.steps}.jpg"))]})
                    self.generate_completions(self.run_folder / f"{self.steps}_comp.jpg")
                    wandb.log({"Completions": [wandb.Image(str(self.run_folder / f"{self.steps}_comp.jpg"))]})
                self.steps += 1

    @torch.inference_mode()
    def generate_samples(self, path):
        context = torch.ones((16,1), dtype=torch.long, device=device)*self.config.vqgan_config.K
        res = self.gpt.generate(context, 256)[:,1:]
        res[res >= self.config.vqgan_config.K] = self.config.vqgan_config.K-1
        images = denormalize(self.vqgan.decode(res))
        grid = torchvision.utils.make_grid(images, nrow=4).permute(1, 2, 0).cpu().detach().numpy() * 255
        grid_final = PIL.Image.fromarray(grid.astype("uint8"))
        grid_final.save(path)
    
    @torch.inference_mode()
    def generate_completions(self, path):
        idxs = torch.randint(0, len(self.test_dataset), (4,))
        images_gt = torch.stack([self.test_dataset[i][0] for i in idxs]).to(device)

        with torch.inference_mode():
            images_rec, quantized, _ = self.vqgan(images_gt.to(device))
        quantized = quantized.view(images_gt.shape[0], -1)[:,:128]
        image_tokens = torch.cat([torch.ones((images_gt.shape[0],1), device=device).to(torch.int64)*self.config.vqgan_config.K, quantized], dim=1)
        res = self.gpt.generate(image_tokens, 128)[:,1:]
        images_comp = denormalize(self.vqgan.decode(res))
        images_gt = denormalize(images_gt)
        images_rec = denormalize(images_rec)

        grid_gt = torchvision.utils.make_grid(images_gt, nrow=4)
        grid_rec = torchvision.utils.make_grid(images_rec, nrow=4)
        grid_comp = torchvision.utils.make_grid(images_comp, nrow=4)
        grid = torch.cat([grid_gt, grid_rec, grid_comp], dim=1).permute(1, 2, 0).cpu().detach().numpy() * 255
        grid = PIL.Image.fromarray(grid.astype("uint8"))
        grid.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--lr", type=float, default=2.25e-6)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    vqgan_config = VQGANConfig(K=1024, D=256)
    gpt_config = GPTConfig(
            block_size=256, 
            vocab_size=1025, 
            n_embd=1024, 
            n_head=16, 
            n_layer=24,
            causal=True,
    ) 
    # vqgan_path = "runs_vqgan/vqgan-imagenet-1725884613/imagenet_best.pth"
    # vqgan_path = f"runs_vqgan/vqgan-flower-1725870990/flower_best.pth"
    vqgan_path = f"runs_vqgan/vqgan-bird-1726056084/best.pth"
    train_config = TrainGPTConfig(gpt_config=gpt_config, vqgan_config=vqgan_config, vqgan_path=vqgan_path, dataset=args.dataset, lr=args.lr, batch_size=args.batch_size)
    trainer = TrainGPT(train_config)
    trainer.train()
