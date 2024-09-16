import argparse, time, tqdm, PIL, wandb, numpy as np, os
import torch, torch.nn as nn, torchvision, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from vqgan import VQGAN, VQGANConfig
from model_configs import dataset_loaders
from gpt import GPTLanguageModel, GPTConfig
from utils import get_free_gpu, denormalize
from dataclasses import dataclass, field

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def gamma_func(ratio, mode):
    if mode == "linear":
        return 1 - ratio
    elif mode == "square":
        return 1 - ratio ** 2

@dataclass
class TrainMaskGITConfig:
    gpt_config: GPTConfig
    vqgan_path: str
    vqgan_config: VQGANConfig = field(default_factory=lambda: VQGANConfig(K=1024, D=256))
    dataset: str = "flower"
    batch_size: int = 128
    epochs: int = 1000
    lr: float = 4e-5
    schedule: str = "square"
    betas: tuple = (0.5, 0.9)
    log_interval: int = 10
    eval_interval: int = 150
    multi_gpu: bool = False

class TrainMaskGIT:
    def __init__(self, config: TrainMaskGITConfig):
        self.config = config
        self.master_process = torch.distributed.get_rank() == 0 if self.config.multi_gpu else True
        self.local_rank = int(os.environ["LOCAL_RANK"]) if self.config.multi_gpu else 0
        self.device = int(os.environ["LOCAL_RANK"]) if self.config.multi_gpu else torch.device(get_free_gpu())
        print(f"Master process: {self.master_process}")

        # 1. init models
        self.vqgan = VQGAN(self.config.vqgan_config).to(self.device).eval()
        self.vqgan.load_state_dict(torch.load(self.config.vqgan_path))

        self.gpt = GPTLanguageModel(self.config.gpt_config).to(self.device)
        if self.config.multi_gpu:
            self.gpt = DDP(self.gpt, device_ids=[self.local_rank])
            self.gpt_raw = self.gpt.module
        else:
            self.gpt_raw = self.gpt

        # 2. optimizers
        self.optim = self.configure_optimizers()

        # 3. dataset
        self.train_loader, self.test_loader = dataset_loaders[self.config.dataset](self.config.batch_size, self.config.multi_gpu)
        self.train_dataset, self.test_dataset = self.train_loader.dataset, self.test_loader.dataset

        # 4. run folder
        if self.master_process:
            run_name = f"maskgit-{self.config.dataset}-{time.time():.0f}"
            self.run_folder = Path("runs_maskgit") / run_name
            self.run_folder.mkdir(exist_ok=True, parents=True)
            wandb.init(project="maskgit", name=run_name, config={**self.config.__dict__})

        # 4. running variables
        self.steps = 0
        self.best_loss = float("inf")

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.gpt_raw.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"): no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules): decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules): no_decay.add(fpn)
        no_decay.add("position_embedding_table.weight")
        param_dict = {pn: p for pn, p in self.gpt_raw.named_parameters()}
        optim_groups = [{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}]
        return torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=self.config.betas)

    @torch.no_grad()
    def evaluate(self):
        val_loss = torch.tensor(0.0, device=self.device)
        bar = tqdm.tqdm(self.test_loader, desc="eval") if self.master_process else self.test_loader
        for x, y in bar:
            x, y = x.to(self.device), y.to(self.device)
            _, quantized, _ = self.vqgan(x)
            quantized = quantized.view(x.shape[0], -1)
            B, T = quantized.shape

            ratio = torch.rand((B,1), device=self.device)
            gamma_r = gamma_func(ratio, self.config.schedule)
            mask_count = (gamma_r * T).ceil().to(dtype=torch.int64)
            mask = torch.zeros((B, T), device=self.device).to(dtype=torch.int64)
            for b in range(B):
                indices = torch.randperm(T, device=self.device)[:mask_count[b]]
                mask[b, indices] = 1
            tokens_x = mask * self.config.vqgan_config.K + (1 - mask) * quantized
            tokens_y = quantized
            logits, _ = self.gpt(tokens_x)
            loss = F.cross_entropy(logits.view(B*T, -1), tokens_y.reshape(B*T))
            if self.master_process: bar.set_postfix(loss=f"{loss.item():.4f}")
            val_loss += loss

        if self.config.multi_gpu:
            torch.distributed.all_reduce(val_loss)
            val_loss /= torch.distributed.get_world_size()
        val_loss /= len(self.test_loader)

        if self.master_process:
            wandb.log({"val/loss": val_loss.item()})
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                # torch.save(self.gpt.state_dict(), self.run_folder / "best.pth")

    def train(self):
        for epoch in range(self.config.epochs):
            if self.config.multi_gpu: self.train_loader.sampler.set_epoch
            bar = tqdm.tqdm(self.train_loader) if self.master_process else self.train_loader
            for x, y in bar:
                x, y = x.to(self.device), y.to(self.device)

                # 1. get VQ-GAN quantized tokens
                with torch.no_grad(): 
                    _, quantized, _ = self.vqgan(x)
                quantized = quantized.view(x.shape[0], -1)
                B, T = quantized.shape

                # 2. sample mask for each input
                ratio = torch.ones((B,1), device=self.device) * torch.rand((1,), device=self.device)
                gamma_r = gamma_func(ratio, self.config.schedule)
                mask_count = (gamma_r * T).ceil().to(dtype=torch.int64)

                mask = torch.zeros((B, T), device=self.device).to(dtype=torch.int64)
                for b in range(B):
                    indices = torch.randperm(T, device=self.device)[:mask_count[b]]
                    mask[b, indices] = 1

                # 3. input & target tokens
                tokens_x = mask * self.config.vqgan_config.K + (1 - mask) * quantized
                tokens_y = quantized

                # 4. train
                logits, _ = self.gpt(tokens_x)
                loss = F.cross_entropy(logits.view(B*T, -1), tokens_y.reshape(B*T))
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                if self.config.multi_gpu:
                    torch.distributed.all_reduce(loss)
                    loss = loss / torch.distributed.get_world_size()
                # 5. log
                if self.master_process:
                    bar.set_postfix(loss=f"{loss.item():.4f}")
                    if self.steps % self.config.log_interval == 0:
                        wandb.log({"train/loss": loss.item()})

                if self.steps % self.config.eval_interval == 0:
                    self.evaluate()
                    if self.master_process:
                        self.generate_samples(self.run_folder / f"{self.steps}.jpg")
                        wandb.log({"Generated samples": [wandb.Image(str(self.run_folder / f"{self.steps}.jpg"))]})
                        self.generate_completions(self.run_folder / f"{self.steps}_train_comp.jpg", self.train_dataset)
                        wandb.log({"Train completions": [wandb.Image(str(self.run_folder / f"{self.steps}_train_comp.jpg"))]})
                        self.generate_completions(self.run_folder / f"{self.steps}_test_comp.jpg", self.test_dataset)
                        wandb.log({"Test completions": [wandb.Image(str(self.run_folder / f"{self.steps}_test_comp.jpg"))]})
                self.steps += 1

    @torch.inference_mode()
    def generate_samples(self, path):
        context = torch.ones((16,256), dtype=torch.long, device=self.device)*self.config.vqgan_config.K
        res = self.gpt_raw.generate_maskgit(context, steps=8)
        res[res >= self.config.vqgan_config.K] = self.config.vqgan_config.K-1
        images = denormalize(self.vqgan.decode(res))
        grid = torchvision.utils.make_grid(images, nrow=4).permute(1, 2, 0).cpu().detach().numpy() * 255
        grid_final = PIL.Image.fromarray(grid.astype("uint8"))
        grid_final.save(path)
    
    @torch.inference_mode()
    def generate_completions(self, path, dataset):
        idxs = torch.randint(0, len(dataset), (4,))
        images_gt = torch.stack([dataset[i][0] for i in idxs]).to(self.device)

        with torch.inference_mode():
            images_rec, quantized, _ = self.vqgan(images_gt.to(self.device))
        quantized = quantized.view(images_gt.shape[0], -1)
        B, T = quantized.shape

        grid_gt = torchvision.utils.make_grid(denormalize(images_gt), nrow=4)
        grid_rec = torchvision.utils.make_grid(denormalize(images_rec), nrow=4)
        grids = [grid_gt, grid_rec]

        # sample different masks
        for mask_ratio in [0.15, 0.3, 0.5, 0.75]:
            mask_ratio = torch.ones((B,1), device=self.device) * mask_ratio
            mask_count = (mask_ratio * T).ceil().to(dtype=torch.int64)
            mask = torch.zeros((B, T), device=self.device).to(dtype=torch.int64)
            for b in range(B):
                indices = torch.randperm(T, device=self.device)[:mask_count[b]]
                mask[b, indices] = 1
            tokens_x = mask * self.config.vqgan_config.K + (1 - mask) * quantized
            tokens_y = quantized
            logits, _ = self.gpt_raw(tokens_x)
            probs = F.softmax(logits, dim=-1)
            samples_vec = torch.ones((B, T), device=self.device, dtype=torch.int64)
            for i in range(B):
                samples_vec[i, :] = torch.multinomial(probs[i, :], num_samples=1).view(-1)
            samples_vec[samples_vec >= self.config.vqgan_config.K] = self.config.vqgan_config.K-1
            samples = samples_vec.view(B, 16, 16)
            images_comp = denormalize(self.vqgan.decode(samples))
            grids.append(torchvision.utils.make_grid(images_comp, nrow=4))

        grid = torch.cat(grids, dim=1).permute(1, 2, 0).cpu().detach().numpy() * 255
        grid = PIL.Image.fromarray(grid.astype("uint8"))
        grid.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--multi_gpu", action="store_true")
    args = parser.parse_args()

    if args.multi_gpu:
        init_process_group(backend="nccl")
    else:
        device = torch.device(get_free_gpu())

    vqgan_config = VQGANConfig(K=1024, D=256)
    gpt_config = GPTConfig(
            block_size=256, 
            vocab_size=1025, 
            n_embd=1024, 
            n_head=16, 
            n_layer=24,
            causal=False,
            dropout=0.0
    ) 
    # vqgan_path = "runs_vqgan/vqgan-imagenet-1725884613/imagenet_best.pth"
    # vqgan_path = f"runs_vqgan/vqgan-flower-1725870990/flower_best.pth"
    vqgan_path = f"runs_vqgan/vqgan-bird-1726056084/best.pth"
    train_config = TrainMaskGITConfig(
            gpt_config=gpt_config, 
            vqgan_config=vqgan_config, 
            vqgan_path=vqgan_path, 
            dataset=args.dataset, 
            lr=args.lr, 
            batch_size=args.batch_size,
            multi_gpu=args.multi_gpu
    )

    if args.multi_gpu: print(f"bs={train_config.batch_size*torch.distributed.get_world_size()}, per_gpu_bs={train_config.batch_size}")
    else: print(f"bs={train_config.batch_size}")

    train_maskgit = TrainMaskGIT(train_config)
    train_maskgit.train()
    destroy_process_group()

