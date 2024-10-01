import argparse, time, tqdm, PIL, wandb, numpy as np, os
import torch, torch.nn as nn, torchvision
from pathlib import Path
from vqgan import VQGAN, VQGANConfig
from model_configs import dataset_loaders
from gpt import GPTLanguageModel, GPTConfig
from gpt_llama import GPT_B, GPT_L, GPT_XL
from utils import get_free_gpu, denormalize
from dataclasses import dataclass, field
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

@dataclass
class TrainGPTConfig:
    gpt_config: GPTConfig
    vqgan_path: str
    vqgan_config: VQGANConfig = field(default_factory=lambda: VQGANConfig(K=1024, D=256))
    dataset: str = "flower"
    batch_size: int = 16
    per_gpu_bs: int = 16
    accum_steps: int = 1
    epochs: int = 100
    lr: float = 6e-5
    warmup_steps: int = 5_000
    max_steps: int = 200_000
    betas: tuple = (0.9, 0.95)
    log_interval: int = 10
    eval_interval: int = 2000
    class_cond: bool = False
    multi_gpu: bool = False

    def tokens_per_batch(self):
        return self.per_gpu_bs * self.gpt_config.block_size * (torch.distributed.get_world_size() if self.multi_gpu else 1)

class TrainGPT:
    def __init__(self, config: TrainGPTConfig):
        self.config = config
        self.master_process = torch.distributed.get_rank() == 0 if self.config.multi_gpu else True
        self.local_rank = int(os.environ["LOCAL_RANK"]) if self.config.multi_gpu else 0
        self.device = int(os.environ["LOCAL_RANK"]) if self.config.multi_gpu else torch.device(get_free_gpu())

        # 1. init models
        self.vqgan = VQGAN(self.config.vqgan_config).to(self.device).eval()
        self.vqgan.load_state_dict(torch.load(self.config.vqgan_path, weights_only=True))

        # self.gpt = GPTLanguageModel(self.config.gpt_config).to(self.device)
        self.gpt = GPT_L().to(self.device)
        # self.gpt.setup_caches(max_batch_size=16, max_seq_length=257, dtype=self.gpt.tok_embeddings.weight.dtype)
        if self.config.multi_gpu:
            self.gpt = DDP(self.gpt, device_ids=[self.local_rank])
            self.gpt.load_state_dict(torch.load("runs_gpt/gpt-imagenet-1727429496/best.pth"))
            self.gpt_raw = self.gpt.module
        else:
            self.gpt_raw = self.gpt

        # 2. optimizers
        self.optim = self.configure_optimizers()
        self.cos_lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.config.max_steps, eta_min=self.config.lr / 10)
        self.warmup_sched = torch.optim.lr_scheduler.LambdaLR(self.optim, lambda s: min(1, s / self.config.warmup_steps))
        self.lr_sched = torch.optim.lr_scheduler.SequentialLR(self.optim, [self.warmup_sched, self.cos_lr_sched], [self.config.warmup_steps])

        # 3. dataset
        self.train_loader, self.test_loader = dataset_loaders[self.config.dataset](self.config.per_gpu_bs, self.config.multi_gpu)
        self.train_dataset, self.test_dataset = self.train_loader.dataset, self.test_loader.dataset

        # 4. run folder
        if self.master_process:
            run_name = f"gpt-{self.config.dataset}-{time.time():.0f}"
            self.run_folder = Path("runs_gpt") / run_name
            self.run_folder.mkdir(exist_ok=True, parents=True)
            wandb.init(project="gpt-vqgan", name=run_name, config={**self.config.__dict__})

        # 5. running variables
        self.steps = 0
        self.best_loss = float("inf")

    def configure_optimizers(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.gpt_raw.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': 0.05},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=self.config.betas, weight_decay=0.05)

    @torch.no_grad()
    def evaluate(self):
        val_loss = torch.tensor(0.0, device=self.device)
        bar = tqdm.tqdm(self.test_loader, desc="eval") if self.master_process else self.test_loader
        for x, y in bar:
            x, y = x.to(self.device), y.to(self.device)
            _, quantized, _ = self.vqgan(x)
            image_tokens = quantized.view(x.shape[0], -1)
            tokens_x = image_tokens[:,0:self.config.gpt_config.block_size-1].clone()
            cls_token = y.view((-1,)).clone()
            tokens_y = image_tokens[:,0:self.config.gpt_config.block_size].clone()
            _, loss = self.gpt(
                    idx=tokens_x,
                    cond_idx=cls_token,
                    targets=tokens_y)
            if self.master_process: bar.set_postfix(loss=f"{loss.item():.4f}")
            val_loss += loss.item()

        if self.config.multi_gpu:
            torch.distributed.all_reduce(val_loss)
            val_loss /= torch.distributed.get_world_size()
        val_loss /= len(self.test_loader)

        if self.master_process:
            wandb.log({"val/loss": val_loss})
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.gpt.state_dict(), self.run_folder / "best.pth")

    def train(self):
        for epoch in range(self.config.epochs):
            if self.config.multi_gpu: self.train_loader.sampler.set_epoch
            bar = tqdm.tqdm(self.train_loader) if self.master_process else self.train_loader
            micro_step = 0
            for x, y in bar:
                if micro_step == 0: st = time.time()
                if self.config.multi_gpu: self.gpt.require_backward_grad_sync = micro_step == self.config.accum_steps-1
                x, y = x.to(self.device), y.to(self.device)

                # 1. get VQ-GAN quantized tokens
                with torch.no_grad(): 
                    _, quantized, _ = self.vqgan(x)
                image_tokens = quantized.view(x.shape[0], -1)

                # 2. add sos / class token
                tokens_x = image_tokens[:,0:self.config.gpt_config.block_size-1].clone()
                cls_token = y.view((-1,)).clone()
                tokens_y = image_tokens[:,0:self.config.gpt_config.block_size].clone()

                # 4. train
                _, loss = self.gpt(
                        idx=tokens_x,
                        cond_idx=cls_token,
                        targets=tokens_y)
                            
                loss = loss / self.config.accum_steps
                loss.backward()
                micro_step += 1
                if micro_step != self.config.accum_steps: continue
                else: micro_step = 0
                self.optim.step()
                self.lr_sched.step()
                self.optim.zero_grad()

                if self.config.multi_gpu:
                    torch.distributed.all_reduce(loss)
                    loss = loss / torch.distributed.get_world_size()
                # 5. log
                if self.master_process:
                    loss *= self.config.accum_steps
                    bar.set_postfix(loss=f"{loss.item():.4f}", fps=f"{1/(time.time()-st):.2f}s")
                    if self.steps % self.config.log_interval == 0:
                        wandb.log({"train/loss": loss.item(),
                                   "lr": self.optim.param_groups[0]["lr"],
                                   "steps": self.steps,
                                   "epoch": epoch,
                                   "step_time": time.time()-st,
                                   "tokens_per_second": self.config.tokens_per_batch() / (time.time()-st)})

                if self.steps % self.config.eval_interval == 0 and self.steps >= self.config.warmup_steps:
                    self.evaluate()
                    images = self.compute_metrics()
                    if self.master_process:
                        self.generate_samples(self.run_folder / f"{self.steps}.jpg", images[:16,...])
                        wandb.log({"Generated samples": [wandb.Image(str(self.run_folder / f"{self.steps}.jpg"))]})
                        self.generate_completions(self.run_folder / f"{self.steps}_comp.jpg")
                        wandb.log({"Completions": [wandb.Image(str(self.run_folder / f"{self.steps}_comp.jpg"))]})
                self.steps += 1

    @torch.inference_mode()
    def compute_metrics(self):
        # inference_gpt = GPT_B().to(self.device)
        # inference_gpt.load_state_dict(self.gpt_raw.state_dict())
        # inference_gpt.eval()
        # with torch.device(self.device):
            # inference_gpt.setup_caches(max_batch_size=16, max_seq_length=256, dtype=inference_gpt.tok_embeddings.weight.dtype)

        inference_gpt = self.gpt_raw
        incep_score = InceptionScore(normalize=False).to(self.device)
        context = torch.zeros((16,1), dtype=torch.long, device=self.device)
        if self.config.class_cond: context += 7 # set class to always be cock
        res = inference_gpt.generate(context, 256)[:,1:]
        res[res >= self.config.vqgan_config.K] = self.config.vqgan_config.K-1
        res[res < 0] = 0
        images = (denormalize(self.vqgan.decode(res)) * 255).to(dtype=torch.uint8)
        incep_score.update(images)
        incep_score_val = incep_score.compute()[0].item()
        print(f"Inception score: {incep_score_val}")
        if self.master_process: wandb.log({"val/inception_score": float(incep_score_val)})
        return images # return images to not have to generate them again for visualization

    @torch.inference_mode()
    def generate_samples(self, path, images=None):
        if images is None:
            context = torch.ones((16,1), dtype=torch.long, device=self.device)*self.config.vqgan_config.K
            self.gpt_raw.setup_caches(max_batch_size=16, max_seq_length=258, dtype=self.gpt_raw.tok_embeddings.weight.dtype) res = self.gpt_raw.generate(context, 256)[:,1:] - 1000 res[res >= self.config.vqgan_config.K] = self.config.vqgan_config.K-1 res[res < 0] = 0
            images = denormalize(self.vqgan.decode(res)).to(dtype=torch.uint8) * 255
        grid = torchvision.utils.make_grid(images, nrow=4).permute(1, 2, 0).cpu().detach().numpy()
        grid_final = PIL.Image.fromarray(grid)
        grid_final.save(path)
    
    @torch.inference_mode()
    def generate_completions(self, path):
        idxs = torch.randint(0, len(self.test_dataset), (4,))
        images_gt = torch.stack([self.test_dataset[i][0] for i in idxs]).to(self.device)
        cls_gt = torch.stack([torch.tensor(self.test_dataset[i][1]) for i in idxs]).to(self.device)

        with torch.inference_mode():
            images_rec, quantized, _ = self.vqgan(images_gt.to(self.device))
        quantized = quantized.view(images_gt.shape[0], -1)[:,:128]
        cond_tokens = cls_gt.view((-1,1))
        image_tokens = torch.cat([cond_tokens, quantized], dim=1)
        res = self.gpt_raw.generate(image_tokens, 128)[:,1:]
        res[res >= self.config.vqgan_config.K] = self.config.vqgan_config.K-1
        res[res < 0] = 0
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--multi_gpu", action="store_true")
    args = parser.parse_args()

    max_gpu_bs = 32
    accum_steps = 1

    if args.multi_gpu:
        init_process_group(backend="nccl")
        assert args.batch_size % torch.distributed.get_world_size() == 0
        per_gpu_bs = args.batch_size // torch.distributed.get_world_size()
    else:
        per_gpu_bs = args.batch_size

    if per_gpu_bs > max_gpu_bs:
        assert per_gpu_bs % max_gpu_bs == 0
        accum_steps = per_gpu_bs // max_gpu_bs
        per_gpu_bs = max_gpu_bs

    print(f"bs={args.batch_size}, per_gpu_bs={per_gpu_bs}, accum_steps={accum_steps}")

    vqgan_config = VQGANConfig(K=1024, D=256)
    gpt_config = GPTConfig(
            block_size=256,
            vocab_size=2048,
            n_embd=1024,
            n_head=16,
            n_layer=24,
            causal=True,
            dropout=0.1,
    )

    if args.dataset == "imagenet":
        vqgan_path = "runs_vqgan/vqgan-imagenet-1726089582/best.pth"
    elif args.dataset == "flower":
        vqgan_path = "runs_vqgan/vqgan-flower-1726089427/best.pth"
    elif args.dataset == "bird":
        vqgan_path = f"runs_vqgan/vqgan-bird-1726056084/best.pth"

    train_config = TrainGPTConfig(
            gpt_config=gpt_config, 
            vqgan_config=vqgan_config, 
            vqgan_path=vqgan_path, 
            dataset=args.dataset, 
            lr=args.lr,
            batch_size=args.batch_size,
            per_gpu_bs=per_gpu_bs,
            accum_steps=accum_steps,
            class_cond=True,
            multi_gpu=args.multi_gpu)

    trainer = TrainGPT(train_config)
    trainer.train()
    destroy_process_group()

