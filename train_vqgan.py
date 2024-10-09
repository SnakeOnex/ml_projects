import torch, torch.nn as nn, torchvision, argparse, time, time, tqdm, PIL, wandb, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from vqgan import VQGAN, VQGAN_LFG, VQGANConfig, Discriminator
from model_configs import dataset_loaders
import matplotlib.pyplot as plt
from utils import get_free_gpu, denormalize
from lpips import LPIPS
from dataclasses import dataclass, field

device = torch.device(get_free_gpu())
print("selected device: ", device)

@dataclass
class TrainVQGANConfig:
    vqgan_config: VQGANConfig = field(default_factory=lambda: VQGANConfig(K=1024, D=256))
    dataset: str = "flower"
    batch_size: int = 16
    epochs: int = 1000
    lr: float = 4e-5
    betas: tuple = (0.5, 0.9)
    eps: float = 1e-8
    disc_factor: float = 0.2
    disc_kickoff: int = 100_000
    log_interval: int = 10
    eval_interval: int = 500

@torch.no_grad()
def plot_results(vqgan, dataset, path, idxs=None):
    if idxs is None:
        idxs = torch.randint(0, len(dataset), (16,))

    images_gt = torch.stack([dataset[i][0] for i in idxs])

    with torch.inference_mode():
        # images_pred = vqgan(images_gt.to(device))["output"]
        images_pred, _, _ = vqgan(images_gt.to(device))
    images_pred = images_pred.cpu().detach()

    images_gt = denormalize(images_gt)
    images_pred = denormalize(images_pred)

    grid_pred = torchvision.utils.make_grid(images_pred, nrow=4)
    grid = torchvision.utils.make_grid(images_gt, nrow=4)
    grid_final = torch.cat([grid, grid_pred], dim=2)

    grid_final = grid_final.permute(1, 2, 0)

    grid_final = grid_final.numpy()
    grid_final = (grid_final * 255).astype("uint8")
    grid_final = PIL.Image.fromarray(grid_final)

    grid_final.save(f"{path}")

class TrainVQGAN:
    def __init__(self, config: TrainVQGANConfig):
        self.config = config

        # 1. init models
        # self.vqgan = VQGAN(self.config.vqgan_config).to(device)
        self.vqgan = VQGAN_LFG(self.config.vqgan_config).to(device)
        self.discriminator = Discriminator(self.config.vqgan_config).to(device)

        # 2. losses & optimizers
        self.perceptual_loss_fn = LPIPS().eval().to(device)
        self.optim_vqgan, self.optim_disc = self.configure_optimizers()

        # 3. dataset
        self.train_loader, self.test_loader = dataset_loaders[self.config.dataset](self.config.batch_size)
        self.train_dataset, self.test_dataset = self.train_loader.dataset, self.test_loader.dataset

        # 4. run folder
        run_name = f"vqgan-{self.config.dataset}-{time.time():.0f}"
        self.run_folder = Path("runs_vqgan") / run_name
        self.run_folder.mkdir(exist_ok=True, parents=True)

        wandb.init(project="my_vqgan",
                   name=run_name,
                   config={**self.config.__dict__})

        # 5. running variables
        self.steps = 0
        self.best_loss = float("inf")

    def configure_optimizers(self):
        optim_vqgan = torch.optim.Adam(
                self.vqgan.parameters(), 
                lr=self.config.lr,
                eps=self.config.eps,
                betas=self.config.betas,
                )
        optim_disc = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.lr,
                eps=self.config.eps,
                betas=self.config.betas)
        return optim_vqgan, optim_disc

    @torch.no_grad()
    def evaluate(self):
        val_l, q_l, recon_l, cnt = 0, 0, 0, 0
        for x, _ in tqdm.tqdm(self.test_loader):
            x = x.to(device)
            out, _, quantize_loss = self.vqgan(x)

            mse_loss = abs(out - x)
            percep_loss = self.perceptual_loss_fn(out, x)
            recon_loss = (mse_loss + percep_loss).mean()

            val_l += (quantize_loss + recon_loss).item()
            q_l += quantize_loss.item()
            recon_l += recon_loss.item()
            cnt += 1
        wandb.log({"valid/loss": val_l/cnt, "valid/quantize_loss": q_l/cnt, "valid/recon_loss": recon_l/cnt})

        if self.best_loss > val_l/cnt:
            self.best_loss = val_l/cnt
            torch.save(self.vqgan.state_dict(), self.run_folder / "best.pth")
        if self.steps % 1000 == 0:
            torch.save(self.vqgan.state_dict(), self.run_folder / f"model_{self.steps}.pth")

    def train(self):
        for epoch in range(self.config.epochs):
            util = set()
            bar = tqdm.tqdm(self.train_loader)
            for x, _ in bar:
                x = x.to(device)


                # 1. discriminator step
                self.discriminator.zero_grad()
                self.optim_disc.zero_grad()

                with torch.no_grad():
                    out, _, _ = self.vqgan(x)

                real_pred = self.discriminator(x)
                fake_pred = self.discriminator(out.detach())
                d_loss_real = torch.mean(F.relu(1.0 - real_pred))
                d_loss_fake = torch.mean(F.relu(1.0 + fake_pred))

                disc_factor = self.config.disc_factor if self.steps > self.config.disc_kickoff else 0.0
                gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                gan_loss.backward()
                self.optim_disc.step()

                # 2. vqgan step
                self.optim_vqgan.zero_grad()
                self.vqgan.zero_grad()

                out, indices, quantize_loss = self.vqgan(x)
                l1_loss = abs(out - x)
                percep_loss = self.perceptual_loss_fn(out, x)
                recon_loss = (l1_loss + percep_loss).mean()
                train_loss = quantize_loss + recon_loss

                fake_pred = self.discriminator(out)
                g_loss = -torch.mean(fake_pred)

                # calculate lambda
                lamb = self.vqgan.calculate_lambda(recon_loss, g_loss)
                train_loss += disc_factor * lamb * g_loss

                util.update(indices.view(-1).cpu().detach().numpy().tolist())

                train_loss.backward()
                self.optim_vqgan.step()

                if self.steps % self.config.log_interval == 0:
                    wandb.log({"train/loss": train_loss.item(),
                               "train/recon_loss": recon_loss.item(),
                               "train/quantize_loss": quantize_loss.item(),
                               "gan_loss": gan_loss.item(),
                               "g_loss": g_loss.item(),
                               "lambda": lamb.item(),
                               "disc_factor": disc_factor})

                if self.steps % self.config.eval_interval == 0:
                    self.evaluate()
                    plot_results(self.vqgan, self.test_dataset, path=self.run_folder / f"test_{self.config.dataset}_{self.steps}.jpg", idxs=None)
                    wandb.log({"test_reconstruction": [wandb.Image(str(self.run_folder / f"test_{self.config.dataset}_{self.steps}.jpg"))]})
                    plot_results(self.vqgan, self.train_dataset, path=self.run_folder / f"train_{self.config.dataset}_{self.steps}.jpg", idxs=None)
                    wandb.log({"train_reconstruction": [wandb.Image(str(self.run_folder / f"train_{self.config.dataset}_{self.steps}.jpg"))]})

                self.steps += 1

            wandb.log({"util": len(util)/self.config.vqgan_config.K, "epoch": epoch})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    args = parser.parse_args()

    bs, lr, epochs = 8, 4e-5, 1000

    train_config = TrainVQGANConfig(dataset=args.dataset,
                                    batch_size=bs,
                                    epochs=epochs,
                                    lr=lr,)


    train_vqgan = TrainVQGAN(train_config)
    train_vqgan.train()
