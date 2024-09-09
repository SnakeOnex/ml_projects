import torch, torch.nn as nn, torchvision, argparse, time, time, tqdm, PIL, wandb, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from vqgan import VQGAN, Discriminator
from model_configs import model_configs
import matplotlib.pyplot as plt
from utils import get_free_gpu, denormalize
from lpips import LPIPS

device = torch.device(get_free_gpu())
print("selected device: ", device)

@torch.no_grad()
def eval_vqgan(model, loader):
    with torch.no_grad():
        val_l, q_l, recon_l, cnt = 0, 0, 0, 0
        for x, _ in loader:
            x = x.to(device)
            out, _, quantize_loss = model(x)

            mse_loss = abs(out - x)
            percep_loss = perceptual_loss_fn(out, x)
            recon_loss = (mse_loss + percep_loss).mean()

            val_l += (quantize_loss + recon_loss).item()
            q_l += quantize_loss.item()
            recon_l += recon_loss.item()
            cnt += 1

    return {"valid/loss": val_l/cnt, "valid/quantize_loss": q_l/cnt, "valid/recon_loss": recon_l/cnt}

@torch.no_grad()
def plot_results(model, dataset, path, idxs=None):
    if idxs is None:
        idxs = torch.randint(0, len(dataset), (16,))

    images_gt = torch.stack([dataset[i][0] for i in idxs])

    with torch.inference_mode():
        # images_pred = model(images_gt.to(device))["output"]
        images_pred, _, _ = model(images_gt.to(device))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    args = parser.parse_args()

    run_name = f"vqgan-{args.dataset}-{time.time():.0f}"
    run_folder = Path("runs_vqgan") / run_name
    run_folder.mkdir(exist_ok=True, parents=True)
    print("running: ", run_name)

    config = model_configs[args.dataset]
    vqgan_config = config["vqgan_config"]
    bs, lr, epochs = 16, 4e-5, 1000

    wandb.init(project="my_vqgan",
               name=run_name,
               config={"dataset": args.dataset, 
                       "batch_size": bs, 
                       "epochs": epochs,
                       "lr": lr,
                       "vqgan_config": vqgan_config.__dict__,
                       })

    train_dataset, test_dataset = config["fetch_train"](), config["fetch_test"]()

    print(f"train_sz={len(train_dataset)}, test_sz={len(test_dataset)}")
    print(f"K={vqgan_config.K}, D={vqgan_config.D}")


    idxs = torch.randint(0, len(test_dataset), (16,))

    Path("checkpoints").mkdir(exist_ok=True)

    train_loader = DataLoader(
            train_dataset, 
            batch_size=bs, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
    )
    test_loader = DataLoader(
            test_dataset, 
            batch_size=bs, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
    )

    model = VQGAN(vqgan_config).to(device)
    discriminator = Discriminator(vqgan_config).to(device)

    optim = torch.optim.Adam(
            model.parameters(), 
            lr=lr,
            eps=1e-8,
            betas=(0.5, 0.9),
            )

    optim_disc = torch.optim.Adam(
            discriminator.parameters(),
            lr=lr,
            eps=1e-8,
            betas=(0.5, 0.9))

    amp_enabled = True
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    mse_loss_fn = nn.MSELoss()
    perceptual_loss_fn = LPIPS().eval().to(device)

    best_loss = float("inf")
    best_model = None

    val_res = eval_vqgan(model, test_loader)
    wandb.log({**val_res})

    step = 0
    for epoch in range(epochs):
        trn_l, mse_l, recon_l, percep_l, q_l, cnt, util = 0, 0, 0, 0, 0, 0, set()
        st = time.time()

        st = time.perf_counter()
        for x, _ in tqdm.tqdm(train_loader):
            discriminator.zero_grad()
            optim_disc.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                x = x.to(device)

                with torch.no_grad():
                    out, _, _ = model(x)

                real_pred = discriminator(x)
                fake_pred = discriminator(out.detach())
                d_loss_real = torch.mean(F.relu(1.0 - real_pred))
                d_loss_fake = torch.mean(F.relu(1.0 + fake_pred))

                curr_step = epoch * len(train_loader) + cnt
                disc_factor = 0.2 if curr_step > 25_000 else 0.0
                gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

            scaler.scale(gan_loss).backward()
            scaler.step(optim_disc)
            scaler.update()

            optim.zero_grad()
            model.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                out, indices, quantize_loss = model(x)
                mse_loss = abs(out - x)
                percep_loss = perceptual_loss_fn(out, x)
                recon_loss = (mse_loss + percep_loss).mean()
                train_loss = quantize_loss + recon_loss

                fake_pred = discriminator(out)
                g_loss = -torch.mean(fake_pred)

                # calculate lambda
                lamb = model.calculate_lambda(recon_loss, g_loss)
                train_loss += disc_factor * lamb * g_loss

                trn_l += train_loss.item(); mse_l += mse_loss.mean().item(); q_l += quantize_loss.item(); cnt += 1; percep_l += percep_loss.mean().item()
                recon_l += recon_loss.item()
                util.update(indices.view(-1).cpu().detach().numpy().tolist())

                scaler.scale(train_loss).backward()
                scaler.step(optim)
                scaler.update()

            if step % 5 == 0:
                wandb.log({"train/loss": train_loss.item(),
                           "train/recon_loss": recon_loss.item(),
                           "train/quantize_loss": quantize_loss.item(),
                           "gan_loss": gan_loss.item(),
                           "g_loss": g_loss.item(),
                           "lambda": lamb.item(),
                           "disc_factor": disc_factor,
                           "util": len(util)/vqgan_config.K})

            if step % 100 == 0:
                val_res = eval_vqgan(model, test_loader)
                wandb.log({**val_res})
                plot_results(model, test_dataset, path=run_folder / f"test_{args.dataset}_{step}.jpg", idxs=idxs)
                wandb.log({"test_reconstruction": [wandb.Image(str(run_folder / f"test_{args.dataset}_{step}.jpg"))]})
                plot_results(model, train_dataset, path=run_folder / f"train_{args.dataset}_{step}.jpg", idxs=idxs)
                wandb.log({"train_reconstruction": [wandb.Image(str(run_folder / f"train_{args.dataset}_{step}.jpg"))]})

            step += 1
        et = time.perf_counter()

        val_res = eval_vqgan(model, test_loader)
        wandb.log({**val_res, "epoch":epoch})

        if val_res["valid/loss"] < best_loss:
            best_loss = val_res["valid/loss"]
            torch.save(model.state_dict(), run_folder / f"{args.dataset}_best.pth")
