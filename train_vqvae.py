import torch, torch.nn as nn, torchvision, argparse, time, time, tqdm, PIL, wandb
from pathlib import Path
from torch.utils.data import DataLoader
from vqvae import VQVAE, PerceptualLoss
from model_configs import model_configs
import matplotlib.pyplot as plt
from utils import get_free_gpu, denormalize

device = torch.device(get_free_gpu())
print("selected device: ", device)

def eval_vqvae(model, loader):
    with torch.no_grad():
        r_loss, q_loss, count = 0, 0, 0
        for x, _ in loader:
            x = x.to(device)
            out  = model(x)

            reconstruct_loss = crit(out["output"], x)
            r_loss += reconstruct_loss
            q_loss += out["quantize_loss"]
            count += 1
    return r_loss/count, q_loss/count

def plot_results(model, dataset, stats, path="vqvae.png", idxs=None):
    if idxs is None:
        idxs = torch.randint(0, len(dataset), (16,))

    images_gt = torch.stack([dataset[i][0] for i in idxs])

    with torch.inference_mode():
        images_pred = model(images_gt.to(device))["output"]
    images_pred = images_pred.cpu().detach()

    images_gt = denormalize(images_gt, stats)
    images_pred = denormalize(images_pred, stats)

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

    run_name = f"vqvae-{args.dataset}-{time.time():.0f}"
    run_folder = Path("runs_vqvae") / run_name
    run_folder.mkdir(exist_ok=True, parents=True)
    print("running: ", run_name)

    config = model_configs[args.dataset]
    vqvae_config = config["vqvae_config"]
    bs, lr, epochs = 16, 3e-4, 1000

    wandb.init(project="vqvae",
               name=run_name,
               config={"dataset": args.dataset, 
                       "batch_size": bs, 
                       "epochs": epochs,
                       "lr": lr,
                       "vqvae_config": vqvae_config.__dict__
                       })

    train_dataset, test_dataset = config["fetch_train"](), config["fetch_test"]()

    print(f"train_sz={len(train_dataset)}, test_sz={len(test_dataset)}")
    print(f"K={vqvae_config.K}, D={vqvae_config.D}")


    idxs = torch.randint(0, len(test_dataset), (16,))

    Path("checkpoints").mkdir(exist_ok=True)

    train_loader = DataLoader(
            train_dataset, 
            batch_size=bs, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True
    )
    test_loader = DataLoader(
            test_dataset, 
            batch_size=bs, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True
    )

    model = VQVAE(vqvae_config).to(device)
    wandb.watch(model)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    crit = PerceptualLoss().to(device)

    with torch.no_grad(): model(torch.randn(1, vqvae_config.in_channels, vqvae_config.image_sz, vqvae_config.image_sz).to(device), verbose=True)

    best_loss = float("inf")
    best_model = None

    for epoch in range(epochs):
        r_loss, q_loss, count = 0, 0, 0
        st = time.time()
        for x, _ in tqdm.tqdm(train_loader):
            x = x.to(device)
            optim.zero_grad()
            out = model(x)

            reconstruct_loss = crit(out["output"], x)
            r_loss += reconstruct_loss
            q_loss += out["quantize_loss"]
            count += 1
            loss = reconstruct_loss + out["quantize_loss"]

            loss.backward()
            optim.step()
        r_loss /= count
        q_loss /= count
        val_r_loss, val_q_loss = eval_vqvae(model, test_loader)
        wandb.log({"train_loss": r_loss + q_loss,
                  "val_loss": val_r_loss + val_q_loss,
                  "train_reconstruction_loss": r_loss, 
                   "train_quantize_loss": q_loss, 
                   "val_reconstruction_loss": val_r_loss, 
                   "val_quantize_loss": val_q_loss})

        if epoch % 10 == 0:
            plot_results(model, test_dataset, config["stats"], path=run_folder / f"{args.dataset}_{epoch}.png", idxs=idxs)
            wandb.save(run_folder / f"{args.dataset}_{epoch}.png")

        if val_r_loss + val_q_loss < best_loss:
            best_loss = val_r_loss + val_q_loss
            torch.save(model.state_dict(), run_folder / f"{args.dataset}_best.pth")

        print(f"e={epoch:2}, trn_r_l={r_loss:.4f}, val_r_l={val_r_loss:.4f}, trn_q_l={q_loss:.4f}, val_q_l={val_q_loss:.4f}, t={(time.time() - st):.2f} s")

    model.load_state_dict(torch.load(run_folder / f"{args.dataset}_best.pth"))
    plot_results(model, test_dataset, config["stats"], path=f"{args.dataset}_random_sample.png")
