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
        val_l, mse_l, percept_l, q_l, cnt = 0, 0, 0, 0, 0
        for x, _ in loader:
            x = x.to(device)
            out  = model(x)

            mse_loss = mse_loss_fn(out["output"], x)
            perceptual_loss = perceptual_loss_fn(out["output"], x)
            quantize_loss = out["quantize_loss"]

            val_l += mse_loss + perceptual_loss + quantize_loss
            mse_l += mse_loss
            percept_l += perceptual_loss
            q_l += quantize_loss
            cnt += 1

    return {"valid_loss": val_l/cnt, "valid_mse_loss": mse_l/cnt, "valid_perceptual_loss": percept_l/cnt, "valid_quantize_loss": q_l/cnt}

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
    bs, lr, epochs = 64, 1e-4, 1000

    wandb.init(project="vqvae",
               name=run_name,
               config={"dataset": args.dataset, 
                       "batch_size": bs, 
                       "epochs": epochs,
                       "lr": lr,
                       "vqvae_config": vqvae_config.__dict__,
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

    model = VQVAE(vqvae_config).to(device)
    wandb.watch(model)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    mse_loss_fn = nn.MSELoss()
    perceptual_loss_fn = PerceptualLoss().to(device)

    with torch.no_grad(): model(torch.randn(1, vqvae_config.in_channels, vqvae_config.image_sz, vqvae_config.image_sz).to(device), verbose=True)

    best_loss = float("inf")
    best_model = None

    for epoch in range(epochs):
        trn_l, mse_l, percept_l, q_l, cnt, util = 0, 0, 0, 0, 0, set()
        st = time.time()
        for x, _ in tqdm.tqdm(train_loader):
            x = x.to(device)
            optim.zero_grad()
            out = model(x)

            perceptual_loss = perceptual_loss_fn(out["output"], x)
            mse_loss = mse_loss_fn(out["output"], x)
            quantize_loss = out["quantize_loss"]
            train_loss = perceptual_loss + mse_loss + quantize_loss

            trn_l += train_loss; mse_l += mse_loss; percept_l += perceptual_loss; q_l += quantize_loss; cnt += 1
            util.update(out["closest"].view(-1).cpu().detach().numpy().tolist())

            train_loss.backward()
            optim.step()
        trn_l /= cnt; mse_l /= cnt; percept_l /= cnt; q_l /= cnt

        val_res = eval_vqvae(model, test_loader)
        val_l, val_mse_l, val_percept_l, val_q_l = val_res["valid_loss"], val_res["valid_mse_loss"], val_res["valid_perceptual_loss"], val_res["valid_quantize_loss"]
        wandb.log({"train_loss": trn_l,
                   "train_mse_loss": mse_l,
                   "train_perceptual_loss": percept_l,
                   "train_quantize_loss": q_l,
                   **val_res,
                   "util": len(util)/vqvae_config.K,})

        if epoch % 10 == 0:
            plot_results(model, test_dataset, config["stats"], path=run_folder / f"{args.dataset}_{epoch}.png", idxs=idxs)
            wandb.save(run_folder / f"{args.dataset}_{epoch}.png")

        if val_l < best_loss:
            best_loss = val_l
            torch.save(model.state_dict(), run_folder / f"{args.dataset}_best.pth")

        # print(f"e={epoch:2}, trn_r_l={r_loss:.4f}, val_r_l={val_r_loss:.4f}, trn_q_l={q_loss:.4f}, val_q_l={val_q_loss:.4f}, t={(time.time() - st):.2f} s")
        print(f"e={epoch:2}, trn_l={trn_l:.4f} val_l={val_l:.4f}, trn_mse_l={mse_l:.4f}, val_mse_l={val_mse_l:.4f}, trn_percept_l={percept_l:.4f}, val_percept_l={val_percept_l:.4f}, trn_q_l={q_l:.4f}, val_q_l={val_q_l:.4f}, util={len(util)/vqvae_config.K:.2f}")

    model.load_state_dict(torch.load(run_folder / f"{args.dataset}_best.pth"))
    plot_results(model, test_dataset, config["stats"], path=f"{args.dataset}_random_sample.png")
