import torch, torch.nn as nn, torchvision, argparse, time, pathlib, time, tqdm, PIL
from torch.utils.data import DataLoader
from vqvae import VQVAE
from model_configs import model_configs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_vqvae(model, loader):
    with torch.no_grad():
        r_loss = 0
        q_loss = 0
        count = 0
        for x, _ in loader:
            x = x.to(device)
            out, _, loss = model(x)

            reconstruct_loss = crit(out, x)
            r_loss += reconstruct_loss
            q_loss += loss
            count += 1
            loss = reconstruct_loss + loss
    return r_loss/count, q_loss/count

def plot_results(model, dataset, image_count, path="vqvae.png", idxs=None):
    if idxs is None:
        idxs = torch.randint(0, len(dataset), (16,))

    images_gt = torch.stack([dataset[i][0] for i in idxs])

    with torch.inference_mode():
        images_pred, quantized, _ = model(images_gt.to(device))
    images_pred = images_pred.cpu().detach()

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

    dataset = args.dataset

    config = model_configs[dataset]

    print(f"training VQ-VAE on the {dataset} dataset")

    train_dataset, test_dataset = config["fetch_train"](), config["fetch_test"]()

    print(f"train_sz={len(train_dataset)}, test_sz={len(test_dataset)}")
    print(f"K={config['K']}, D={config['D']}")

    pathlib.Path("results").mkdir(parents=True, exist_ok=True)

    idxs = torch.randint(0, len(test_dataset), (16,))
    train_loader = DataLoader(
            train_dataset, 
            batch_size=128, 
            shuffle=True, 
            num_workers=4, 
            prefetch_factor=2, 
            pin_memory=True,
            persistent_workers=True
    )
    test_loader = DataLoader(
            test_dataset, 
            batch_size=128, 
            shuffle=False, 
            num_workers=4, 
            prefetch_factor=2, 
            pin_memory=True,
            persistent_workers=True
    )

    model = VQVAE(config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.MSELoss()

    for epoch in range(100):
        r_loss = 0
        q_loss = 0
        count = 0
        st = time.time()
        for x, _ in tqdm.tqdm(train_loader):
        # for x, _ in train_loader:
            x = x.to(device)
            optim.zero_grad()
            out, _, loss = model(x)

            reconstruct_loss = crit(out, x)
            r_loss += reconstruct_loss
            q_loss += loss
            count += 1
            loss = 2 * reconstruct_loss + loss

            loss.backward()
            optim.step()
        r_loss /= count
        q_loss /= count
        val_r_loss, val_q_loss = eval_vqvae(model, test_loader)
        print(f"e={epoch:2}, trn_r_l={r_loss:.4f}, val_r_l={val_r_loss:.4f}, trn_q_l={q_loss:.4f}, val_q_l={val_q_loss:.4f}, t={(time.time() - st):.2f} s")

        if epoch % 1 == 0:
            plot_results(model, test_dataset, 16, path=f"results/vqvae_{epoch}.png", idxs=idxs)

    # save model
    torch.save(model.state_dict(), "vqvae.pth")

    model.load_state_dict(torch.load("vqvae.pth"))

    plot_results(model, test_dataset, 16, path=f"random_sample.png")
