import argparse, time, time, tqdm, PIL, wandb, numpy as np, pickle
import torch, torch.nn as nn, torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from vqvae import VQVAE
from model_configs import model_configs
from gpt import GPTLanguageModel
from utils import get_free_gpu, denormalize

device = torch.device(get_free_gpu())
# device = torch.device("cpu")
print("selected device: ", device)

def generate_sample(path, stats):
    images = torch.zeros((0,C,SZ,SZ)).to(device)
    for _ in range(16):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        idx = torch.randint(0, tokens.shape[0], (1,))
        context[0, 0] = tokens[idx,0]
        # print(context[0, 0])

        res = gpt.generate(context, IMAGE_TOKENS-1)
        res = res[:,1:]
        img = model.decode(res)

        img = denormalize(img, stats)

        images = torch.cat([images, img], dim=0)

    grid_pred = torchvision.utils.make_grid(images, nrow=4)
    grid_final = grid_pred.permute(1, 2, 0)

    grid_final = grid_final.cpu().detach().numpy()
    grid_final = (grid_final * 255).astype("uint8")
    grid_final = PIL.Image.fromarray(grid_final)
    grid_final.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_name = f"gpt-{args.dataset}-{time.time():.0f}"
    print("run_name: ", run_name)

    run_folder = Path("runs") / run_name
    run_folder.mkdir(exist_ok=True, parents=True)


    config = model_configs[args.dataset]
    vqvae_config = config["vqvae_config"]
    C, SZ, K, D = vqvae_config.in_channels, vqvae_config.image_sz, vqvae_config.K, vqvae_config.D
    CONVS = 4
    IMAGE_TOKENS = (SZ//CONVS)**2+1
    block_size = IMAGE_TOKENS-1
    batch_size = 64
    eval_iters = 10
    eval_interval = 100
    max_iters = 500000
    print(f"dataset={args.dataset}, {C=}, {SZ=}, {IMAGE_TOKENS=}, {block_size=}, {batch_size=}\
            {eval_iters=}, {max_iters=}, {K=} {D=}")

    gpt_config = {
        "block_size": block_size,
        "vocab_size": K+10,
        "n_embd": 364,
        "n_head": 4,
        "n_layer": 2,
    }


    model = VQVAE(vqvae_config).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{args.dataset}_best.pth", map_location=device))
    
    """
    gpt = GPTLanguageModel(**gpt_config).to(device)
    gpt.load_state_dict(torch.load(f"checkpoints/{args.dataset}_gpt.pth", map_location=device))
    params = sum(p.numel() for p in gpt.parameters())
    print(f"number of parameters: {params / 1_000_000:.1f}M")
    wandb.watch(gpt)
    gpt.eval()
    model.eval()
    """

    gened = torch.load("generated.pt", map_location=device)
    print(gened.shape)
    # exit(0)

    for i in range(10):
        # init = torch.tensor([32+i], dtype=torch.long, device=device).view(1, 1)
        # print(init)
        # tokens = gpt.generate(init, 49)
        tokens = gened[i:i+1,:]

        print(tokens.shape)
        print(tokens)

        image = model.decode(tokens[:,1:])
        print(image.shape)
        image = denormalize(image, config["stats"])
        # print(image.shape)
        # exit(0)

        image = image[0].permute(1, 2, 0).cpu().detach().numpy()
        image = (image * 255).astype("uint8")
        print(image.shape)
        plt.imshow(image, cmap="gray")
        plt.savefig(f"gens/init_{i}.png")
        # break

    exit(0)

    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, tokens):
            self.tokens = tokens
            self.block_size = block_size

        def __len__(self):
            return self.tokens.shape[0]

        def __getitem__(self, idx):
            return self.tokens[idx, :self.block_size], self.tokens[idx, 1:self.block_size+1]

    optim = torch.optim.AdamW(gpt.parameters(), lr=args.lr)

    split = int(0.8 * len(tokens))
    train_tokens, val_tokens = tokens[:split], tokens[split:]

    train_dataset = TokenDataset(train_tokens)
    val_dataset = TokenDataset(val_tokens)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    Path("gens").mkdir(exist_ok=True)

    best_val_loss = float('inf')

    for iter in range(max_iters):

        # bar = tqdm.tqdm(range(len(train_loader)), position=0)
        start_time = time.time()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            _, loss = gpt(xb, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

            full_time = time.time() - start_time
            start_time = time.time()

            # bar.set_description(f"loss: {loss.item():.4f} time: {full_time:.2f}s, fps={1/full_time:.2f}")
            # bar.update(1)

        valid_losses = []
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                _, loss = gpt(xb, yb)
            valid_losses.append(loss.item())

        valid_loss = torch.tensor(valid_losses).mean().item()
        train_loss = torch.tensor(train_losses).mean().item()
        print(f"{iter=}, {train_loss=:.3f}, {valid_loss=:.3f}")
        generate_sample(run_folder / f"{iter}.png", config["stats"])

    torch.save(gpt.state_dict(), f"checkpoints/{args.dataset}_gpt.pth")
