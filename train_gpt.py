import argparse, time, time, tqdm, PIL, wandb, numpy as np, pickle
import torch, torch.nn as nn, torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from vqgan import VQGAN
from model_configs import model_configs
from gpt import GPTLanguageModel
from utils import get_free_gpu, denormalize

device = torch.device(get_free_gpu())
# device = torch.device("cpu")
print("selected device: ", device)

def generate_sample(path, stats):
    gpt.eval()
    images = torch.zeros((0,C,SZ,SZ)).to(device)

    # context = torch.zeros((16, 1), dtype=torch.long, device=device)
    # idx = torch.randint(0, tokens.shape[0], (1,))
    idx = torch.ones((16,1), dtype=torch.long, device=device)*K
    # context[:, 0] = idx
    context = idx

    res = gpt.generate(context, IMAGE_TOKENS-1)

    res = res[:,1:]
    res[res >= K] = K-1
    imgs = model.decode(res)
    images = denormalize(imgs, stats)

    grid_pred = torchvision.utils.make_grid(images, nrow=4)
    grid_final = grid_pred.permute(1, 2, 0)

    grid_final = grid_final.cpu().detach().numpy()
    grid_final = (grid_final * 255).astype("uint8")
    grid_final = PIL.Image.fromarray(grid_final)
    grid_final.save(path)
    gpt.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_tokens", action="store_true")
    args = parser.parse_args()

    run_name = f"gpt-{args.dataset}-{time.time():.0f}"
    print("run_name: ", run_name)

    run_folder = Path("runs") / run_name
    run_folder.mkdir(exist_ok=True, parents=True)


    config = model_configs[args.dataset]
    vqvae_config = config["vqvae_config"]
    # C, SZ, K, D = config["channels"], config["image_sz"], config["K"], config["D"]
    C, SZ, K, D = vqvae_config.in_channels, vqvae_config.image_sz, vqvae_config.K, vqvae_config.D
    CONVS = vqvae_config.num_resolutions-1
    # IMAGE_TOKENS = (SZ//CONVS)**2+1
    IMAGE_TOKENS = (SZ//(2**CONVS))**2+1
    block_size = IMAGE_TOKENS-1
    batch_size = 16
    eval_iters = 10
    eval_interval = 100
    max_iters = 500000
    print(f"dataset={args.dataset}, {C=}, {SZ=}, {IMAGE_TOKENS=}, {block_size=}, {batch_size=}\
            {eval_iters=}, {max_iters=}, {K=} {D=}")

    gpt_config = {
        "block_size": block_size,
        "vocab_size": K+1,
        "n_embd": 768,
        "n_head": 10,
        "n_layer": 10,
    }

    wandb.init(project="gpt-vqvae",
               name=run_name,
               config={"dataset": args.dataset, 
                       "batch_size": batch_size, 
                       "max_iters": max_iters,
                       "lr": args.lr,
                       "K": K,
                       "SZ": SZ,
                       "C": C,
                       "gpt_config": gpt_config,
                       })

    train_dataset, test_dataset = config["fetch_train"](), config["fetch_test"]()
    print(f"train_sz={len(train_dataset)}, test_sz={len(test_dataset)}")

    train_loader = DataLoader(
            train_dataset, 
            batch_size=8, 
            shuffle=False,
            num_workers=8, 
            prefetch_factor=4, 
            pin_memory=True,
            persistent_workers=False
    )
    test_loader = DataLoader(
            test_dataset, 
            batch_size=8, 
            shuffle=False, 
            num_workers=8, 
            prefetch_factor=4, 
            pin_memory=True,
            persistent_workers=False
    )

    model = VQGAN(vqvae_config).to(device)
    # model.load_state_dict(torch.load(f"checkpoints/{args.dataset}_best.pth", map_location=device))
    model.load_state_dict(torch.load(f"runs_vqvae/vqvae-bird-1725276726/bird_best.pth", map_location=device))
    model.eval()

    gpt = GPTLanguageModel(**gpt_config).to(device)
    params = sum(p.numel() for p in gpt.parameters())
    print(f"number of parameters: {params / 1_000_000:.1f}M")
    gpt.train()

    tokens = torch.zeros((0,IMAGE_TOKENS), dtype=torch.long, device=device)
    optim = torch.optim.AdamW(gpt.parameters(), lr=args.lr)

    # for x, y in tqdm.tqdm(test_loader):

    # do the same for loop but instatiate the loading bar

    for epoch in range(10):
        bar = tqdm.tqdm(train_loader)

        for i, (x, y) in enumerate(bar):
            x, y = x.to(device), y.to(device)
            _, quantized, _ = model(x)

            quantized = quantized.view(x.shape[0], -1)
            # quantized = torch.cat([y.view((-1,1))+K, quantized], dim=1)
            quantized = torch.cat([(y.view((-1,1))*0)+K, quantized], dim=1)

            tokens_x = quantized[:,:block_size]
            tokens_y = quantized[:,1:block_size+1]

            tokens, loss = gpt(tokens_x, tokens_y)
            bar.set_description(f"loss: {loss.item():.4f}")
            wandb.log({"loss": loss.item()})

            loss.backward()
            optim.step()
            optim.zero_grad()

            if i % 1000 == 0:
                generate_sample(run_folder / f"{epoch}.png", config["stats"])
                wandb.log({"gen_sample": [wandb.Image(str(run_folder / f"{epoch}.png"))]})

        bar = tqdm.tqdm(test_loader)
        val_loss, count = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(bar):
                x, y = x.to(device), y.to(device)
                _, quantized, _ = model(x)

                quantized = quantized.view(x.shape[0], -1)
                # quantized = torch.cat([y.view((-1,1))+K, quantized], dim=1)
                quantized = torch.cat([(y.view((-1,1))*0)+K, quantized], dim=1)

                tokens_x = quantized[:,:block_size]
                tokens_y = quantized[:,1:block_size+1]

                tokens, loss = gpt(tokens_x, tokens_y)
                val_loss += loss.item(); count += 1

                bar.set_description(f"loss: {loss.item():.4f}")
        print("val_loss=", val_loss/count)

        # log val loss with epoch on x axis
        wandb.log({"val_loss": val_loss/count, "epoch": epoch})

    # tokens = tokens[torch.randperm(tokens.shape[0])]
    tokens = tokens.to(torch.device("cpu"))


    print("tokens shape: ", tokens.shape)
    print("tokens_count: ", tokens.view(-1).shape[0])



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

    if args.save_tokens:
        train_tokens_np = train_tokens.to(torch.device("cpu")).numpy().astype("uint16")
        val_tokens_np = val_tokens.to(torch.device("cpu")).numpy().astype("uint16")
        train_tokens_np.tofile(run_folder / "train.bin")
        val_tokens_np.tofile(run_folder / "val.bin")
        meta = {"vocab_size": K+10, "block_size": block_size}
        with open(run_folder / "meta.pkl", "wb") as f: pickle.dump(meta, f)
        # exit(0)

    train_dataset = TokenDataset(train_tokens)
    val_dataset = TokenDataset(val_tokens)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=False)

    @torch.no_grad()
    def eval():
        losses = []
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = gpt(xb, yb)
            losses.append(loss.item())
        return torch.tensor(losses).mean().item()

    Path("gens").mkdir(exist_ok=True)

    best_val_loss = float('inf')

    for iter in range(max_iters):

        bar = tqdm.tqdm(range(len(train_loader)), position=0)
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

            bar.set_description(f"loss: {loss.item():.4f} time: {full_time:.2f}s, fps={1/full_time:.2f}")
            bar.update(1)

        valid_losses = []
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                _, loss = gpt(xb, yb)
            valid_losses.append(loss.item())

        valid_loss = torch.tensor(valid_losses).mean().item()
        train_loss = torch.tensor(train_losses).mean().item()
        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(gpt.state_dict(), run_folder / f"{args.dataset}_gpt.pth")
        print(f"{iter=}, {train_loss=:.3f}, {valid_loss=:.3f}")
        generate_sample(run_folder / f"{iter}.png", config["stats"])

        if iter % 5 == 0:
            wandb.save(run_folder / f"{iter}.png")

    torch.save(gpt.state_dict(), f"checkpoints/{args.dataset}_gpt.pth")
