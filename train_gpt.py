import argparse, time, time, tqdm, PIL, wandb
import torch, torch.nn as nn, torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from vqvae import VQVAE
from model_configs import model_configs
from gpt import GPTLanguageModel
from utils import get_free_gpu

device = torch.device(get_free_gpu())
# device = torch.device("cpu")
print("selected device: ", device)

def generate_sample(path):
    images = torch.zeros((0,C,SZ,SZ)).to(device)
    for _ in range(16):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        idx = torch.randint(0, tokens.shape[0], (1,))
        context[0, 0] = tokens[idx,0]
        print(context[0, 0])

        res = gpt.generate(context, IMAGE_TOKENS-1)
        res = res[:,1:]
        img = model.decode(res)

        if args.dataset == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, C, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
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
    C, SZ, K, D = config["channels"], config["image_sz"], config["K"], config["D"]
    CONVS = 2
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
            batch_size=128, 
            shuffle=False,
            num_workers=8, 
            prefetch_factor=4, 
            pin_memory=True,
            persistent_workers=False
    )
    test_loader = DataLoader(
            test_dataset, 
            batch_size=128, 
            shuffle=False, 
            num_workers=8, 
            prefetch_factor=4, 
            pin_memory=True,
            persistent_workers=False
    )

    model = VQVAE(config).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{args.dataset}_best.pth"))

    tokens = torch.zeros((0,IMAGE_TOKENS), dtype=torch.long, device=device)
    for x, y in tqdm.tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        quantized = model(x)["closest"]
        quantized = torch.cat([y.view((-1,1))+K, quantized], dim=1)
        tokens = torch.cat([tokens, quantized], dim=0)

    # for x, _ in tqdm.tqdm(train_loader):
        # x = x.to(device)
        # quantized = model(x)["closest"]
        # tokens = torch.cat([tokens, quantized], dim=0)
    for x, y in tqdm.tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        quantized = model(x)["closest"]
        quantized = torch.cat([y.view((-1,1))+K, quantized], dim=1)
        tokens = torch.cat([tokens, quantized], dim=0)

    tokens = tokens[torch.randperm(tokens.shape[0])]
    tokens = tokens.to(torch.device("cpu"))


    print("tokens shape: ", tokens.shape)
    print("tokens_count: ", tokens.view(-1).shape[0])


    gpt = GPTLanguageModel(**gpt_config).to(device)
    params = sum(p.numel() for p in gpt.parameters())
    print(f"number of parameters: {params / 1_000_000:.1f}M")
    wandb.watch(gpt)
    gpt.train()

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

    # bar = tqdm.tqdm(range(max_iters))
    for iter in range(max_iters):

        start_time = time.time()
        # for xb, yb in train_loader:
        for xb, yb in tqdm.tqdm(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            logits, loss = gpt(xb, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()

            full_time = time.time() - start_time
            start_time = time.time()

            # bar.set_description(f"loss: {loss.item():.4f} time: {full_time:.2f}s, fps={1/full_time:.2f}")
            # bar.update(1)

        # losses = []
        # for xb, yb in val_loader:
            # xb, yb = xb.to(device), yb.to(device)
            # _, loss = gpt(xb, yb)
            # losses.append(loss.item())
        # val_loss = torch.tensor(losses).mean().item()
        print(f"validation loss: {eval():.4f}")
        generate_sample(run_folder / f"{iter}.png")

    torch.save(gpt.state_dict(), f"checkpoints/{args.dataset}_gpt.pth")
