import argparse, time, time, tqdm, PIL, wandb
import torch, torch.nn as nn, torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from vqvae import VQVAE
from model_configs import model_configs
from gpt import GPTLanguageModel
from utils import get_free_gpu

device = torch.device(get_free_gpu())
print("selected device: ", device)

def generate_sample(path):
    images = torch.zeros((0,C,SZ,SZ)).to(device)
    for _ in range(16):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        idx = torch.randint(0, tokens.shape[0], (1,))
        context[0, 0] = tokens[idx,0]

        res = gpt.generate(context, IMAGE_TOKENS-1)
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
    run_folder = Path("runs") / run_name
    run_folder.mkdir(exist_ok=True, parents=True)


    config = model_configs[args.dataset]
    C, SZ, K, D = config["channels"], config["image_sz"], config["K"], config["D"]
    IMAGE_TOKENS = (SZ//4)**2
    block_size = IMAGE_TOKENS-1
    batch_size = 512
    eval_iters = 50
    eval_interval = 100
    max_iters = 500000
    print(f"dataset={args.dataset}, {C=}, {SZ=}, {IMAGE_TOKENS=}, {block_size=}, {batch_size=}\
            {eval_iters=}, {max_iters=}, {K=} {D=}")

    gpt_config = {
        "block_size": block_size,
        "vocab_size": K,
        "n_embd": 768,
        "n_head": 6,
        "n_layer": 6,
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
            shuffle=True, 
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
    for x, _ in tqdm.tqdm(test_loader):
        x = x.to(device)
        quantized = model(x)["closest"]
        tokens = torch.cat([tokens, quantized], dim=0)

    for x, _ in tqdm.tqdm(train_loader):
        x = x.to(device)
        quantized = model(x)["closest"]
        tokens = torch.cat([tokens, quantized], dim=0)

    tokens = tokens[:1000000]
    tokens = tokens[torch.randperm(tokens.shape[0])]

    print("tokens shape: ", tokens.shape)
    print("tokens_count: ", tokens.view(-1).shape[0])


    gpt = GPTLanguageModel(**gpt_config).to(device)
    wandb.watch(gpt)
    gpt.train()


    optim = torch.optim.AdamW(gpt.parameters(), lr=args.lr)

    split = int(0.8 * len(tokens))
    train_tokens, val_tokens = tokens[:split], tokens[split:]

    def get_batch(data):
        # generate a small batch of data of inputs x and targets y
        ix = torch.randint(data.shape[0], (batch_size,))
        x = torch.stack([data[i,0:block_size] for i in ix])
        y = torch.stack([data[i,1:block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(train_tokens if split == 'train' else val_tokens)
                _, loss = gpt(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    Path("gens").mkdir(exist_ok=True)

    best_val_loss = float('inf')

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters-1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            generate_sample(run_folder / f"{args.dataset}_generated_{iter}.png")
            wandb.log({"train_loss": losses["train"], "val_loss": losses["val"]})
            wandb.save(run_folder / f"{args.dataset}_generated_{iter}.png")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "step": iter},
                       run_folder / f"best_model.pth")

        # sample a batch of data
        xb, yb = get_batch(train_tokens)

        # evaluate the loss
        logits, loss = gpt(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    torch.save(gpt.state_dict(), f"checkpoints/{args.dataset}_gpt.pth")
