import torch, torch.nn as nn, torchvision, argparse, time, time, tqdm, PIL, wandb
from pathlib import Path
from torch.utils.data import DataLoader
from vqvae import VQVAE
from model_configs import model_configs
from gpt import GPTLanguageModel

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def plot_results(model, dataset, image_count, path="vqvae.png", idxs=None):
    if idxs is None:
        idxs = torch.randint(0, len(dataset), (16,))

    images_gt = torch.stack([dataset[i][0] for i in idxs])

    with torch.inference_mode():
        images_pred = model(images_gt.to(device))["output"]
    images_pred = images_pred.cpu().detach()

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images_pred = images_pred * std + mean
    images_pred = torch.clamp(images_pred, 0, 1)

    grid_pred = torchvision.utils.make_grid(images_pred, nrow=4)
    grid = torchvision.utils.make_grid(images_gt, nrow=4)
    grid_final = torch.cat([grid, grid_pred], dim=2)

    grid_final = grid_final.permute(1, 2, 0)


    grid_final = grid_final.numpy()
    grid_final = (grid_final * 255).astype("uint8")
    grid_final = PIL.Image.fromarray(grid_final)

    grid_final.save(f"{path}")

def generate_sample(path):
    images = torch.zeros((0,C,SZ,SZ)).to(device)
    for i in range(16):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        idx = torch.randint(0, tokens.shape[0], (1,))
        context[0, 0] = tokens[idx,0]

        res = gpt.generate(context, IMAGE_TOKENS-1)

        img = model.decode(res)
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        images = torch.cat([images, img], dim=0)

    grid_pred = torchvision.utils.make_grid(images, nrow=4)

    grid_final = grid_pred.permute(1, 2, 0)


    grid_final = grid_final.cpu().detach().numpy()
    grid_final = (grid_final * 255).astype("uint8")
    grid_final = PIL.Image.fromarray(grid_final)

    # grid_final.save(f"gens/{args.dataset}_generated.png")
    grid_final.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    config = model_configs[args.dataset]
    C, SZ, K = config["channels"], config["image_sz"], config["K"]
    IMAGE_TOKENS = (SZ//4)**2
    block_size = IMAGE_TOKENS-1
    batch_size = 128
    eval_iters = 50
    max_iters = 50000
    print(f"dataset={args.dataset}, {C=}, {SZ=}, {IMAGE_TOKENS=}, {block_size=}, {batch_size=}")

    print(f"training VQ-VAE on the {args.dataset} dataset")

    train_dataset, test_dataset = config["fetch_train"](), config["fetch_test"]()

    print(f"train_sz={len(train_dataset)}, test_sz={len(test_dataset)}")
    print(f"K={config['K']}, D={config['D']}")

    idxs = torch.randint(0, len(test_dataset), (16,))

    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    train_loader = DataLoader(
            train_dataset, 
            batch_size=256, 
            shuffle=True, 
            num_workers=16, 
            prefetch_factor=4, 
            pin_memory=True,
            persistent_workers=False
    )
    test_loader = DataLoader(
            test_dataset, 
            batch_size=256, 
            shuffle=False, 
            num_workers=16, 
            prefetch_factor=4, 
            pin_memory=True,
            persistent_workers=False
    )

    model = VQVAE(config).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{args.dataset}_best.pth"))
    print("loaded model")

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

    gpt = GPTLanguageModel(block_size, K).to(device)
    # gpt.load_state_dict(torch.load(f"checkpoints/{args.dataset}_gpt.pth"))
    gpt.train()

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)

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
                logits, loss = gpt(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    Path("gens").mkdir(exist_ok=True)
    if args.train:
        wandb.init(project="gpt-vqvae",
                   config={"dataset": args.dataset, 
                           "block_size": block_size, 
                           "batch_size": batch_size, 
                           "max_iters": max_iters,
                           "K": K,
                           "SZ": SZ,
                           "C": C})

        for iter in range(max_iters):

            # every once in a while evaluate the loss on train and val sets
            if iter % 50 == 0 or iter == max_iters-1:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                generate_sample(f"gens/{args.dataset}_generated_{iter}.png")
                wandb.log({"train_loss": losses["train"], "val_loss": losses["val"]})
                torch.save(gpt.state_dict(), f"checkpoints/{args.dataset}_gpt_{iter}.pth")

                if iter % 500 == 0:
                    wandb.save(f"gens/{args.dataset}_generated_{iter}.png")

            # sample a batch of data
            xb, yb = get_batch(train_tokens)

            # evaluate the loss
            logits, loss = gpt(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        torch.save(gpt.state_dict(), f"checkpoints/{args.dataset}_gpt.pth")
    else:
        gpt.load_state_dict(torch.load(f"checkpoints/{args.dataset}_gpt.pth"))
        gpt.eval()


    exit(0)

    # images = torch.zeros((0,1,28,28)).to(device)
    images = torch.zeros((0,C,SZ,SZ)).to(device)
    for i in range(16):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        idx = torch.randint(0, tokens.shape[0], (1,))
        context[0, 0] = tokens[idx,0]

        res = gpt.generate(context, IMAGE_TOKENS-1)

        img = model.decode(res)
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        images = torch.cat([images, img], dim=0)

    grid_pred = torchvision.utils.make_grid(images, nrow=4)

    grid_final = grid_pred.permute(1, 2, 0)


    grid_final = grid_final.cpu().detach().numpy()
    grid_final = (grid_final * 255).astype("uint8")
    grid_final = PIL.Image.fromarray(grid_final)

    grid_final.save(f"gens/{args.dataset}_generated.png")





