import argparse, time, time, tqdm, PIL, wandb, numpy as np, pickle
import torch, torch.nn as nn, torchvision, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from vqgan import VQGAN
from model_configs import model_configs
from gpt import GPTLanguageModel, GPTConfig
from utils import get_free_gpu, denormalize

device = torch.device(get_free_gpu())
# device = torch.device("cpu")
print("selected device: ", device)

def generate_sample(path):
    gpt.eval()
    images = torch.zeros((0,C,SZ,SZ)).to(device)

    # context = torch.zeros((16, 1), dtype=torch.long, device=device)
    # idx = torch.randint(0, tokens.shape[0], (1,))
    idx = torch.ones((16,1), dtype=torch.long, device=device)*K
    # context[:, 0] = idx
    context = idx

    context = torch.ones((16,256), dtype=torch.long, device=device)*K
    res = gpt.generate_maskgit(context, steps=8)
    # print("res: ", res.shape)
    # exit(0)

    # res = res[:,1:]
    print(f"{res.min()=}, {res.max()=}, {res.float().mean()=}")
    res[res >= K] = K-1
    print(f"{res.min()=}, {res.max()=}, {res.float().mean()=}")
    # exit(0)
    imgs = vqgan.decode(res)
    images = denormalize(imgs)

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
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_tokens", action="store_true")
    args = parser.parse_args()

    run_name = f"gpt-{args.dataset}-{time.time():.0f}"
    print("run_name: ", run_name)

    run_folder = Path("runs") / run_name
    run_folder.mkdir(exist_ok=True, parents=True)


    config = model_configs[args.dataset]
    vqvae_config = config["vqgan_config"]
    # C, SZ, K, D = config["channels"], config["image_sz"], config["K"], config["D"]
    C, SZ, K, D = vqvae_config.in_channels, vqvae_config.image_sz, vqvae_config.K, vqvae_config.D
    CONVS = 4
    # IMAGE_TOKENS = (SZ//CONVS)**2+1
    IMAGE_TOKENS = (SZ//(2**CONVS))**2+1
    block_size = IMAGE_TOKENS-1
    batch_size = 4
    eval_iters = 10
    eval_interval = 250
    max_iters = 500000
    print(f"dataset={args.dataset}, {C=}, {SZ=}, {IMAGE_TOKENS=}, {block_size=}, {batch_size=}\
            {eval_iters=}, {max_iters=}, {K=} {D=}")

    gpt_config = GPTConfig(
            block_size=block_size, 
            vocab_size=K+1, 
            n_embd=1024, 
            n_head=16, 
            n_layer=24,
            causal=False,
    )

    wandb.init(project="maskgit-vqgan",
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
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2, 
            prefetch_factor=4, 
            pin_memory=True,
            persistent_workers=True
    )
    test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            prefetch_factor=4, 
            pin_memory=True,
            persistent_workers=True
    )

    vqgan = VQGAN(vqvae_config).to(device)
    # model.load_state_dict(torch.load(f"checkpoints/{args.dataset}_best.pth", map_location=device))
    vqgan.load_state_dict(torch.load(f"runs_vqvae/vqvae-bird-1725458794/bird_best.pth", map_location=device))
    vqgan.eval()

    gpt = GPTLanguageModel(gpt_config).to(device)
    params = sum(p.numel() for p in gpt.parameters())
    print(f"number of parameters: {params / 1_000_000:.1f}M")
    gpt.train()

    tokens = torch.zeros((0,IMAGE_TOKENS), dtype=torch.long, device=device)

    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    # for mn, m in gpt.transformer.named_modules():
    for mn, m in gpt.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn

            if pn.endswith("bias"):
                no_decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    no_decay.add("position_embedding_table.weight")

    param_dict = {pn: p for pn, p in gpt.named_parameters()}

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    # optim = torch.optim.AdamW(gpt.parameters(), lr=args.lr)
    amp_enabled = False
    optim = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9,0.95))
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    # for x, y in tqdm.tqdm(test_loader):

    # do the same for loop but instatiate the loading bar

    for epoch in range(1000):
        bar = tqdm.tqdm(train_loader)

        for i, (x, y) in enumerate(bar):
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                _, quantized, _ = vqgan(x)

                quantized = quantized.view(x.shape[0], -1)

                B, T = quantized.shape
                # quantized = torch.cat([y.view((-1,1))+K, quantized], dim=1)
                # quantized = torch.cat([(y.view((-1,1))*0)+K, quantized], dim=1)
                # print(quantized.shape)
                # exit(0)


                # tokens_x = quantized[:,:block_size]
                # tokens_y = quantized[:,1:block_size+1]

                # p_mask = torch.randint(0, 8, (B,1), device=device) * 32
                # mask = torch.zeros((B, T), device=device).to(dtype=torch.int64)

                # for b in range(B):
                    # indices = torch.randperm(T, device=device)[:p_mask[b]]
                    # mask[b, indices] = 1
                    # print(torch.sum(mask[b,:]).item())
                # tokens_x = mask * quantized + (1 - mask) * K

                p_keep = torch.rand((B,1), device=device)
                mask = torch.bernoulli(p_keep * torch.ones(quantized.shape, device=device)).to(dtype=torch.int64)
                masked_input = torch.zeros_like(quantized)+K
                tokens_x = mask * quantized + (1 - mask) * masked_input

                # print(tokens_x.shape)
                # exit(0)

                # print("masked_input.shape: ", masked_input.shape)
                # print("tokens_x.shape: ", tokens_x.shape)

                logits, _ = gpt(tokens_x)
                # confidences = torch.softmax(logits, dim=-1)
                targets = quantized

                # print(logits.shape, targets.shape)
                # exit(0)
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.reshape(B*T)
                loss = F.cross_entropy(logits, targets)

                # what is the loss?
                # what is the mask?
                
                # print(f"{logits.shape=}")

                # exit(0)

                # mask = torch.bernoulli(p_keep * torch.ones(tokens_x.shape, device=device)).to(dtype=torch.int64)
                # random_indices = torch.randint_like(tokens_x, gpt_config["vocab_size"])
                # tokens_x = mask * tokens_x + (1 - mask) * random_indices

                # tokens, loss = gpt(tokens_x, tokens_y)
                bar.set_description(f"loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item()})

                # print(f"{i=}, {loss.item()=}")

                # scaler.scale(loss).backward()
                # scaler.step(optim)
                # scaler.update()

                loss.backward()
                optim.step()
                optim.zero_grad()

            if i % 1000 == 0:
                generate_sample(run_folder / f"{epoch}.png")
                wandb.log({"gen_sample": [wandb.Image(str(run_folder / f"{epoch}.png"))]})

        # bar = tqdm.tqdm(test_loader)
        # val_loss, count = 0, 0
        # with torch.no_grad():
            # for i, (x, y) in enumerate(bar):
                # x, y = x.to(device), y.to(device)
                # _, quantized, _ = vqgan(x)

                # quantized = quantized.view(x.shape[0], -1)
                # # quantized = torch.cat([y.view((-1,1))+K, quantized], dim=1)
                # quantized = torch.cat([(y.view((-1,1))*0)+K, quantized], dim=1)

                # tokens_x = quantized[:,:block_size]
                # tokens_y = quantized[:,1:block_size+1]

                # tokens, loss = gpt(tokens_x, tokens_y)
                # val_loss += loss.item(); count += 1

                # bar.set_description(f"loss: {loss.item():.4f}")
        # print("val_loss=", val_loss/count)
        # wandb.log({"val_loss": val_loss/count, "epoch": epoch})
