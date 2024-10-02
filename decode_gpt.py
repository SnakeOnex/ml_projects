import time, PIL
import torch, torch.nn as nn, torchvision, torch.nn.functional as F
from gpt_llama import GPT_L
from vqgan import VQGAN, VQGANConfig
from utils import get_free_gpu, denormalize
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_one_token(model, x, input_pos, cfg_scale=1.0):
    if cfg_scale == 1.0:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos)
    else:
        x_comb = torch.cat([x,x])
        logits, _ = model(x_comb, cond_idx=None, input_pos=input_pos)
        cond_logits = logits[:x.shape[0], ...]
        uncond_logits = logits[x.shape[0]:, ...]
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    probs = F.softmax(logits[:, -1, :], dim=-1)
    return torch.multinomial(probs, num_samples=1)

def prefill(model, cond, input_pos, cfg_scale=1.0):
    if cfg_scale == 1.0:
        logits, _ = model(None, cond_idx=cond, input_pos=input_pos)
    else:
        cond_null = torch.ones_like(cond) * 1000
        cond_comb = torch.cat([cond, cond_null])
        logits, _ = model(None, cond_idx=cond_comb, input_pos=input_pos)
        cond_logits = logits[:cond.shape[0], ...]
        uncond_logits = logits[cond.shape[0]:, ...]
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    probs = F.softmax(logits[:, -1, :], dim=-1)
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def generate(model, cond, max_new_tokens, cfg_scale):
    T = 1 + max_new_tokens # cond + new tokens
    max_batch_size = cond.shape[0] if cfg_scale == 1.0 else cond.shape[0] * 2

    device = cond.device
    with torch.device(device):
        model.setup_caches(max_batch_size=max_batch_size, max_seq_length=T, dtype=model.tok_embeddings.weight.dtype)

    # 1. prefill input_seq and K/V caches with the conditioning tokens
    seq = torch.empty((cond.shape[0], T-1), dtype=torch.int, device=device)
    input_pos = torch.arange(0, 1, device=device)
    next_token = prefill(model, cond, input_pos, cfg_scale)
    seq[:, 0:1] = next_token

    # 2. generate new tokens
    input_pos = torch.tensor([1], device=device, dtype=torch.int)
    new_tokens = []
    for t in range(max_new_tokens-1):
        next_token = decode_one_token(model, next_token, input_pos, cfg_scale)
        input_pos += 1
        new_tokens.append(next_token.clone())
        next_token = next_token.view(-1, 1)
    seq[:, 1:] = torch.cat(new_tokens, dim=1)

    return seq[:, 0:]

if __name__ == "__main__":
    gpt = GPT_L().to(device)
    # self.gpt = DDP(self.gpt, device_ids=[self.local_rank])
    state_dict = torch.load("runs_gpt/gpt-imagenet-1727429496/best.pth")
    new_state_dict = {k[7:]: v for k, v in state_dict.items()}
    gpt.load_state_dict(new_state_dict)
    gpt.eval()

    vqgan = VQGAN(VQGANConfig(K=1024, D=256)).to(device).eval()
    vqgan_path = "runs_vqgan/vqgan-imagenet-1726089582/best.pth"
    vqgan.load_state_dict(torch.load(vqgan_path))
    bs = 16
    cfg_scale = 2.0

    st = time.time()
    for i in trange(0,1000):
        cond = torch.zeros((bs,), dtype=torch.long, device=device)
        cond += i
        output_tokens = generate(gpt, cond, 256, cfg_scale)
        filename = f"samples/{i}{'_cfg' if cfg_scale>1.0 else ''}.png"

        images = (denormalize(vqgan.decode(output_tokens)) * 255).to(dtype=torch.uint8)
        grid = torchvision.utils.make_grid(images, nrow=4).permute(1, 2, 0).cpu().detach().numpy()
        grid_final = PIL.Image.fromarray(grid)
        grid_final.save(filename)
        print(f"Saved {filename}")

    dur = time.time()-st
    print(f"Time taken: {dur:.2f} seconds")
    print(f"Speed: {256*1000/dur:.2f} tokens per second")
    print(f"Speed: {16*256*1000/dur:.2f} batched tokens per second")
