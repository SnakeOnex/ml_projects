import time
import torch, torch.nn as nn, torchvision
from gpt_llama import GPT_B

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    gpt = GPT_B().to(device)
    bs = 16
    # with torch.device(device):
        # gpt.setup_caches(max_batch_size=bs, max_seq_length=258, dtype=gpt.tok_embeddings.weight.dtype)
    st = time.time()
    idx = None
    cond = torch.zeros((bs,1), dtype=torch.long, device=device)
    cond += 7

    with torch.inference_mode():
        gpt.generate(cond, 256, verbose=True)[:,1:]
    dur = time.time()-st
    print(f"Time taken: {dur:.2f} seconds")
    print(f"Speed: {256/dur:.2f} tokens per second")
