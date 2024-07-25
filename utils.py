import subprocess, torch, numpy as np

def get_free_gpu():
    command = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
    mem_per_gpu = [int(x) for x in subprocess.check_output(command.split(), encoding='utf-8').strip().split('\n')]
    min_mem_gpu = np.argmin(mem_per_gpu)
    print(f"Selected GPU: {min_mem_gpu}")
    assert mem_per_gpu[min_mem_gpu] < 100, "No GPU has free memory"
    return f"cuda:{min_mem_gpu}"

def denormalize(x, stats):
    mean = torch.tensor(stats[0]).reshape(1, x.shape[1], 1, 1).to(x.device)
    std = torch.tensor(stats[1]).reshape(1, x.shape[1], 1, 1).to(x.device)
    return torch.clamp(x * std + mean, 0, 1)
