import torch
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bench_with():
    next_obs = np.random.randn(8, 4).astype(np.float32)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    start = time.time()
    for _ in range(10000):
        next_obs = torch.tensor(next_obs, device=device)
        obs = next_obs
        next_obs = np.random.randn(8, 4).astype(np.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    return time.time() - start

def bench_without():
    next_obs = np.random.randn(8, 4).astype(np.float32)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    start = time.time()
    for _ in range(10000):
        # next_obs = torch.tensor(next_obs, device=device)
        obs = next_obs
        next_obs = np.random.randn(8, 4).astype(np.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    return time.time() - start

print(f"With duplicate tensor creation: {bench_with():.4f}s")
print(f"Without duplicate tensor creation: {bench_without():.4f}s")
