import torch
import time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def method_1(num_iters):
    tracked = []
    start = time.time()
    for _ in range(num_iters):
        t = torch.randn(1, 1000, 10, device=device)
        tracked.append(t.detach().cpu().numpy())
    res = np.concatenate(tracked, axis=0)
    end = time.time()
    return end - start, res.shape

def method_2(num_iters):
    tracked = []
    start = time.time()
    for _ in range(num_iters):
        t = torch.randn(1, 1000, 10, device=device)
        tracked.append(t.detach())
    res = torch.cat(tracked, dim=0).cpu().numpy()
    end = time.time()
    return end - start, res.shape

print(f"Method 1: {method_1(1000)[0]:.4f}s")
print(f"Method 2: {method_2(1000)[0]:.4f}s")
