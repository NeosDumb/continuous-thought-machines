import time
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark_old(iterations, batch_size, d_model):
    state_trace = torch.randn(batch_size, d_model, 1, device=device)

    pre_activations_tracking = []

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    for _ in range(iterations):
        t = torch.randn(batch_size, d_model, device=device)
        pre_activations_tracking.append(t.detach().cpu().numpy())

    result = np.array(pre_activations_tracking)

    torch.cuda.synchronize() if device == "cuda" else None
    end = time.perf_counter()
    return end - start, result

def benchmark_new(iterations, batch_size, d_model):
    state_trace = torch.randn(batch_size, d_model, 1, device=device)

    pre_activations_tracking = []

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    for _ in range(iterations):
        t = torch.randn(batch_size, d_model, device=device)
        pre_activations_tracking.append(t.detach())

    result = torch.stack(pre_activations_tracking).cpu().numpy()

    torch.cuda.synchronize() if device == "cuda" else None
    end = time.perf_counter()
    return end - start, result

iters = 100
b = 32
d = 256

# warm up
benchmark_old(10, b, d)
benchmark_new(10, b, d)

n_trials = 100
t_old = sum(benchmark_old(iters, b, d)[0] for _ in range(n_trials)) / n_trials
t_new = sum(benchmark_new(iters, b, d)[0] for _ in range(n_trials)) / n_trials

print(f"Old: {t_old:.4f}s")
print(f"New: {t_new:.4f}s")
print(f"Speedup: {t_old / t_new:.2f}x")
