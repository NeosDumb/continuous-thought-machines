import time
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark_old(iterations, batch_size, d_model):
    state_trace = torch.randn(batch_size, d_model, 1, device=device)

    pre_activations_tracking = []
    post_activations_tracking = []
    attention_tracking = []
    synch_out_tracking = []
    synch_action_tracking = []

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    for _ in range(iterations):
        t1 = torch.randn(batch_size, d_model, device=device)
        t2 = torch.randn(batch_size, d_model, device=device)
        t3 = torch.randn(batch_size, d_model, device=device)
        t4 = torch.randn(batch_size, d_model, device=device)
        t5 = torch.randn(batch_size, d_model, device=device)

        pre_activations_tracking.append(t1.detach().cpu().numpy())
        post_activations_tracking.append(t2.detach().cpu().numpy())
        attention_tracking.append(t3.detach().cpu().numpy())
        synch_out_tracking.append(t4.detach().cpu().numpy())
        synch_action_tracking.append(t5.detach().cpu().numpy())

    res1 = np.array(pre_activations_tracking)
    res2 = np.array(post_activations_tracking)
    res3 = np.array(attention_tracking)
    res4 = np.array(synch_out_tracking)
    res5 = np.array(synch_action_tracking)

    torch.cuda.synchronize() if device == "cuda" else None
    end = time.perf_counter()
    return end - start

def benchmark_new(iterations, batch_size, d_model):
    state_trace = torch.randn(batch_size, d_model, 1, device=device)

    pre_activations_tracking = []
    post_activations_tracking = []
    attention_tracking = []
    synch_out_tracking = []
    synch_action_tracking = []

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    for _ in range(iterations):
        t1 = torch.randn(batch_size, d_model, device=device)
        t2 = torch.randn(batch_size, d_model, device=device)
        t3 = torch.randn(batch_size, d_model, device=device)
        t4 = torch.randn(batch_size, d_model, device=device)
        t5 = torch.randn(batch_size, d_model, device=device)

        pre_activations_tracking.append(t1.detach())
        post_activations_tracking.append(t2.detach())
        attention_tracking.append(t3.detach())
        synch_out_tracking.append(t4.detach())
        synch_action_tracking.append(t5.detach())

    if len(pre_activations_tracking) > 0:
        res1 = torch.stack(pre_activations_tracking).cpu().numpy()
        res2 = torch.stack(post_activations_tracking).cpu().numpy()
        res3 = torch.stack(attention_tracking).cpu().numpy()
        res4 = torch.stack(synch_out_tracking).cpu().numpy()
        res5 = torch.stack(synch_action_tracking).cpu().numpy()

    torch.cuda.synchronize() if device == "cuda" else None
    end = time.perf_counter()
    return end - start

iters = 100
b = 32
d = 256

# warm up
benchmark_old(10, b, d)
benchmark_new(10, b, d)

n_trials = 100
t_old = sum(benchmark_old(iters, b, d) for _ in range(n_trials)) / n_trials
t_new = sum(benchmark_new(iters, b, d) for _ in range(n_trials)) / n_trials

print(f"Old: {t_old:.4f}s")
print(f"New: {t_new:.4f}s")
print(f"Speedup: {t_old / t_new:.2f}x")
