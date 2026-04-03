import torch
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simulate env data
num_envs = 8
obs_shape = (4, 84, 84)

def benchmark_baseline(iters=10000):
    rewards = torch.zeros((100, num_envs)).to(device)

    start_time = time.time()
    for step in range(100):
        for _ in range(iters // 100):
            reward = np.random.rand(num_envs).astype(np.float32)
            next_obs = np.random.rand(num_envs, *obs_shape).astype(np.float32)
            next_done = np.random.randint(0, 2, num_envs).astype(bool)

            # Baseline code
            rewards_t = torch.tensor(reward).to(device).view(-1)
            next_obs_t, next_done_t = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

    return time.time() - start_time

def benchmark_optimized(iters=10000):
    rewards = torch.zeros((100, num_envs)).to(device)

    start_time = time.time()
    for step in range(100):
        for _ in range(iters // 100):
            reward = np.random.rand(num_envs).astype(np.float32)
            next_obs = np.random.rand(num_envs, *obs_shape).astype(np.float32)
            next_done = np.random.randint(0, 2, num_envs).astype(bool)

            # Optimized code
            rewards_t = torch.tensor(reward, device=device).view(-1)
            next_obs_t = torch.tensor(next_obs, device=device)
            next_done_t = torch.tensor(next_done, dtype=torch.float32, device=device)

    return time.time() - start_time

# Warmup
benchmark_baseline(100)
benchmark_optimized(100)

t_base = benchmark_baseline()
t_opt = benchmark_optimized()

print(f"Baseline: {t_base:.4f}s")
print(f"Optimized: {t_opt:.4f}s")
print(f"Improvement: {(t_base - t_opt) / t_base * 100:.2f}%")
