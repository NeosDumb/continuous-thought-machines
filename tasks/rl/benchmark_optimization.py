import time
import timeit

def benchmark_simulated():
    print("Benchmarking simulated host-to-device transfer overhead...")

    # Simulate host-to-device transfer latency (exaggerated for demonstration)
    H2D_LATENCY = 0.0001 # 100 microseconds

    def slow_init():
        # torch.zeros(n).to(device)
        x = [0] * 1000 # CPU allocation
        time.sleep(H2D_LATENCY) # Simulated transfer
        return x

    def fast_init():
        # torch.zeros(n, device=device)
        x = [0] * 1000 # Direct allocation (simulated)
        return x

    slow_time = timeit.timeit(slow_init, number=1000)
    fast_time = timeit.timeit(fast_init, number=1000)

    print(f"Slow init (simulated): {slow_time:.4f}s")
    print(f"Fast init (simulated): {fast_time:.4f}s")
    print(f"Improvement: {(slow_time - fast_time) / slow_time * 100:.2f}%")

try:
    import torch
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False

def benchmark_torch():
    if not HAS_TORCH:
        print("Torch not available, skipping real benchmark.")
        return

    print(f"Benchmarking with Torch on {DEVICE}...")
    n = 10000

    def slow_init():
        return torch.zeros(n).to(DEVICE)

    def fast_init():
        return torch.zeros(n, device=DEVICE)

    # Warmup
    for _ in range(100):
        slow_init()
        fast_init()

    slow_time = timeit.timeit(slow_init, number=1000)
    fast_time = timeit.timeit(fast_init, number=1000)

    print(f"Slow init (torch): {slow_time:.4f}s")
    print(f"Fast init (torch): {fast_time:.4f}s")
    print(f"Improvement: {(slow_time - fast_time) / slow_time * 100:.2f}%")

if __name__ == "__main__":
    benchmark_simulated()
    if HAS_TORCH:
        benchmark_torch()
    else:
        print("\nRationale: Direct device allocation avoids redundant CPU memory allocation and subsequent host-to-device memory copy. Even on CPU-only systems, it avoids a redundant copy operation.")
