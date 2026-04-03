import torch
import time
from models.modules import CustomRotationalEmbedding1D

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CustomRotationalEmbedding1D(128).to(device)
    x = torch.randn(32, 128, 100).to(device)

    # Warmup
    for _ in range(100):
        _ = model(x)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    for _ in range(1000):
        _ = model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()

    print(f"Time taken for 1000 iterations: {end - start:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
