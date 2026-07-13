import time
import torch
import numpy as np
from data.custom_datasets import MazeImageFolder

def run_benchmark(num_workers):
    print(f"Benchmarking with num_workers={num_workers}...")
    try:
        # We need a small mock dataset since we don't have the real mazes
        dataset = MazeImageFolder(
            root="data/mazes/medium/train/", # Doesn't exist, will just mock
            which_set="train",
            maze_route_length=100,
            expand_range=True,
        )
    except FileNotFoundError:
        print("Mocking dataset for benchmark...")
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, size=1000):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                # Simulate some I/O and processing delay typical for image loading
                time.sleep(0.005)
                return torch.zeros((3, 39, 39)), torch.zeros(100, dtype=torch.long)

        dataset = MockDataset(size=500)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(loader):
        pass # Just iterate
    end_time = time.time()

    duration = end_time - start_time
    print(f"  Took {duration:.4f} seconds")
    return duration

if __name__ == '__main__':
    t_0 = run_benchmark(0)
    t_4 = run_benchmark(4)
    t_8 = run_benchmark(8)

    print("\nResults:")
    print(f"num_workers=0: {t_0:.4f}s (Baseline)")
    print(f"num_workers=4: {t_4:.4f}s ({t_0/t_4:.2f}x speedup)")
    print(f"num_workers=8: {t_8:.4f}s ({t_0/t_8:.2f}x speedup)")
