import time
import numpy as np

def softmax_simple(x, axis=None):
    # Simplified softmax for benchmarking purposes (since scipy is missing)
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def benchmark_loop_1(data, iterations):
    start = time.perf_counter()
    results = []
    for stepi in range(iterations):
        # Current inefficient approach: softmax inside the loop
        pred_avg = softmax_simple(data, 1)[:, :, :stepi + 1].mean(-1)
        results.append(pred_avg.sum()) # dummy operation
    end = time.perf_counter()
    return end - start, results

def benchmark_loop_1_optimized(data, iterations):
    start = time.perf_counter()
    # Optimized approach: softmax outside the loop
    concatenated_probs = softmax_simple(data, 1)
    results = []
    for stepi in range(iterations):
        pred_avg = concatenated_probs[:, :, :stepi + 1].mean(-1)
        results.append(pred_avg.sum()) # dummy operation
    end = time.perf_counter()
    return end - start, results

def benchmark_loop_2(data, targets, iterations):
    start = time.perf_counter()
    results = []
    for stepi in range(iterations):
        # Current inefficient approach: softmax inside the loop
        pred = data[:, :, stepi].argmax(1)
        probabilities = softmax_simple(data[:, :, :stepi + 1], axis=1)[np.arange(data.shape[0]), pred].mean(-1)
        results.append(probabilities.sum()) # dummy operation
    end = time.perf_counter()
    return end - start, results

def benchmark_loop_2_optimized(data, targets, iterations):
    start = time.perf_counter()
    # Optimized approach: softmax outside the loop
    concatenated_probs = softmax_simple(data, axis=1)
    results = []
    for stepi in range(iterations):
        pred = data[:, :, stepi].argmax(1)
        probabilities = concatenated_probs[np.arange(data.shape[0]), pred, :stepi + 1].mean(-1)
        results.append(probabilities.sum()) # dummy operation
    end = time.perf_counter()
    return end - start, results

if __name__ == "__main__":
    # Synthetic data similar to Imagenet results
    # concatenated_predictions shape: (B, Classes, Steps)
    B = 100
    Classes = 1000
    Steps = 50
    data = np.random.randn(B, Classes, Steps)
    targets = np.random.randint(0, Classes, B)

    print(f"Benchmarking with B={B}, Classes={Classes}, Steps={Steps}")

    t1, r1 = benchmark_loop_1(data, Steps)
    t1_opt, r1_opt = benchmark_loop_1_optimized(data, Steps)
    print(f"Loop 1: Baseline={t1:.4f}s, Optimized={t1_opt:.4f}s, Speedup={t1/t1_opt:.2f}x")
    np.testing.assert_allclose(r1, r1_opt)

    t2, r2 = benchmark_loop_2(data, targets, Steps)
    t2_opt, r2_opt = benchmark_loop_2_optimized(data, targets, Steps)
    print(f"Loop 2: Baseline={t2:.4f}s, Optimized={t2_opt:.4f}s, Speedup={t2/t2_opt:.2f}x")
    np.testing.assert_allclose(r2, r2_opt)
    print("Verification successful: Results match.")
