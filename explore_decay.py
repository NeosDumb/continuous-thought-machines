import torch
from models.utils import compute_decay

def explore():
    T = 5
    params = torch.tensor([0.1, 1.0, 10.0])
    # Mocking params.device and params.shape
    # Actually params IS a tensor, so it has .device and .shape

    print(f"T: {T}")
    print(f"params: {params}")

    decay = compute_decay(T, params)
    print(f"Decay output:\n{decay}")
    print(f"Decay shape: {decay.shape}")

    # Test clamping
    params_to_clamp = torch.tensor([-1.0, 20.0])
    decay_clamped = compute_decay(T, params_to_clamp, clamp_lims=(0, 15))
    print(f"\nParams to clamp: {params_to_clamp}")
    print(f"Decay clamped output:\n{decay_clamped}")

    # Manual calculation for verification
    # indices = [4, 3, 2, 1, 0]
    # out = exp(-indices * clamped_params)
    # for params = 0.1, clamped = 0.1
    # out[:, 0] = [exp(-4*0.1), exp(-3*0.1), exp(-2*0.1), exp(-1*0.1), exp(0)]
    #           = [0.6703, 0.7408, 0.8187, 0.9048, 1.0]

    indices = torch.arange(T-1, -1, -1).reshape(T, 1)
    clamped_params = torch.clamp(params, 0, 15).unsqueeze(0)
    expected = torch.exp(-indices * clamped_params)
    print(f"\nExpected for basic params:\n{expected}")
    assert torch.allclose(decay, expected)
    print("Basic check passed!")

if __name__ == "__main__":
    explore()
