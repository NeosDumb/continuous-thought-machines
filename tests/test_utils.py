import torch
import pytest
from models.utils import compute_decay

def test_compute_decay_basic():
    """Test basic functionality of compute_decay."""
    T = 3
    params = torch.tensor([1.0, 2.0])

    # Expected calculation:
    # indices = [2, 1, 0]
    # out[0, 0] = exp(-2 * 1.0) = exp(-2.0)
    # out[1, 0] = exp(-1 * 1.0) = exp(-1.0)
    # out[2, 0] = exp(0 * 1.0) = 1.0
    # out[0, 1] = exp(-2 * 2.0) = exp(-4.0)
    # out[1, 1] = exp(-1 * 2.0) = exp(-2.0)
    # out[2, 1] = exp(0 * 2.0) = 1.0

    expected = torch.tensor([
        [torch.exp(torch.tensor(-2.0)), torch.exp(torch.tensor(-4.0))],
        [torch.exp(torch.tensor(-1.0)), torch.exp(torch.tensor(-2.0))],
        [1.0, 1.0]
    ])

    output = compute_decay(T, params)

    assert output.shape == (T, 2)
    assert torch.allclose(output, expected)

def test_compute_decay_clamping():
    """Test that params are correctly clamped."""
    T = 2
    # clamp_lims=(0, 15) by default
    params = torch.tensor([-1.0, 20.0])

    # Expected: -1.0 becomes 0, 20.0 becomes 15.0
    # indices = [1, 0]
    # out[0, 0] = exp(-1 * 0) = 1.0
    # out[1, 0] = exp(0 * 0) = 1.0
    # out[0, 1] = exp(-1 * 15.0)
    # out[1, 1] = exp(0 * 15.0) = 1.0

    expected = torch.tensor([
        [1.0, torch.exp(torch.tensor(-15.0))],
        [1.0, 1.0]
    ])

    output = compute_decay(T, params)

    assert torch.allclose(output, expected)

def test_compute_decay_custom_clamping():
    """Test with custom clamp limits."""
    T = 2
    params = torch.tensor([0.5, 5.0])
    clamp_lims = (1.0, 2.0)

    # Expected: 0.5 becomes 1.0, 5.0 becomes 2.0
    # indices = [1, 0]
    # out[0, 0] = exp(-1 * 1.0)
    # out[1, 0] = exp(0 * 1.0) = 1.0
    # out[0, 1] = exp(-1 * 2.0)
    # out[1, 1] = exp(0 * 2.0) = 1.0

    expected = torch.tensor([
        [torch.exp(torch.tensor(-1.0)), torch.exp(torch.tensor(-2.0))],
        [1.0, 1.0]
    ])

    output = compute_decay(T, params, clamp_lims=clamp_lims)

    assert torch.allclose(output, expected)

def test_compute_decay_t1():
    """Test with T=1."""
    T = 1
    params = torch.tensor([1.0, 2.0])

    # indices = [0]
    # out[0, 0] = exp(0) = 1.0
    # out[0, 1] = exp(0) = 1.0

    expected = torch.tensor([[1.0, 1.0]])

    output = compute_decay(T, params)

    assert output.shape == (1, 2)
    assert torch.allclose(output, expected)

def test_compute_decay_invalid_clamp_lims():
    """Test that invalid clamp_lims raises an assertion error."""
    T = 2
    params = torch.tensor([1.0])

    # Test not a tuple
    with pytest.raises(AssertionError) as excinfo:
        compute_decay(T, params, clamp_lims=[0, 15])
    assert 'Clamp lims should be tuple' in str(excinfo.value)

    # Test empty tuple (currently it checks len(clamp_lims) which is truthy if not empty)
    # If it's an empty tuple, len is 0, which is falsy, so it should fail the first assert.
    with pytest.raises(AssertionError) as excinfo:
        compute_decay(T, params, clamp_lims=())
    assert 'Clamp lims should be length 2' in str(excinfo.value)

def test_compute_decay_device_consistency():
    """Test that the output is on the same device as the input params."""
    T = 3
    params = torch.tensor([1.0, 2.0])

    output = compute_decay(T, params)
    assert output.device == params.device
