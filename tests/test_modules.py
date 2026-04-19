import torch
import torch.nn as nn
import pytest
from models.modules import Residual, Identity

def test_residual_identity():
    """Test Residual module with Identity function."""
    x = torch.randn(2, 4)
    model = Residual(Identity())
    output = model(x)

    # Expected: x + Identity(x) = 2x
    assert torch.allclose(output, 2 * x)

def test_residual_linear():
    """Test Residual module with a Linear layer."""
    d = 8
    x = torch.randn(2, d)
    linear = nn.Linear(d, d)
    model = Residual(linear)
    output = model(x)

    expected = x + linear(x)
    assert torch.allclose(output, expected)

def test_residual_complex():
    """Test Residual module with a Sequential block."""
    d = 16
    x = torch.randn(4, d)
    fn = nn.Sequential(
        nn.Linear(d, 2 * d),
        nn.ReLU(),
        nn.Linear(2 * d, d)
    )
    model = Residual(fn)
    output = model(x)

    expected = x + fn(x)
    assert torch.allclose(output, expected)

def test_residual_shape_preservation():
    """Test if Residual module preserves input shape."""
    shape = (3, 5, 7)
    x = torch.randn(*shape)
    model = Residual(Identity())
    output = model(x)

    assert output.shape == x.shape
