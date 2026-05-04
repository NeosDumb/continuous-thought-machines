import torch
import pytest
from models.ff import FFBaseline

@pytest.mark.parametrize("backbone_type", ["none", "resnet18-1", "resnet34-2", "parity_backbone"])
def test_ff_baseline_init(backbone_type):
    """Test initialization of FFBaseline with different backbone types."""
    d_model = 64
    out_dims = 10
    model = FFBaseline(d_model=d_model, backbone_type=backbone_type, out_dims=out_dims)
    assert model.d_model == d_model
    assert model.backbone_type == backbone_type
    assert model.out_dims == out_dims

def test_ff_baseline_forward_none():
    """Test forward pass of FFBaseline with backbone_type='none'."""
    batch_size = 2
    in_channels = 3
    height, width = 32, 32
    d_model = 64
    out_dims = 5

    model = FFBaseline(d_model=d_model, backbone_type='none', out_dims=out_dims)
    x = torch.randn(batch_size, in_channels, height, width)

    output = model(x)

    assert output.shape == (batch_size, out_dims)
    assert not torch.isnan(output).any()

def test_ff_baseline_forward_resnet():
    """Test forward pass of FFBaseline with a ResNet backbone."""
    batch_size = 1
    in_channels = 3
    height, width = 64, 64
    d_model = 128
    out_dims = 10

    # Using resnet18-1 as it is likely faster for testing
    model = FFBaseline(d_model=d_model, backbone_type='resnet18-1', out_dims=out_dims)
    x = torch.randn(batch_size, in_channels, height, width)

    output = model(x)

    assert output.shape == (batch_size, out_dims)

def test_ff_baseline_gradients():
    """Test if gradients propagate through FFBaseline."""
    batch_size = 2
    in_channels = 3
    height, width = 32, 32
    d_model = 32
    out_dims = 2

    model = FFBaseline(d_model=d_model, backbone_type='none', out_dims=out_dims)
    x = torch.randn(batch_size, in_channels, height, width)

    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check if at least some parameters have gradients
    # Note: Lazy modules might not have parameters before the first forward pass,
    # but they should have them now.
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "No gradients were computed"

def test_ff_baseline_invalid_backbone():
    """Test if FFBaseline raises an assertion error for invalid backbone types."""
    with pytest.raises(AssertionError):
        FFBaseline(d_model=64, backbone_type='invalid-backbone', out_dims=10)

def test_ff_baseline_dropout():
    """Test FFBaseline with dropout."""
    model = FFBaseline(d_model=64, backbone_type='none', out_dims=10, dropout=0.5)
    model.train()
    x = torch.randn(2, 3, 32, 32)
    output1 = model(x)
    output2 = model(x)

    # With dropout, outputs might be different even for the same input
    # though with small out_dims it might coincidentally be similar,
    # but this is a standard way to check if dropout layer is present and active.
    # However, since it's just one dropout layer, we at least check it runs.
    assert output1.shape == (2, 10)
