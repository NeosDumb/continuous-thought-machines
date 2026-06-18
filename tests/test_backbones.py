import torch
from models.modules import (
    MNISTBackbone,
    ShallowWide,
    ParityBackbone,
    MiniGridBackbone,
    ClassicControlBackbone
)

def test_mnist_backbone_shape():
    """Test MNISTBackbone output shape."""
    d_input = 64
    batch_size = 2
    model = MNISTBackbone(d_input)
    x = torch.randn(batch_size, 1, 28, 28)

    # Perform dummy forward pass to initialize Lazy layers
    output = model(x)

    # Expected: 28x28 -> MaxPool(14x14) -> MaxPool(7x7)
    assert output.shape == (batch_size, d_input, 7, 7)

def test_shallow_wide_shape():
    """Test ShallowWide output shape."""
    batch_size = 2
    model = ShallowWide()
    x = torch.randn(batch_size, 3, 32, 32)

    # Perform dummy forward pass to initialize Lazy layers
    output = model(x)

    # Expected: 32x32 -> Stride 2 (16x16) -> Output channels 2048 after GLU
    assert output.shape == (batch_size, 2048, 16, 16)

def test_parity_backbone_shape():
    """Test ParityBackbone output shape."""
    n_embeddings = 2
    d_embedding = 128
    batch_size = 2
    length = 10
    model = ParityBackbone(n_embeddings, d_embedding)
    # Parity input is usually 1 or -1
    x = torch.randint(0, 2, (batch_size, length)).float() * 2 - 1

    output = model(x)

    # Expected: (B, length) -> Embedding (B, length, d_embedding) -> Transpose (B, d_embedding, length)
    assert output.shape == (batch_size, d_embedding, length)

def test_minigrid_backbone_shape():
    """Test MiniGridBackbone output shape."""
    d_input = 64
    grid_size = 7
    batch_size = 2
    model = MiniGridBackbone(d_input, grid_size=grid_size)

    # Input: (B, H, W, C) where C=3 (object, color, state)
    # The maximum value for state is 2, since num_states=3
    x = torch.zeros(batch_size, grid_size, grid_size, 3, dtype=torch.long)
    x[:,:,:,0] = torch.randint(0, 11, (batch_size, grid_size, grid_size))
    x[:,:,:,1] = torch.randint(0, 6, (batch_size, grid_size, grid_size))
    x[:,:,:,2] = torch.randint(0, 3, (batch_size, grid_size, grid_size))

    output = model(x)

    # Expected: (B, H, W, d_input)
    assert output.shape == (batch_size, grid_size, grid_size, d_input)

def test_classic_control_backbone_shape():
    """Test ClassicControlBackbone output shape."""
    d_input = 64
    batch_size = 2
    input_dim = 4
    model = ClassicControlBackbone(d_input)
    x = torch.randn(batch_size, input_dim)

    # Perform dummy forward pass to initialize Lazy layers
    output = model(x)

    # Expected: (B, d_input)
    assert output.shape == (batch_size, d_input)
