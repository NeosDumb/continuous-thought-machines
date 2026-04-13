import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from utils.losses import compute_ctc_loss, sort_loss

def test_compute_ctc_loss_basic():
    """Test basic functionality of compute_ctc_loss with standard inputs."""
    batch_size = 2
    num_classes = 5
    prediction_length = 10
    target_length = 4
    blank_label = 0

    # Predictions: [B, C, L] - logits
    predictions = torch.randn(batch_size, num_classes, prediction_length)
    # Targets: [B, T] - indices
    targets = torch.randint(1, num_classes, (batch_size, target_length))

    loss = compute_ctc_loss(predictions, targets, blank_label=blank_label)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_compute_ctc_loss_consistency():
    """Verify compute_ctc_loss matches torch.nn.CTCLoss directly."""
    batch_size = 2
    num_classes = 4
    prediction_length = 15
    target_length = 8
    blank_label = 3

    predictions = torch.randn(batch_size, num_classes, prediction_length)
    # Ensure targets don't contain the blank label
    targets = torch.randint(0, blank_label, (batch_size, target_length))

    loss = compute_ctc_loss(predictions, targets, blank_label=blank_label)

    # Manual calculation for verification
    log_probs = F.log_softmax(predictions, dim=1).permute(2, 0, 1)  # [L, B, C]
    input_lengths = torch.full((batch_size,), prediction_length, dtype=torch.long)
    target_lengths = torch.full((batch_size,), target_length, dtype=torch.long)

    ctc_loss_fn = nn.CTCLoss(blank=blank_label, reduction='mean')
    expected_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

    assert torch.allclose(loss, expected_loss)

def test_compute_ctc_loss_variable_target_lengths():
    """Test with a list of target tensors with different lengths."""
    batch_size = 3
    num_classes = 10
    prediction_length = 20
    blank_label = 0

    predictions = torch.randn(batch_size, num_classes, prediction_length)
    # Targets as a list of tensors of varying lengths
    targets = [
        torch.randint(1, num_classes, (5,)),
        torch.randint(1, num_classes, (10,)),
        torch.randint(1, num_classes, (15,))
    ]

    loss = compute_ctc_loss(predictions, targets, blank_label=blank_label)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

def test_compute_ctc_loss_edge_cases():
    """Test edge cases such as minimum sequence lengths."""
    batch_size = 1
    num_classes = 2
    prediction_length = 2
    target_length = 2
    blank_label = 0

    # For CTC, prediction_length must be >= target_length
    predictions = torch.randn(batch_size, num_classes, prediction_length)
    targets = torch.tensor([[1, 1]])

    loss = compute_ctc_loss(predictions, targets, blank_label=blank_label)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

def test_compute_ctc_loss_shapes():
    """Test various input dimensions for batch size, classes, and lengths."""
    shapes = [
        (1, 2, 5, 2),     # Single batch, small classes/length
        (4, 10, 20, 15),  # Medium setup
        (2, 5, 30, 5),    # Long predictions, short targets
    ]
    for B, C, L, T in shapes:
        predictions = torch.randn(B, C, L)
        targets = torch.randint(1, C, (B, T))
        loss = compute_ctc_loss(predictions, targets, blank_label=0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

def test_sort_loss():
    """Test the sort_loss wrapper function."""
    batch_size = 2
    N = 5
    prediction_length = 10

    # sort_loss assumes blank_label is predictions.shape[1] - 1
    # num_classes should be N + 1
    predictions = torch.randn(batch_size, N + 1, prediction_length)
    targets = torch.randint(0, N, (batch_size, N))

    loss = sort_loss(predictions, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
