import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import numpy as np
from utils.losses import compute_ctc_loss, sort_loss, image_classification_loss, parity_loss, maze_loss

def test_maze_loss_basic():
    """Test basic functionality of maze_loss with standard inputs."""
    batch_size = 2
    route_length = 5
    num_classes = 5
    internal_ticks = 4

    # Predictions: [B, route_length, class, internal_ticks]
    predictions = torch.randn(batch_size, route_length, num_classes, internal_ticks)

    # Certainties: [B, 2, internal_ticks]
    certainties = torch.rand(batch_size, 2, internal_ticks)
    certainties[:, 0, :] = 1.0 - certainties[:, 1, :]

    # Targets: [B, route_length]
    targets = torch.randint(0, num_classes, (batch_size, route_length))

    # Test use_most_certain=True
    loss, loss_index_2, upto_where = maze_loss(predictions, certainties, targets, use_most_certain=True)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    assert isinstance(loss_index_2, torch.Tensor)
    assert loss_index_2.shape == (batch_size,)
    assert loss_index_2.dtype == torch.long

    assert type(upto_where).__name__ in ['ndarray', 'RealMockArray']
    assert upto_where.shape == (batch_size,)

def test_maze_loss_use_most_certain_false():
    """Test maze_loss when use_most_certain=False."""
    batch_size = 3
    route_length = 4
    num_classes = 5
    internal_ticks = 6

    predictions = torch.randn(batch_size, route_length, num_classes, internal_ticks)
    certainties = torch.rand(batch_size, 2, internal_ticks)
    targets = torch.randint(0, num_classes, (batch_size, route_length))

    loss, loss_index_2, upto_where = maze_loss(predictions, certainties, targets, use_most_certain=False)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

    # When use_most_certain=False, loss_index_2 should be all -1
    assert torch.all(loss_index_2 == -1)
    assert not torch.isnan(loss)

def test_maze_loss_auto_curriculum_logic():
    """Test the logic for auto-curriculum and upto_where."""
    batch_size = 2
    route_length = 10
    num_classes = 5
    internal_ticks = 1

    predictions = torch.randn(batch_size, route_length, num_classes, internal_ticks)
    targets = torch.randint(0, num_classes, (batch_size, route_length))
    certainties = torch.rand(batch_size, 2, internal_ticks)

    # For batch 0, let's make the first 3 predictions correct
    targets[0, 0:3] = predictions[0, 0:3, :, 0].argmax(dim=1)
    targets[0, 3] = (predictions[0, 3, :, 0].argmax(dim=0) + 1) % num_classes

    # For batch 1, let's make 0 predictions correct
    targets[1, 0] = (predictions[1, 0, :, 0].argmax(dim=0) + 1) % num_classes

    cirriculum_lookahead = 3
    loss, loss_index_2, upto_where = maze_loss(predictions, certainties, targets, cirriculum_lookahead=cirriculum_lookahead)

    # batch 0 is correct up to index 2 (length 3), so upto_where = 2 + lookahead
    assert upto_where[0] == 2 + cirriculum_lookahead
    # batch 1 is correct up to index -1 (length 0, handled as index 0 prefix length), so upto_where = 0 + lookahead
    assert upto_where[1] == 0 + cirriculum_lookahead

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

def test_parity_loss_basic():
    """Test basic functionality of parity_loss with standard inputs."""
    batch_size = 2
    parity_sequence_length = 5
    num_classes = 5
    internal_ticks = 4

    # Predictions: [B, parity_sequence_length, class, internal_ticks]
    predictions = torch.randn(batch_size, parity_sequence_length, num_classes, internal_ticks)

    # Certainties: [B, 2, internal_ticks]
    certainties = torch.rand(batch_size, 2, internal_ticks)
    certainties[:, 0, :] = 1.0 - certainties[:, 1, :]

    # Targets: [B, parity_sequence_length]
    targets = torch.randint(0, num_classes, (batch_size, parity_sequence_length))

    # Test use_most_certain=True
    loss, loss_index_2 = parity_loss(predictions, certainties, targets, use_most_certain=True)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    assert isinstance(loss_index_2, torch.Tensor)
    assert loss_index_2.shape == (batch_size,)
    assert loss_index_2.dtype == torch.long

def test_parity_loss_use_most_certain_false():
    """Test parity_loss when use_most_certain=False."""
    batch_size = 3
    parity_sequence_length = 4
    num_classes = 5
    internal_ticks = 6

    predictions = torch.randn(batch_size, parity_sequence_length, num_classes, internal_ticks)
    certainties = torch.rand(batch_size, 2, internal_ticks)
    targets = torch.randint(0, num_classes, (batch_size, parity_sequence_length))

    loss, loss_index_2 = parity_loss(predictions, certainties, targets, use_most_certain=False)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

    # When use_most_certain=False, loss_index_2 should be all -1
    assert torch.all(loss_index_2 == -1)
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

def test_image_classification_loss_basic():
    """Test basic functionality of image_classification_loss."""
    batch_size = 4
    num_classes = 10
    internal_ticks = 5

    predictions = torch.randn(batch_size, num_classes, internal_ticks)
    certainties = torch.rand(batch_size, 2, internal_ticks)
    # Ensure probabilities sum to 1 roughly (not strictly required but good for semantics)
    certainties = certainties / certainties.sum(dim=1, keepdim=True)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss, loss_index_2 = image_classification_loss(predictions, certainties, targets, use_most_certain=True)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert not torch.isnan(loss)
    assert loss_index_2.shape == (batch_size,)
    # Since use_most_certain=True, the indices should be between 0 and internal_ticks-1
    assert (loss_index_2 >= 0).all() and (loss_index_2 < internal_ticks).all()

def test_image_classification_loss_use_most_certain_false():
    """Test functionality when use_most_certain is False."""
    batch_size = 4
    num_classes = 10
    internal_ticks = 5

    predictions = torch.randn(batch_size, num_classes, internal_ticks)
    certainties = torch.rand(batch_size, 2, internal_ticks)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss, loss_index_2 = image_classification_loss(predictions, certainties, targets, use_most_certain=False)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss_index_2.shape == (batch_size,)
    # Since use_most_certain=False, the indices should be -1
    assert (loss_index_2 == -1).all()

def test_image_classification_loss_logic():
    """Test the logic of image_classification_loss manually."""
    batch_size = 2
    num_classes = 3
    internal_ticks = 2

    # Predictions: [B, C, ticks]
    # Set it up so we know exact losses
    # B=0:
    # Target = 0
    # Tick 0: C=[10, 0, 0] -> very low loss
    # Tick 1: C=[0, 10, 0] -> high loss
    # B=1:
    # Target = 1
    # Tick 0: C=[10, 0, 0] -> high loss
    # Tick 1: C=[0, 10, 0] -> very low loss
    predictions = torch.zeros(batch_size, num_classes, internal_ticks)
    predictions[0, 0, 0] = 10.0
    predictions[0, 1, 1] = 10.0
    predictions[1, 0, 0] = 10.0
    predictions[1, 1, 1] = 10.0

    targets = torch.tensor([0, 1])

    # Certainties: [B, 2, ticks]
    # B=0: Most certain at tick 0 (certainties[0, 1, 0] > certainties[0, 1, 1])
    # B=1: Most certain at tick 1
    certainties = torch.zeros(batch_size, 2, internal_ticks)
    certainties[0, 1, 0] = 0.9
    certainties[0, 1, 1] = 0.1
    certainties[1, 1, 0] = 0.1
    certainties[1, 1, 1] = 0.9

    loss, loss_index_2 = image_classification_loss(predictions, certainties, targets, use_most_certain=True)

    # Minimum CE should pick tick 0 for B=0, tick 1 for B=1
    # Both minimum CE losses are close to 0
    # Certainties will also pick tick 0 for B=0, tick 1 for B=1
    # Both selected losses are close to 0
    assert loss.item() < 0.01

    assert loss_index_2[0].item() == 0
    assert loss_index_2[1].item() == 1


def test_qamnist_loss_basic():
    """Test basic functionality of qamnist_loss."""
    batch_size = 4
    num_classes = 10
    internal_ticks = 5

    predictions = torch.randn(batch_size, num_classes, internal_ticks)
    certainties = torch.rand(batch_size, 2, internal_ticks)
    certainties = certainties / certainties.sum(dim=1, keepdim=True)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss, loss_index_2 = qamnist_loss(predictions, certainties, targets, use_most_certain=True)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert not torch.isnan(loss)
    assert loss_index_2.shape == (batch_size,)
    assert (loss_index_2 >= 0).all() and (loss_index_2 < internal_ticks).all()


def test_qamnist_loss_use_most_certain_false():
    """Test functionality when use_most_certain is False."""
    batch_size = 4
    num_classes = 10
    internal_ticks = 5

    predictions = torch.randn(batch_size, num_classes, internal_ticks)
    certainties = torch.rand(batch_size, 2, internal_ticks)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss, loss_index_2 = qamnist_loss(predictions, certainties, targets, use_most_certain=False)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss_index_2.shape == (batch_size,)
    assert (loss_index_2 == -1).all()


def test_qamnist_loss_logic():
    """Test the logic of qamnist_loss manually."""
    batch_size = 2
    num_classes = 3
    internal_ticks = 2

    # Predictions: [B, C, ticks]
    predictions = torch.zeros(batch_size, num_classes, internal_ticks)
    # Target 0: B=0, tick 0 -> C=[10, 0, 0] (low loss)
    predictions[0, 0, 0] = 10.0
    predictions[0, 1, 1] = 10.0
    # Target 1: B=1, tick 1 -> C=[0, 10, 0] (low loss)
    predictions[1, 0, 0] = 10.0
    predictions[1, 1, 1] = 10.0

    targets = torch.tensor([0, 1])

    # Certainties: [B, 2, ticks]
    certainties = torch.zeros(batch_size, 2, internal_ticks)
    certainties[0, 1, 0] = 0.9
    certainties[0, 1, 1] = 0.1
    certainties[1, 1, 0] = 0.1
    certainties[1, 1, 1] = 0.9

    loss, loss_index_2 = qamnist_loss(predictions, certainties, targets, use_most_certain=True)

    assert loss.item() < 0.01

    assert loss_index_2[0].item() == 0
    assert loss_index_2[1].item() == 1
