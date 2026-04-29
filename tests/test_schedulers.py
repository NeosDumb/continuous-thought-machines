import torch
import pytest
import math
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR

def test_warmup_cosine_annealing_lr_basic():
    optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.1)
    warmup_epochs = 5
    max_epochs = 10
    warmup_start_lr = 0.01
    eta_min = 0.001

    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min
    )

    lrs = []
    # last_epoch starts at 0 (set by _LRScheduler.__init__ calling step())
    for epoch in range(11):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Based on implementation in utils/schedulers.py:
    # epoch 0: 0.01 (warmup_start_lr)
    # Warmup phase: epoch 1 to warmup_epochs-1
    # step = (base_lr - warmup_start_lr) / (warmup_epochs - 1) = (0.1 - 0.01) / 4 = 0.0225
    # epoch 1: 0.01 + 0.0225 = 0.0325
    # epoch 4: 0.1 (base_lr)
    # epoch 5: 0.1 (base_lr)
    # epoch 10: 0.001 (eta_min)

    assert math.isclose(lrs[0], 0.01, rel_tol=1e-5)
    assert math.isclose(lrs[1], 0.0325, rel_tol=1e-5)
    assert math.isclose(lrs[4], 0.1, rel_tol=1e-5)
    assert math.isclose(lrs[5], 0.1, rel_tol=1e-5)
    assert math.isclose(lrs[10], 0.001, rel_tol=1e-5)

def test_warmup_cosine_annealing_lr_multi_groups():
    optimizer = torch.optim.SGD([
        {'params': [torch.tensor([1.0])], 'lr': 0.1},
        {'params': [torch.tensor([1.0])], 'lr': 0.05}
    ])
    warmup_epochs = 5
    max_epochs = 10
    warmup_start_lr = 0.01
    eta_min = 0.001

    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min
    )

    for epoch in range(11):
        lr0 = optimizer.param_groups[0]['lr']
        lr1 = optimizer.param_groups[1]['lr']

        if epoch == 0:
            assert math.isclose(lr0, 0.01)
            assert math.isclose(lr1, 0.01)
        elif epoch == 5:
            assert math.isclose(lr0, 0.1)
            assert math.isclose(lr1, 0.05)
        elif epoch == 10:
            assert math.isclose(lr0, 0.001)
            assert math.isclose(lr1, 0.001)

        scheduler.step()

def test_warmup_cosine_annealing_lr_no_warmup():
    optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.1)
    # Implementation quirk: when warmup_epochs=0, it starts at warmup_start_lr and decays from there.
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=0, max_epochs=10, warmup_start_lr=0.01, eta_min=0.001)

    assert math.isclose(optimizer.param_groups[0]['lr'], 0.01)
    scheduler.step()
    # Should be decaying from 0.01
    assert optimizer.param_groups[0]['lr'] < 0.01

def test_warmup_cosine_annealing_lr_warmup_one():
    optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.1)
    # warmup_epochs = 1 avoids division by zero because last_epoch < 1 only for 0.
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=1, max_epochs=10, warmup_start_lr=0.01)

    assert math.isclose(optimizer.param_groups[0]['lr'], 0.01)
    scheduler.step()
    assert math.isclose(optimizer.param_groups[0]['lr'], 0.1)

def test_warmup_cosine_annealing_lr_periodicity():
    optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.1)
    warmup_epochs = 5
    max_epochs = 10
    eta_min = 0.001

    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs, eta_min=eta_min)

    for _ in range(21): # Go beyond max_epochs
        scheduler.step()

    # Ensure it remains within bounds [eta_min, base_lr]
    assert eta_min <= optimizer.param_groups[0]['lr'] <= 0.1 + 1e-6


def test_warmup_multi_step_lr_basic():
    optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.1)
    warmup_steps = 5
    milestones = [10, 15]
    gamma = 0.1

    scheduler = WarmupMultiStepLR(
        optimizer,
        warmup_steps=warmup_steps,
        milestones=milestones,
        gamma=gamma
    )

    lrs = []
    # Test for 20 epochs
    for epoch in range(20):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    # Based on our analysis:
    # epoch 0: 0.0
    # epoch 1: 0.02
    # epoch 5 to 9: 0.1 (warmup ends)
    # epoch 10 to 14: 0.01 (first milestone)
    # epoch 15 to 19: 0.001 (second milestone)

    assert math.isclose(lrs[0], 0.0, rel_tol=1e-5, abs_tol=1e-8)
    assert math.isclose(lrs[1], 0.02, rel_tol=1e-5)
    assert math.isclose(lrs[4], 0.08, rel_tol=1e-5)
    assert math.isclose(lrs[5], 0.1, rel_tol=1e-5)
    assert math.isclose(lrs[9], 0.1, rel_tol=1e-5)
    assert math.isclose(lrs[10], 0.01, rel_tol=1e-5)
    assert math.isclose(lrs[14], 0.01, rel_tol=1e-5)
    assert math.isclose(lrs[15], 0.001, rel_tol=1e-5)
    assert math.isclose(lrs[19], 0.001, rel_tol=1e-5)

def test_warmup_multi_step_lr_multi_groups():
    optimizer = torch.optim.SGD([
        {'params': [torch.tensor([1.0])], 'lr': 0.1},
        {'params': [torch.tensor([1.0])], 'lr': 0.05}
    ])
    warmup_steps = 5
    milestones = [10, 15]
    gamma = 0.1

    scheduler = WarmupMultiStepLR(
        optimizer,
        warmup_steps=warmup_steps,
        milestones=milestones,
        gamma=gamma
    )

    for epoch in range(20):
        lr0 = optimizer.param_groups[0]['lr']
        lr1 = optimizer.param_groups[1]['lr']

        if epoch == 0:
            assert math.isclose(lr0, 0.0, abs_tol=1e-8)
            assert math.isclose(lr1, 0.0, abs_tol=1e-8)
        elif epoch == 5: # Warmup ends
            assert math.isclose(lr0, 0.1)
            assert math.isclose(lr1, 0.05)
        elif epoch == 10: # First milestone
            assert math.isclose(lr0, 0.01)
            assert math.isclose(lr1, 0.005)
        elif epoch == 15: # Second milestone
            assert math.isclose(lr0, 0.001)
            assert math.isclose(lr1, 0.0005)

        optimizer.step()
        scheduler.step()

def test_warmup_multi_step_lr_state_dict():
    optimizer1 = torch.optim.SGD([torch.tensor([1.0])], lr=0.1)
    scheduler1 = WarmupMultiStepLR(
        optimizer1,
        warmup_steps=5,
        milestones=[10, 15],
        gamma=0.1
    )

    optimizer2 = torch.optim.SGD([torch.tensor([1.0])], lr=0.1)
    scheduler2 = WarmupMultiStepLR(
        optimizer2,
        warmup_steps=5,
        milestones=[10, 15],
        gamma=0.1
    )

    # Advance scheduler1
    for _ in range(12):
        optimizer1.step()
        scheduler1.step()

    # Save state
    state_dict = scheduler1.state_dict()

    # Load state into scheduler2
    scheduler2.load_state_dict(state_dict)

    # In PyTorch, learning rates are stored in the optimizer's param_groups.
    # When loading a scheduler state, it only updates internal scheduler states,
    # so we often need to manually apply the learning rate to the optimizer if resuming.
    # Alternatively, the next scheduler.step() might compute it based on the loaded state.
    # To properly simulate resumption from a checkpoint, the optimizer state should also be loaded,
    # but since we only load the scheduler state here, we can set the lr manually for testing.
    optimizer2.param_groups[0]['lr'] = optimizer1.param_groups[0]['lr']

    # Check that learning rates match after stepping
    optimizer1.step()
    scheduler1.step()

    # Step optimizer2 to see if the loaded scheduler computes the next lr identically
    optimizer2.step()
    scheduler2.step()

    assert math.isclose(optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr'])
