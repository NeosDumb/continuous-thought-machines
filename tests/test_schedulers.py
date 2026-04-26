import torch
import pytest
import math
from utils.schedulers import WarmupCosineAnnealingLR

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
