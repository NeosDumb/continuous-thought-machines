import pytest
import torch
import numpy as np
from utils.samplers import QAMNISTSampler

class MockQAMNISTDataset:
    def __init__(self, num_samples=100, num_images_range=(1, 5), num_operations_range=(1, 3)):
        self.num_samples = num_samples
        self.num_images_range = num_images_range
        self.num_operations_range = num_operations_range
        self.num_digits_set = []
        self.num_operations_set = []

    def __len__(self):
        return self.num_samples

    def set_num_digits(self, num_digits):
        self.num_digits_set.append(num_digits)

    def set_num_operations(self, num_operations):
        self.num_operations_set.append(num_operations)

def test_qamnist_sampler_len():
    dataset = MockQAMNISTDataset(num_samples=100)
    sampler = QAMNISTSampler(dataset, batch_size=32)
    assert len(sampler) == 100 // 32

def test_qamnist_sampler_iter():
    dataset = MockQAMNISTDataset(num_samples=100, num_images_range=(2, 6), num_operations_range=(3, 7))
    sampler = QAMNISTSampler(dataset, batch_size=32)

    batches = list(sampler)

    assert len(batches) == 4 # 100 // 32 is 3 remainder 4, so 4 batches: 32, 32, 32, 4
    assert len(batches[0]) == 32
    assert len(batches[1]) == 32
    assert len(batches[2]) == 32
    assert len(batches[3]) == 4

    assert len(dataset.num_digits_set) == 4
    assert len(dataset.num_operations_set) == 4

    for num_digits in dataset.num_digits_set:
        assert 2 <= num_digits < 6

    for num_operations in dataset.num_operations_set:
        assert 3 <= num_operations < 7

def test_qamnist_sampler_fixed_ranges():
    dataset = MockQAMNISTDataset(num_samples=10, num_images_range=(5, 5), num_operations_range=(4, 4))
    sampler = QAMNISTSampler(dataset, batch_size=10)

    batches = list(sampler)

    assert len(batches) == 1

    assert len(dataset.num_digits_set) == 1
    assert len(dataset.num_operations_set) == 1

    assert dataset.num_digits_set[0] == 5
    assert dataset.num_operations_set[0] == 4
