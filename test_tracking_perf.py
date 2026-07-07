import pytest
from models.ctm import ContinuousThoughtMachine
import time

def test_ctm_forward_perf(benchmark):
    kwargs = {
        "iterations": 10,
        "d_model": 64,
        "d_input": 64,
        "heads": 2,
        "n_synch_out": 8,
        "n_synch_action": 8,
        "synapse_depth": 1,
        "memory_length": 5,
        "deep_nlms": False,
        "memory_hidden_dims": 16,
        "do_layernorm_nlm": False,
        "backbone_type": "none",
        "positional_embedding_type": "none",
        "out_dims": 10,
        "neuron_select_type": "random-pairing",
    }
    model = ContinuousThoughtMachine(**kwargs)

    import torch
    x = torch.randn(128, 64, 10) # B, C, L

    def run_forward():
        model(x, track=True)

    benchmark(run_forward)
