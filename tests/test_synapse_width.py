import torch
from models.ctm import ContinuousThoughtMachine
from models.ctm_sort import ContinuousThoughtMachineSORT
from models.ctm_rl import ContinuousThoughtMachineRL
from models.ctm_qamnist import ContinuousThoughtMachineQAMNIST
from models.modules import SynapseUNET

def test_synapse_min_width_ctm():
    d_model = 64
    synapse_depth = 2
    synapse_min_width = 32
    model = ContinuousThoughtMachine(
        iterations=1,
        d_model=d_model,
        d_input=32,
        heads=1,
        n_synch_out=1,
        n_synch_action=1,
        synapse_depth=synapse_depth,
        memory_length=1,
        deep_nlms=False,
        memory_hidden_dims=1,
        do_layernorm_nlm=False,
        backbone_type='none',
        positional_embedding_type='none',
        out_dims=1,
        synapse_min_width=synapse_min_width
    )

    assert isinstance(model.synapses, SynapseUNET)
    # With depth=2, widths = linspace(d_model, synapse_min_width, 2) = [d_model, synapse_min_width]
    # down_projections[0] should be Linear(d_model, synapse_min_width)
    assert model.synapses.down_projections[0][1].out_features == synapse_min_width

def test_synapse_min_width_sort():
    d_model = 64
    synapse_depth = 2
    synapse_min_width = 32
    model = ContinuousThoughtMachineSORT(
        iterations=1,
        d_model=d_model,
        d_input=32,
        heads=0,
        n_synch_out=1,
        n_synch_action=0,
        synapse_depth=synapse_depth,
        memory_length=1,
        deep_nlms=False,
        memory_hidden_dims=1,
        do_layernorm_nlm=False,
        backbone_type='none',
        positional_embedding_type='none',
        out_dims=1,
        synapse_min_width=synapse_min_width
    )
    assert isinstance(model.synapses, SynapseUNET)
    assert model.synapses.down_projections[0][1].out_features == synapse_min_width

def test_synapse_min_width_rl():
    d_model = 64
    synapse_depth = 2
    synapse_min_width = 32
    model = ContinuousThoughtMachineRL(
        iterations=1,
        d_model=d_model,
        d_input=32,
        n_synch_out=1,
        synapse_depth=synapse_depth,
        memory_length=1,
        deep_nlms=False,
        memory_hidden_dims=1,
        do_layernorm_nlm=False,
        backbone_type='classic-control-backbone',
        synapse_min_width=synapse_min_width
    )
    assert isinstance(model.synapses, SynapseUNET)
    assert model.synapses.down_projections[0][1].out_features == synapse_min_width

def test_synapse_min_width_qamnist():
    d_model = 64
    synapse_depth = 2
    synapse_min_width = 32
    model = ContinuousThoughtMachineQAMNIST(
        iterations=1,
        d_model=d_model,
        d_input=32,
        heads=1,
        n_synch_out=1,
        n_synch_action=1,
        synapse_depth=synapse_depth,
        memory_length=1,
        deep_nlms=False,
        memory_hidden_dims=1,
        do_layernorm_nlm=False,
        out_dims=1,
        iterations_per_digit=1,
        iterations_per_question_part=1,
        iterations_for_answering=1,
        synapse_min_width=synapse_min_width
    )
    assert isinstance(model.synapses, SynapseUNET)
    assert model.synapses.down_projections[0][1].out_features == synapse_min_width

if __name__ == "__main__":
    test_synapse_min_width_ctm()
    test_synapse_min_width_sort()
    test_synapse_min_width_rl()
    test_synapse_min_width_qamnist()
    print("All synapse width tests passed!")
