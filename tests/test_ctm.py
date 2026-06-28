import torch
import pytest
from models.ctm import ContinuousThoughtMachine


def get_default_ctm_kwargs():
    return {
        "iterations": 2,
        "d_model": 32,
        "d_input": 16,
        "heads": 2,
        "n_synch_out": 4,
        "n_synch_action": 4,
        "synapse_depth": 1,
        "memory_length": 3,
        "deep_nlms": False,
        "memory_hidden_dims": 8,
        "do_layernorm_nlm": False,
        "backbone_type": "none",
        "positional_embedding_type": "none",
        "out_dims": 5,
        "neuron_select_type": "random-pairing",
    }


def test_ctm_init():
    """Test basic initialization of ContinuousThoughtMachine."""
    kwargs = get_default_ctm_kwargs()
    model = ContinuousThoughtMachine(**kwargs)

    assert model.iterations == kwargs["iterations"]
    assert model.d_model == kwargs["d_model"]
    assert model.out_dims == kwargs["out_dims"]
    assert model.neuron_select_type == kwargs["neuron_select_type"]


def test_ctm_forward_none_backbone():
    """Test forward pass with no backbone."""
    kwargs = get_default_ctm_kwargs()
    model = ContinuousThoughtMachine(**kwargs)

    batch_size = 2
    # input shape for 'none' backbone needs to match d_input when flattened,
    # or the initial input after backbone matches the attention expected shape.
    # Actually, for 'none', it just flattens input at dimension 2 and projects.
    # It takes x, then kv = compute_features(x)
    # self.initial_rgb = nn.Identity(), self.backbone = Identity(), pos_emb = 0
    # combined_features = x.flatten(2).transpose(1, 2)
    # kv = self.kv_proj(combined_features)
    # So x shape e.g. (B, C, L) where C*something isn't well defined.
    # It's (B, d_input, L)? wait, kv_proj expects self.d_input size in the last dim.
    # Let's check combined_features shape: flatten(2) makes it (B, C, H*W). transpose(1,2) -> (B, H*W, C).
    # Then kv_proj takes C. So we need C = d_input.
    C = kwargs["d_input"]
    L = 5
    x = torch.randn(
        batch_size, C, L
    )  # Shape: (B, C, L) => flatten(2) -> (B, C, L) -> transpose(1,2) -> (B, L, C)

    predictions, certainties, synchronisation_out = model(x)

    assert predictions.shape == (batch_size, kwargs["out_dims"], kwargs["iterations"])
    assert certainties.shape == (batch_size, 2, kwargs["iterations"])
    assert synchronisation_out.shape[0] == batch_size


def test_ctm_forward_resnet_backbone():
    """Test forward pass with a ResNet backbone."""
    kwargs = get_default_ctm_kwargs()
    kwargs["backbone_type"] = "resnet18-1"
    kwargs["d_input"] = 64  # resnet18-1 outputs 64 channels

    model = ContinuousThoughtMachine(**kwargs)

    batch_size = 1
    in_channels = 3
    height, width = 32, 32

    x = torch.randn(batch_size, in_channels, height, width)

    predictions, certainties, synchronisation_out = model(x)

    assert predictions.shape == (batch_size, kwargs["out_dims"], kwargs["iterations"])


@pytest.mark.parametrize("neuron_select", ["first-last", "random", "random-pairing"])
def test_ctm_neuron_select_types(neuron_select):
    """Test initialization and forward with different neuron selection strategies."""
    kwargs = get_default_ctm_kwargs()
    kwargs["neuron_select_type"] = neuron_select
    kwargs["d_model"] = 64  # ensure enough neurons

    model = ContinuousThoughtMachine(**kwargs)

    x = torch.randn(2, kwargs["d_input"], 5)

    predictions, certainties, synchronisation_out = model(x)
    assert predictions.shape == (2, kwargs["out_dims"], kwargs["iterations"])


def test_ctm_invalid_args():
    """Test argument validation in CTM."""
    kwargs = get_default_ctm_kwargs()

    # Invalid neuron select type
    with pytest.raises(AssertionError, match="Invalid neuron selection type"):
        ContinuousThoughtMachine(**{**kwargs, "neuron_select_type": "invalid_type"})

    # Invalid backbone type
    with pytest.raises(AssertionError, match="Invalid backbone_type"):
        ContinuousThoughtMachine(**{**kwargs, "backbone_type": "invalid_backbone"})

    # Invalid positional embedding type
    with pytest.raises(AssertionError, match="Invalid positional_embedding_type"):
        ContinuousThoughtMachine(
            **{**kwargs, "positional_embedding_type": "invalid_pos"}
        )

    # first-last requires d_model >= n_synch_out + n_synch_action
    with pytest.raises(
        AssertionError, match="d_model must be >= n_synch_out \\+ n_synch_action"
    ):
        ContinuousThoughtMachine(
            **{
                **kwargs,
                "neuron_select_type": "first-last",
                "d_model": 4,
                "n_synch_out": 3,
                "n_synch_action": 3,
            }
        )

    # None backbone with positional embedding
    with pytest.raises(
        AssertionError,
        match="There should be no positional embedding if there is no backbone",
    ):
        ContinuousThoughtMachine(
            **{
                **kwargs,
                "backbone_type": "none",
                "positional_embedding_type": "learnable-fourier",
            }
        )


def test_ctm_tracking():
    """Test that tracking returns the expected numpy arrays."""
    kwargs = get_default_ctm_kwargs()
    model = ContinuousThoughtMachine(**kwargs)

    x = torch.randn(1, kwargs["d_input"], 4)

    outputs = model(x, track=True)

    # Check that tracking returns a tuple of 6 elements
    assert len(outputs) == 6

    (
        predictions,
        certainties,
        synch_track,
        pre_activations,
        post_activations,
        attention_weights,
    ) = outputs

    assert isinstance(predictions, torch.Tensor)
    assert isinstance(certainties, torch.Tensor)

    # Check numpy array types
    assert (
        type(synch_track[0]).__name__ == "ndarray"
        or type(synch_track[0]).__name__ == "RealMockArray"
    )
    assert (
        type(synch_track[1]).__name__ == "ndarray"
        or type(synch_track[1]).__name__ == "RealMockArray"
    )
    assert (
        type(pre_activations).__name__ == "ndarray"
        or type(pre_activations).__name__ == "RealMockArray"
    )
    assert (
        type(post_activations).__name__ == "ndarray"
        or type(post_activations).__name__ == "RealMockArray"
    )
    assert (
        type(attention_weights).__name__ == "ndarray"
        or type(attention_weights).__name__ == "RealMockArray"
    )
