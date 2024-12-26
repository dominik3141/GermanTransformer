import torch
import pytest
from src.nn import TransformerEncoder, AttentionBlock, AttentionHead


@pytest.fixture
def model_params():
    return {
        "max_sequence_length": 512,
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 6,
        "num_classes": 3,
        "dropout_rate": 0.1,
    }


@pytest.fixture
def sample_input():
    return ["This is a sample text for testing"]


def test_transformer_encoder_output_shape(model_params, sample_input):
    """Validates transformer encoder produces correct output shape and probability distribution"""
    model = TransformerEncoder(**model_params)
    model.eval()

    with torch.no_grad():
        output = model(sample_input)

    assert output.shape == (1, model_params["num_classes"])
    assert torch.allclose(output.sum(), torch.tensor(1.0), atol=1e-6)


def test_attention_block_shape(model_params):
    """Ensures attention block maintains expected tensor dimensions through transformation"""
    d_model = model_params["d_model"]
    batch_size = 2
    seq_length = 10

    block = AttentionBlock(
        d_model, model_params["num_heads"], dropout_rate=model_params["dropout_rate"]
    )
    x = torch.randn(batch_size, seq_length, d_model)

    output = block(x)
    assert output.shape == (batch_size, seq_length, d_model)


def test_attention_head_shape(model_params):
    """Verifies attention head correctly projects input to lower dimensional space"""
    d_model = model_params["d_model"]
    d_k = d_model // model_params["num_heads"]
    batch_size = 2
    seq_length = 10

    head = AttentionHead(d_model, d_k)
    x = torch.randn(batch_size, seq_length, d_model)

    output = head(x)
    assert output.shape == (batch_size, seq_length, d_k)


def test_transformer_encoder_gradient_flow(model_params, sample_input):
    """Confirms gradients propagate through all parameters of the transformer model"""
    model = TransformerEncoder(**model_params)
    output = model(sample_input)

    # Check if gradients flow through the model
    loss = output.sum()
    loss.backward()

    # Check if gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_attention_block_residual(model_params):
    """Verifies attention block performs meaningful transformation on input"""
    d_model = model_params["d_model"]
    batch_size = 2
    seq_length = 10

    block = AttentionBlock(
        d_model, model_params["num_heads"], dropout_rate=model_params["dropout_rate"]
    )
    x = torch.randn(batch_size, seq_length, d_model)
    x_copy = x.clone()

    output = block(x)
    assert not torch.allclose(output, x_copy)
