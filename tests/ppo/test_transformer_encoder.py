import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import pytest
import torch
import torch.nn as nn

from src.env_definitions import BOARD_DIM
from src.ppo.transformer_encoder import (
    PositionalEncoding2D,
    TransformerEncoder,
    get_emb,
)


class TestGetEmb:
    """Test class for get_emb function."""

    def test_get_emb_shape(self):
        """Test that get_emb returns correct shape."""
        # Test with various input shapes
        sin_inp = torch.randn(10, 64)
        result = get_emb(sin_inp)

        # Output should have twice the channels (sin and cos interleaved)
        assert result.shape == (10, 128)

    def test_get_emb_single_value(self):
        """Test get_emb with a single value."""
        sin_inp = torch.tensor([[1.0]])
        result = get_emb(sin_inp)

        expected_shape = (1, 2)  # sin and cos of single value
        assert result.shape == expected_shape

        # Check that it contains both sin and cos
        assert torch.allclose(result[0, 0], torch.sin(torch.tensor(1.0)), atol=1e-6)
        assert torch.allclose(result[0, 1], torch.cos(torch.tensor(1.0)), atol=1e-6)

    def test_get_emb_zero_input(self):
        """Test get_emb with zero input."""
        sin_inp = torch.zeros(5, 10)
        result = get_emb(sin_inp)

        # sin(0) = 0, cos(0) = 1
        expected = torch.zeros(5, 20)
        expected[:, 1::2] = 1.0  # Set cos values to 1

        assert torch.allclose(result, expected, atol=1e-6)

    def test_get_emb_gradient_flow(self):
        """Test that gradients flow through get_emb."""
        sin_inp = torch.randn(3, 4, requires_grad=True)
        result = get_emb(sin_inp)
        loss = result.sum()
        loss.backward()

        assert sin_inp.grad is not None
        assert not torch.allclose(sin_inp.grad, torch.zeros_like(sin_inp.grad))


class TestPositionalEncoding2D:
    """Test class for PositionalEncoding2D functionality."""

    @pytest.fixture
    def pos_encoding(self):
        """Create a PositionalEncoding2D instance for testing."""
        return PositionalEncoding2D(x_size=4, y_size=4, channels=64, dropout=0.1)

    @pytest.fixture
    def sample_tensor_4d(self):
        """Create sample 4D tensor for testing."""
        batch_size = 8
        return torch.randn(batch_size, 4, 4, 64)

    @pytest.fixture
    def sample_tensor_3d(self):
        """Create sample 3D tensor for testing."""
        batch_size = 8
        seq_len = 16  # 4x4 flattened
        channels = 64
        return torch.randn(batch_size, seq_len, channels)

    def test_initialization(self, pos_encoding):
        """Test that PositionalEncoding2D initializes correctly."""
        assert isinstance(pos_encoding, nn.Module)
        assert pos_encoding.org_channels == 64
        assert isinstance(pos_encoding.dropout, nn.Dropout)

        # Check that positional encoding buffer is created
        assert hasattr(pos_encoding, "pe")
        assert pos_encoding.pe.shape == (1, 4, 4, pos_encoding.channels * 2)

    def test_initialization_custom_params(self):
        """Test PositionalEncoding2D initialization with custom parameters."""
        pos_enc = PositionalEncoding2D(x_size=8, y_size=6, channels=128, dropout=0.2)
        assert pos_enc.org_channels == 128
        assert pos_enc.pe.shape[1] == 8  # x_size
        assert pos_enc.pe.shape[2] == 6  # y_size

    def test_forward_4d_shape(self, pos_encoding, sample_tensor_4d):
        """Test forward pass with 4D tensor returns correct shape."""
        output = pos_encoding.forward(sample_tensor_4d)
        assert output.shape == sample_tensor_4d.shape

    def test_forward_4d_adds_positional_encoding(self, pos_encoding):
        """Test that forward adds positional encoding to input."""
        input_tensor = torch.zeros(2, 4, 4, 64)
        output = pos_encoding.forward(input_tensor)

        # Output should not be zero (positional encoding added)
        # Note: dropout might make some values zero, so we check that not all are zero
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_forward_flat_shape(self, pos_encoding, sample_tensor_3d):
        """Test forward_flat returns correct shape."""
        output = pos_encoding.forward_flat(sample_tensor_3d)
        assert output.shape == sample_tensor_3d.shape

    def test_forward_flat_adds_positional_encoding(self, pos_encoding):
        """Test that forward_flat adds positional encoding to flattened input."""
        input_tensor = torch.zeros(3, 16, 64)  # 4x4=16 flattened
        output = pos_encoding.forward_flat(input_tensor)

        # Output should not be zero (positional encoding added)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_forward_with_inds_shape(self, pos_encoding):
        """Test forward_with_inds returns correct shape."""
        batch_size, seq_len = 4, 8
        x = torch.randn(batch_size, seq_len, 64)
        inds = torch.randint(
            0, 16, (batch_size, seq_len)
        )  # Valid indices for 4x4 board

        output = pos_encoding.forward_with_inds(x, inds)
        assert output.shape == x.shape

    def test_forward_with_inds_valid_indices(self, pos_encoding):
        """Test forward_with_inds with valid indices."""
        batch_size, seq_len = 2, 4
        x = torch.zeros(batch_size, seq_len, 64)
        inds = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)

        output = pos_encoding.forward_with_inds(x, inds)

        # Should add positional encoding based on indices
        assert not torch.allclose(output, torch.zeros_like(output))
        assert torch.all(torch.isfinite(output))

    def test_different_input_sizes(self):
        """Test with different input sizes."""
        sizes = [(2, 2, 32), (6, 8, 128), (1, 1, 16)]

        for x_size, y_size, channels in sizes:
            pos_enc = PositionalEncoding2D(x_size, y_size, channels)

            # Test 4D forward
            input_4d = torch.randn(3, x_size, y_size, channels)
            output_4d = pos_enc.forward(input_4d)
            assert output_4d.shape == input_4d.shape

            # Test flat forward
            input_flat = torch.randn(3, x_size * y_size, channels)
            output_flat = pos_enc.forward_flat(input_flat)
            assert output_flat.shape == input_flat.shape

    def test_dropout_behavior(self):
        """Test dropout behavior in training vs eval mode."""
        pos_enc = PositionalEncoding2D(4, 4, 64, dropout=0.5)
        input_tensor = torch.ones(2, 4, 4, 64)

        # Training mode - dropout active
        pos_enc.train()
        output_train1 = pos_enc.forward(input_tensor)
        output_train2 = pos_enc.forward(input_tensor)

        # Eval mode - no dropout
        pos_enc.eval()
        with torch.no_grad():
            output_eval1 = pos_enc.forward(input_tensor)
            output_eval2 = pos_enc.forward(input_tensor)

            # Eval mode should be deterministic
            assert torch.allclose(output_eval1, output_eval2, atol=1e-6)

    def test_gradient_flow(self, pos_encoding, sample_tensor_3d):
        """Test that gradients flow through positional encoding."""
        input_tensor = sample_tensor_3d.clone().requires_grad_(True)
        output = pos_encoding.forward_flat(input_tensor)
        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None
        assert not torch.allclose(
            input_tensor.grad, torch.zeros_like(input_tensor.grad)
        )

    def test_positional_encoding_properties(self, pos_encoding):
        """Test properties of the positional encoding."""
        # Positional encoding should be finite
        assert torch.all(torch.isfinite(pos_encoding.pe))

        # Should have correct shape
        assert pos_encoding.pe.shape == (1, 4, 4, pos_encoding.channels * 2)

        # Different positions should have different encodings
        pe_flat = pos_encoding.pe.reshape(1, -1, pos_encoding.channels * 2)
        for i in range(min(10, pe_flat.shape[1])):
            for j in range(i + 1, min(10, pe_flat.shape[1])):
                assert not torch.allclose(pe_flat[0, i], pe_flat[0, j], atol=1e-6)

    def test_channels_adjustment(self):
        """Test that channels are adjusted correctly for positional encoding."""
        # Test with odd number of channels
        pos_enc_odd = PositionalEncoding2D(4, 4, 63)  # odd number
        assert pos_enc_odd.org_channels == 63

        # Test with even number
        pos_enc_even = PositionalEncoding2D(4, 4, 64)  # even number
        assert pos_enc_even.org_channels == 64

    def test_device_compatibility(self, pos_encoding):
        """Test positional encoding works on different devices."""
        # Test CPU
        cpu_input = torch.randn(2, 16, 64)
        cpu_output = pos_encoding.forward_flat(cpu_input)
        assert cpu_output.device == torch.device("cpu")

        # Test GPU if available
        if torch.cuda.is_available():
            pos_encoding_gpu = pos_encoding.cuda()
            gpu_input = cpu_input.cuda()
            gpu_output = pos_encoding_gpu.forward_flat(gpu_input)
            assert gpu_output.device.type == "cuda"


class TestTransformerEncoder:
    """Test class for TransformerEncoder functionality."""

    @pytest.fixture
    def transformer_encoder(self):
        """Create a TransformerEncoder instance for testing."""
        return TransformerEncoder(
            d_model=128,
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.1,
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input for transformer encoder."""
        batch_size = 4
        seq_len = 16  # 4x4 flattened board
        d_model = 128
        return torch.randn(batch_size, seq_len, d_model)

    def test_initialization(self, transformer_encoder):
        """Test that TransformerEncoder initializes correctly."""
        assert isinstance(transformer_encoder, nn.Module)
        assert transformer_encoder.d_model == 128
        assert transformer_encoder.nhead == 8
        assert transformer_encoder.num_layers == 2
        assert transformer_encoder.dim_feedforward == 512
        assert transformer_encoder.dropout == 0.1

        # Check components
        assert isinstance(transformer_encoder.positional_encoding, PositionalEncoding2D)
        assert isinstance(transformer_encoder.encoder, nn.TransformerEncoder)

    def test_initialization_default_dropout(self):
        """Test TransformerEncoder initialization with default dropout."""
        encoder = TransformerEncoder(
            d_model=64, nhead=4, num_layers=1, dim_feedforward=256
        )
        assert encoder.dropout == 0.1  # default value

    def test_initialization_custom_params(self):
        """Test TransformerEncoder initialization with custom parameters."""
        encoder = TransformerEncoder(
            d_model=256,
            nhead=16,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.2,
        )
        assert encoder.d_model == 256
        assert encoder.nhead == 16
        assert encoder.num_layers == 6
        assert encoder.dim_feedforward == 1024
        assert encoder.dropout == 0.2

    def test_forward_shape(self, transformer_encoder, sample_input):
        """Test that forward pass returns correct shape."""
        output = transformer_encoder.forward(sample_input)
        # With default reduction="mean", output shape should be (batch_size, d_model)
        expected_shape = (sample_input.shape[0], sample_input.shape[2])
        assert output.shape == expected_shape

    def test_forward_output_properties(self, transformer_encoder, sample_input):
        """Test properties of forward pass output."""
        output = transformer_encoder.forward(sample_input)

        # Output should be finite
        assert torch.all(torch.isfinite(output))

        # Output should have same device as input
        assert output.device == sample_input.device

        # Output should have same dtype as input
        assert output.dtype == sample_input.dtype

    def test_forward_different_batch_sizes(self, transformer_encoder):
        """Test forward pass with different batch sizes."""
        batch_sizes = [1, 4, 8, 16]
        seq_len, d_model = 16, 128

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            output = transformer_encoder.forward(input_tensor)
            assert output.shape == (batch_size, d_model)

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        encoder = TransformerEncoder(
            d_model=64, nhead=4, num_layers=1, dim_feedforward=256
        )
        batch_size, d_model = 4, 64

        # Note: For this specific encoder, sequence length should be 16 (4x4)
        # since positional encoding is fixed to BOARD_DIM[0] x BOARD_DIM[1]
        seq_len = BOARD_DIM[0] * BOARD_DIM[1]
        input_tensor = torch.randn(batch_size, seq_len, d_model)
        output = encoder.forward(input_tensor)
        assert output.shape == (batch_size, d_model)

    def test_gradient_flow(self, transformer_encoder, sample_input):
        """Test that gradients flow through the transformer."""
        input_tensor = sample_input.clone().requires_grad_(True)
        output = transformer_encoder.forward(input_tensor)
        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None
        # Check that at least some gradients are non-zero (above numerical precision)
        assert torch.any(torch.abs(input_tensor.grad) > 1e-9)

        # Check that transformer parameters have gradients
        for name, param in transformer_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"

    def test_parameter_count(self, transformer_encoder):
        """Test that transformer has reasonable number of parameters."""
        total_params = sum(p.numel() for p in transformer_encoder.parameters())
        trainable_params = sum(
            p.numel() for p in transformer_encoder.parameters() if p.requires_grad
        )

        # Should have some parameters
        assert total_params > 0
        assert trainable_params > 0

        # Most parameters should be trainable (positional encoding is not trainable)
        assert trainable_params <= total_params

    def test_training_vs_eval_mode(self, transformer_encoder, sample_input):
        """Test behavior difference between training and eval modes."""
        # Training mode
        transformer_encoder.train()
        output_train = transformer_encoder.forward(sample_input)

        # Eval mode
        transformer_encoder.eval()
        with torch.no_grad():
            output_eval = transformer_encoder.forward(sample_input)

        # Outputs might be different due to dropout
        assert output_train.shape == output_eval.shape
        assert torch.all(torch.isfinite(output_train))
        assert torch.all(torch.isfinite(output_eval))

    def test_deterministic_output_eval_mode(self, transformer_encoder, sample_input):
        """Test that eval mode produces deterministic output."""
        transformer_encoder.eval()

        with torch.no_grad():
            output1 = transformer_encoder.forward(sample_input)
            output2 = transformer_encoder.forward(sample_input)

            # Should be identical in eval mode
            assert torch.allclose(output1, output2, atol=1e-6)

    def test_attention_mask_compatibility(self, transformer_encoder):
        """Test that transformer works without attention mask."""
        batch_size, seq_len, d_model = 4, 16, 128
        input_tensor = torch.randn(batch_size, seq_len, d_model)

        # Should work without explicit attention mask
        output = transformer_encoder.forward(input_tensor)
        assert output.shape == (batch_size, d_model)

    def test_positional_encoding_integration(self, transformer_encoder, sample_input):
        """Test that positional encoding is properly integrated."""
        # Create zero input to see effect of positional encoding
        zero_input = torch.zeros_like(sample_input)
        output = transformer_encoder.forward(zero_input)

        # Output should not be zero due to positional encoding
        # (though transformer layers might produce some zeros)
        assert torch.any(output != 0)

    def test_different_model_dimensions(self):
        """Test transformer with different d_model dimensions."""
        d_models = [64, 128, 256, 512]

        for d_model in d_models:
            # nhead must divide d_model evenly
            nhead = 4 if d_model >= 64 else 2
            encoder = TransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=1,
                dim_feedforward=d_model * 2,
            )

            input_tensor = torch.randn(2, 16, d_model)
            output = encoder.forward(input_tensor)
            assert output.shape == (2, d_model)

    def test_different_head_counts(self):
        """Test transformer with different numbers of attention heads."""
        d_model = 128
        head_counts = [1, 2, 4, 8, 16]

        for nhead in head_counts:
            if d_model % nhead == 0:  # nhead must divide d_model evenly
                encoder = TransformerEncoder(
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=1,
                    dim_feedforward=512,
                )

                input_tensor = torch.randn(2, 16, d_model)
                output = encoder.forward(input_tensor)
                assert output.shape == (2, d_model)

    def test_different_layer_counts(self):
        """Test transformer with different numbers of layers."""
        layer_counts = [1, 2, 4, 6]

        for num_layers in layer_counts:
            encoder = TransformerEncoder(
                d_model=64,
                nhead=4,
                num_layers=num_layers,
                dim_feedforward=256,
            )

            input_tensor = torch.randn(2, 16, 64)
            output = encoder.forward(input_tensor)
            assert output.shape == (2, 64)

    def test_state_dict_save_load(self, transformer_encoder, sample_input):
        """Test saving and loading transformer state."""
        transformer_encoder.eval()

        with torch.no_grad():
            # Get initial output
            initial_output = transformer_encoder.forward(sample_input)

            # Save state dict
            state_dict = transformer_encoder.state_dict()

            # Create new transformer and load state
            new_encoder = TransformerEncoder(
                d_model=128,
                nhead=8,
                num_layers=2,
                dim_feedforward=512,
                dropout=0.1,
            )
            new_encoder.load_state_dict(state_dict)
            new_encoder.eval()

            # Check outputs are identical
            new_output = new_encoder.forward(sample_input)
            assert torch.allclose(initial_output, new_output, atol=1e-6)

    def test_device_compatibility(self, transformer_encoder):
        """Test transformer works on different devices."""
        # Test CPU
        cpu_input = torch.randn(2, 16, 128)
        cpu_output = transformer_encoder.forward(cpu_input)
        assert cpu_output.device == torch.device("cpu")

        # Test GPU if available
        if torch.cuda.is_available():
            encoder_gpu = transformer_encoder.cuda()
            gpu_input = cpu_input.cuda()
            gpu_output = encoder_gpu.forward(gpu_input)
            assert gpu_output.device.type == "cuda"

    def test_empty_batch(self, transformer_encoder):
        """Test behavior with empty batch."""
        empty_input = torch.randn(0, 16, 128)
        output = transformer_encoder.forward(empty_input)
        assert output.shape == (0, 128)

    def test_single_sample_batch(self, transformer_encoder):
        """Test with single sample batch."""
        single_input = torch.randn(1, 16, 128)
        output = transformer_encoder.forward(single_input)
        assert output.shape == (1, 128)

    def test_large_batch(self, transformer_encoder):
        """Test with large batch size."""
        large_batch_size = 64
        large_input = torch.randn(large_batch_size, 16, 128)
        output = transformer_encoder.forward(large_input)
        assert output.shape == (large_batch_size, 128)

    def test_numerical_stability(self, transformer_encoder):
        """Test numerical stability with extreme inputs."""
        batch_size, seq_len, d_model = 4, 16, 128

        # Test with very large values
        large_input = torch.full((batch_size, seq_len, d_model), 10.0)
        large_output = transformer_encoder.forward(large_input)
        assert torch.all(torch.isfinite(large_output))

        # Test with very small values
        small_input = torch.full((batch_size, seq_len, d_model), 1e-6)
        small_output = transformer_encoder.forward(small_input)
        assert torch.all(torch.isfinite(small_output))

        # Test with mixed large and small values
        mixed_input = torch.randn(batch_size, seq_len, d_model) * 100
        mixed_output = transformer_encoder.forward(mixed_input)
        assert torch.all(torch.isfinite(mixed_output))

    def test_forward_pass_consistency(self, transformer_encoder, sample_input):
        """Test forward pass consistency with same input."""
        transformer_encoder.eval()

        with torch.no_grad():
            output1 = transformer_encoder.forward(sample_input)
            output2 = transformer_encoder.forward(sample_input)

            # Should be identical
            assert torch.allclose(output1, output2, atol=1e-6)

    @pytest.mark.parametrize("d_model,nhead", [(64, 4), (128, 8), (256, 16)])
    def test_parameterized_dimensions(self, d_model, nhead):
        """Test transformer with various parameterized dimensions."""
        encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=2,
            dim_feedforward=d_model * 2,
        )

        input_tensor = torch.randn(3, 16, d_model)
        output = encoder.forward(input_tensor)
        assert output.shape == (3, d_model)

    def test_invalid_nhead_d_model_combination(self):
        """Test that invalid nhead/d_model combinations raise errors."""
        with pytest.raises(AssertionError):
            # nhead=7 does not divide d_model=64 evenly
            encoder = TransformerEncoder(
                d_model=64,
                nhead=7,  # Invalid: doesn't divide 64
                num_layers=1,
                dim_feedforward=256,
            )
            # Error should occur when creating the TransformerEncoderLayer
            input_tensor = torch.randn(2, 16, 64)
            encoder.forward(input_tensor)

    def test_zero_dropout(self):
        """Test transformer with zero dropout."""
        encoder = TransformerEncoder(
            d_model=64,
            nhead=4,
            num_layers=1,
            dim_feedforward=256,
            dropout=0.0,
        )

        input_tensor = torch.randn(4, 16, 64)

        # Training mode with zero dropout should be deterministic
        encoder.train()
        output1 = encoder.forward(input_tensor)
        output2 = encoder.forward(input_tensor)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_high_dropout(self):
        """Test transformer with high dropout."""
        encoder = TransformerEncoder(
            d_model=64,
            nhead=4,
            num_layers=1,
            dim_feedforward=256,
            dropout=0.9,  # Very high dropout
        )

        input_tensor = torch.randn(4, 16, 64)

        # Training mode with high dropout
        encoder.train()
        output = encoder.forward(input_tensor)

        # Should still produce finite output
        assert torch.all(torch.isfinite(output))
        assert output.shape == (4, 64)

    def test_board_size_compatibility(self):
        """Test that transformer is compatible with BOARD_DIM."""
        # The transformer expects sequence length of BOARD_DIM[0] * BOARD_DIM[1]
        expected_seq_len = BOARD_DIM[0] * BOARD_DIM[1]

        encoder = TransformerEncoder(
            d_model=64, nhead=4, num_layers=1, dim_feedforward=256
        )

        input_tensor = torch.randn(2, expected_seq_len, 64)
        output = encoder.forward(input_tensor)
        assert output.shape == (2, 64)

    def test_positional_encoding_channels_match(self):
        """Test that positional encoding channels match d_model."""
        d_model = 128
        encoder = TransformerEncoder(
            d_model=d_model, nhead=8, num_layers=1, dim_feedforward=512
        )

        # Positional encoding should have org_channels == d_model
        assert encoder.positional_encoding.org_channels == d_model

    def test_reduction_parameter(self):
        """Test the reduction parameter functionality."""
        encoder = TransformerEncoder(
            d_model=128, nhead=8, num_layers=1, dim_feedforward=512
        )

        input_tensor = torch.randn(4, 16, 128)

        # Test mean reduction
        output_mean = encoder.forward(input_tensor, reduction="mean")
        assert output_mean.shape == (4, 128)

        # Test cls reduction
        output_cls = encoder.forward(input_tensor, reduction="cls")
        assert output_cls.shape == (4, 128)

        # Outputs should be different
        assert not torch.allclose(output_mean, output_cls)

        # Test invalid reduction
        with pytest.raises(ValueError, match="reduction must be 'mean' or 'cls'"):
            encoder.forward(input_tensor, reduction="invalid")

    def test_cls_token_exists(self):
        """Test that CLS token is created and has correct shape."""
        encoder = TransformerEncoder(
            d_model=128, nhead=8, num_layers=1, dim_feedforward=512
        )

        # CLS token should exist and have correct shape
        assert hasattr(encoder, "cls_token")
        assert encoder.cls_token.shape == (1, 1, 128)
