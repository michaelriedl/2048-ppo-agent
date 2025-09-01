import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import pytest
import torch
import torch.nn as nn

from src.env_definitions import ACTION_DIM, OBS_DIM
from src.ppo.ppo_agent import PPOAgent


class TestPPOAgent:
    """Test class for PPOAgent functionality."""

    @pytest.fixture
    def agent(self):
        """Create a PPOAgent instance for testing."""
        return PPOAgent(
            observation_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=256,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.1,
        )

    @pytest.fixture
    def sample_observations(self):
        """Create sample observations for testing."""
        batch_size = 8
        board_dim = 16  # Flattened 4x4 board
        observation_dim = OBS_DIM
        return torch.randn(batch_size, board_dim, observation_dim)

    @pytest.fixture
    def sample_action_mask(self):
        """Create sample action mask for testing."""
        batch_size = 8
        action_dim = ACTION_DIM
        # Create random binary mask
        mask = torch.randint(0, 2, (batch_size, action_dim)).float()
        # Ensure at least one action is available per batch
        mask[torch.sum(mask, dim=1) == 0, 0] = 1.0
        return mask

    def test_initialization(self, agent):
        """Test that PPOAgent initializes correctly."""
        assert isinstance(agent, nn.Module)
        assert agent.observation_dim == OBS_DIM
        assert agent.action_dim == ACTION_DIM
        assert agent.hidden_dim == 256

        # Check that all components are initialized
        assert hasattr(agent, "input_embedding")
        assert hasattr(agent, "transformer")
        assert hasattr(agent, "actor")
        assert hasattr(agent, "critic")

        # Check component types
        assert isinstance(agent.input_embedding, nn.Linear)
        assert isinstance(agent.actor, nn.Sequential)
        assert isinstance(agent.critic, nn.Sequential)

    def test_initialization_default_params(self):
        """Test PPOAgent initialization with default parameters."""
        agent = PPOAgent()
        assert agent.observation_dim == OBS_DIM  # Default is now OBS_DIM
        assert agent.action_dim == ACTION_DIM
        assert agent.hidden_dim == 512

    def test_initialization_custom_params(self):
        """Test PPOAgent initialization with custom parameters."""
        agent = PPOAgent(
            observation_dim=32,
            action_dim=8,
            hidden_dim=1024,
            d_model=512,
            nhead=16,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.2,
        )
        assert agent.observation_dim == 32
        assert agent.action_dim == 8
        assert agent.hidden_dim == 1024

    def test_forward_pass_shape(self, agent, sample_observations):
        """Test that forward pass returns correct shapes."""
        batch_size = sample_observations.shape[0]
        action_logits, values = agent.forward(sample_observations)

        # Check output shapes
        assert action_logits.shape == (batch_size, agent.action_dim)
        assert values.shape == (batch_size, 1)

        # Check output types
        assert isinstance(action_logits, torch.Tensor)
        assert isinstance(values, torch.Tensor)

    def test_forward_pass_with_action_mask(
        self, agent, sample_observations, sample_action_mask
    ):
        """Test forward pass with action mask applied."""
        action_logits, values = agent.forward(sample_observations, sample_action_mask)

        batch_size = sample_observations.shape[0]
        assert action_logits.shape == (batch_size, agent.action_dim)
        assert values.shape == (batch_size, 1)

        # Check that masked actions have very low logits
        masked_positions = sample_action_mask == 0
        if masked_positions.any():
            masked_logits = action_logits[masked_positions]
            assert torch.all(masked_logits < -1e7)

    def test_forward_pass_without_action_mask(self, agent, sample_observations):
        """Test forward pass without action mask."""
        action_logits, values = agent.forward(sample_observations, action_mask=None)

        batch_size = sample_observations.shape[0]
        assert action_logits.shape == (batch_size, agent.action_dim)
        assert values.shape == (batch_size, 1)

        # Check that logits are reasonable (not extremely negative)
        assert torch.all(action_logits > -1e7)

    def test_get_action_shape(self, agent, sample_observations):
        """Test that get_action returns correct shapes."""
        batch_size = sample_observations.shape[0]
        actions, log_probs, values = agent.get_action(sample_observations)

        # Check output shapes
        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size, 1)

        # Check output types
        assert isinstance(actions, torch.Tensor)
        assert isinstance(log_probs, torch.Tensor)
        assert isinstance(values, torch.Tensor)

    def test_get_action_with_mask(self, agent, sample_observations, sample_action_mask):
        """Test get_action with action mask."""
        actions, log_probs, values = agent.get_action(
            sample_observations, sample_action_mask
        )

        batch_size = sample_observations.shape[0]
        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size, 1)

        # Check that sampled actions are valid (within allowed actions)
        for i in range(batch_size):
            valid_actions = torch.where(sample_action_mask[i] == 1)[0]
            assert actions[i] in valid_actions

    def test_get_action_deterministic_with_seed(self, agent, sample_observations):
        """Test that get_action is deterministic when using same seed."""
        torch.manual_seed(42)
        actions1, log_probs1, values1 = agent.get_action(sample_observations)

        torch.manual_seed(42)
        actions2, log_probs2, values2 = agent.get_action(sample_observations)

        # Should be identical with same seed
        assert torch.allclose(actions1, actions2)
        assert torch.allclose(log_probs1, log_probs2)
        assert torch.allclose(values1, values2)

    def test_evaluate_actions_shape(self, agent, sample_observations):
        """Test that evaluate_actions returns correct shapes."""
        batch_size = sample_observations.shape[0]
        # Create sample actions
        sample_actions = torch.randint(0, agent.action_dim, (batch_size,))

        log_probs, values, entropy = agent.evaluate_actions(
            sample_observations, sample_actions
        )

        # Check output shapes
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size, 1)
        assert entropy.shape == (batch_size,)

        # Check output types
        assert isinstance(log_probs, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)

    def test_evaluate_actions_with_mask(
        self, agent, sample_observations, sample_action_mask
    ):
        """Test evaluate_actions with action mask."""
        batch_size = sample_observations.shape[0]

        # Create actions that respect the mask
        sample_actions = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            valid_actions = torch.where(sample_action_mask[i] == 1)[0]
            sample_actions[i] = valid_actions[
                torch.randint(0, len(valid_actions), (1,))
            ]

        log_probs, values, entropy = agent.evaluate_actions(
            sample_observations, sample_actions, sample_action_mask
        )

        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size, 1)
        assert entropy.shape == (batch_size,)

        # Log probabilities should be finite for valid actions
        assert torch.all(torch.isfinite(log_probs))

    def test_evaluate_actions_consistency(self, agent, sample_observations):
        """Test consistency between get_action and evaluate_actions."""
        agent.eval()  # Set to eval mode to disable dropout

        with torch.no_grad():
            # Get actions using get_action
            actions, log_probs_get, values_get = agent.get_action(sample_observations)

            # Evaluate the same actions using evaluate_actions
            log_probs_eval, values_eval, entropy = agent.evaluate_actions(
                sample_observations, actions
            )

            # Values should be identical
            assert torch.allclose(values_get, values_eval, atol=1e-6)

            # Log probabilities should be identical
            assert torch.allclose(log_probs_get, log_probs_eval, atol=1e-6)

    def test_gradient_flow(self, agent, sample_observations):
        """Test that gradients flow through the network."""
        # Enable gradient computation
        agent.train()

        # Forward pass
        action_logits, values = agent.forward(sample_observations)

        # Compute dummy loss
        loss = action_logits.mean() + values.mean()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in agent.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ), f"Zero gradient for parameter: {name}"

    def test_eval_mode_no_gradients(self, agent, sample_observations):
        """Test that eval mode doesn't compute gradients."""
        agent.eval()

        with torch.no_grad():
            action_logits, values = agent.forward(sample_observations)
            actions, log_probs, values_action = agent.get_action(sample_observations)
            log_probs_eval, values_eval, entropy = agent.evaluate_actions(
                sample_observations, actions
            )

        # All outputs should require no gradients
        assert not action_logits.requires_grad
        assert not values.requires_grad
        assert not actions.requires_grad
        assert not log_probs.requires_grad
        assert not values_action.requires_grad

    def test_different_batch_sizes(self, agent):
        """Test that agent works with different batch sizes."""
        batch_sizes = [1, 4, 16, 32]
        board_dim = 16
        observation_dim = OBS_DIM

        for batch_size in batch_sizes:
            observations = torch.randn(batch_size, board_dim, observation_dim)

            # Test forward pass
            action_logits, values = agent.forward(observations)
            assert action_logits.shape == (batch_size, agent.action_dim)
            assert values.shape == (batch_size, 1)

            # Test get_action
            actions, log_probs, values_action = agent.get_action(observations)
            assert actions.shape == (batch_size,)
            assert log_probs.shape == (batch_size,)
            assert values_action.shape == (batch_size, 1)

    def test_action_validity(self, agent, sample_observations):
        """Test that sampled actions are valid."""
        actions, _, _ = agent.get_action(sample_observations)

        # Actions should be within valid range
        assert torch.all(actions >= 0)
        assert torch.all(actions < agent.action_dim)

        # Actions should be integers
        assert actions.dtype == torch.long

    def test_value_range(self, agent, sample_observations):
        """Test that value estimates are reasonable."""
        _, values = agent.forward(sample_observations)

        # Values should be finite
        assert torch.all(torch.isfinite(values))

        # Values should not be extremely large
        assert torch.all(torch.abs(values) < 1e6)

    def test_entropy_properties(self, agent, sample_observations):
        """Test entropy properties."""
        batch_size = sample_observations.shape[0]
        sample_actions = torch.randint(0, agent.action_dim, (batch_size,))

        _, _, entropy = agent.evaluate_actions(sample_observations, sample_actions)

        # Entropy should be non-negative
        assert torch.all(entropy >= 0)

        # Entropy should be finite
        assert torch.all(torch.isfinite(entropy))

    def test_network_parameters_update(self, agent, sample_observations):
        """Test that network parameters can be updated."""
        # Store initial parameters
        initial_params = {
            name: param.clone() for name, param in agent.named_parameters()
        }

        # Create optimizer
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

        # Forward pass and loss computation
        action_logits, values = agent.forward(sample_observations)
        loss = action_logits.mean() + values.mean()

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters have been updated
        for name, param in agent.named_parameters():
            assert not torch.allclose(
                initial_params[name], param, atol=1e-8
            ), f"Parameter {name} was not updated"

    def test_device_compatibility(self, agent):
        """Test agent works on different devices."""
        # Test CPU
        cpu_observations = torch.randn(4, 16, 31)
        action_logits, values = agent.forward(cpu_observations)
        assert action_logits.device == torch.device("cpu")
        assert values.device == torch.device("cpu")

        # Test GPU if available
        if torch.cuda.is_available():
            agent_gpu = agent.cuda()
            gpu_observations = cpu_observations.cuda()
            action_logits_gpu, values_gpu = agent_gpu.forward(gpu_observations)
            assert action_logits_gpu.device.type == "cuda"
            assert values_gpu.device.type == "cuda"

    def test_state_dict_save_load(self, agent, sample_observations):
        """Test saving and loading agent state."""
        agent.eval()  # Set to eval mode to disable dropout

        with torch.no_grad():
            # Get initial outputs
            initial_logits, initial_values = agent.forward(sample_observations)

            # Save state dict
            state_dict = agent.state_dict()

            # Create new agent and load state
            new_agent = PPOAgent(
                observation_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                hidden_dim=256,
                d_model=128,
                nhead=4,
                num_layers=2,
                dim_feedforward=512,
                dropout=0.1,
            )
            new_agent.load_state_dict(state_dict)
            new_agent.eval()  # Set to eval mode

            # Check outputs are identical
            new_logits, new_values = new_agent.forward(sample_observations)
            assert torch.allclose(initial_logits, new_logits, atol=1e-6)
            assert torch.allclose(initial_values, new_values, atol=1e-6)

    def test_action_mask_edge_cases(self, agent):
        """Test action mask edge cases."""
        batch_size = 4
        board_dim = 16
        observation_dim = OBS_DIM
        observations = torch.randn(batch_size, board_dim, observation_dim)

        # Test with all actions allowed
        all_allowed_mask = torch.ones(batch_size, agent.action_dim)
        actions, log_probs, values = agent.get_action(observations, all_allowed_mask)
        assert torch.all(torch.isfinite(log_probs))

        # Test with single action allowed per batch
        single_action_mask = torch.zeros(batch_size, agent.action_dim)
        for i in range(batch_size):
            single_action_mask[i, i % agent.action_dim] = 1.0

        actions, log_probs, values = agent.get_action(observations, single_action_mask)

        # Check that only allowed actions were sampled
        for i in range(batch_size):
            expected_action = i % agent.action_dim
            assert actions[i] == expected_action

    def test_observation_dimension_mismatch(self, agent):
        """Test behavior with incorrect observation dimensions."""
        # Wrong number of features
        wrong_obs = torch.randn(4, 16, 15)  # Should be 31 features

        with pytest.raises(RuntimeError):
            agent.forward(wrong_obs)

    def test_empty_batch(self, agent):
        """Test behavior with empty batch."""
        empty_obs = torch.randn(0, 16, 31)
        action_logits, values = agent.forward(empty_obs)

        assert action_logits.shape == (0, agent.action_dim)
        assert values.shape == (0, 1)

    def test_single_observation(self, agent):
        """Test with single observation (batch size 1)."""
        single_obs = torch.randn(1, 16, 31)

        # Test forward
        action_logits, values = agent.forward(single_obs)
        assert action_logits.shape == (1, agent.action_dim)
        assert values.shape == (1, 1)

        # Test get_action
        actions, log_probs, values_action = agent.get_action(single_obs)
        assert actions.shape == (1,)
        assert log_probs.shape == (1,)
        assert values_action.shape == (1, 1)

    def test_large_batch(self, agent):
        """Test with large batch size."""
        large_batch_size = 128
        large_obs = torch.randn(large_batch_size, 16, 31)

        action_logits, values = agent.forward(large_obs)
        assert action_logits.shape == (large_batch_size, agent.action_dim)
        assert values.shape == (large_batch_size, 1)

    def test_log_prob_properties(self, agent, sample_observations):
        """Test properties of log probabilities."""
        actions, log_probs, _ = agent.get_action(sample_observations)

        # Log probabilities should be negative or zero
        assert torch.all(log_probs <= 0)

        # Log probabilities should be finite
        assert torch.all(torch.isfinite(log_probs))

    def test_network_output_consistency(self, agent, sample_observations):
        """Test that network outputs are consistent across multiple calls."""
        agent.eval()  # Set to eval mode to disable dropout

        with torch.no_grad():
            # Multiple forward passes should give identical results
            logits1, values1 = agent.forward(sample_observations)
            logits2, values2 = agent.forward(sample_observations)

            assert torch.allclose(logits1, logits2, atol=1e-6)
            assert torch.allclose(values1, values2, atol=1e-6)

    def test_dropout_training_vs_eval(self, agent, sample_observations):
        """Test that dropout behaves differently in training vs eval mode."""
        # Training mode - multiple passes may give different results
        agent.train()
        logits_train1, values_train1 = agent.forward(sample_observations)
        logits_train2, values_train2 = agent.forward(sample_observations)

        # Eval mode - should give consistent results
        agent.eval()
        with torch.no_grad():
            logits_eval1, values_eval1 = agent.forward(sample_observations)
            logits_eval2, values_eval2 = agent.forward(sample_observations)

            assert torch.allclose(logits_eval1, logits_eval2, atol=1e-6)
            assert torch.allclose(values_eval1, values_eval2, atol=1e-6)

    def test_actor_critic_independence(self, agent, sample_observations):
        """Test that actor and critic heads produce independent outputs."""
        action_logits, values = agent.forward(sample_observations)

        # Both should have valid outputs
        assert torch.all(torch.isfinite(action_logits))
        assert torch.all(torch.isfinite(values))

        # They should have different shapes (confirming they're separate heads)
        assert action_logits.shape[-1] == agent.action_dim
        assert values.shape[-1] == 1

    def test_transformer_feature_extraction(self, agent, sample_observations):
        """Test that transformer properly processes sequential features."""
        batch_size = sample_observations.shape[0]

        # Forward pass through embedding and transformer
        embedded = agent.input_embedding(sample_observations)
        transformer_out = agent.transformer(embedded)
        features = transformer_out.mean(dim=1)

        # Check intermediate shapes
        assert embedded.shape == (batch_size, 16, 128)  # d_model = 128
        assert transformer_out.shape == (batch_size, 16, 128)
        assert features.shape == (batch_size, 128)

    @pytest.mark.parametrize("observation_dim,action_dim", [(16, 2), (31, 4), (64, 8)])
    def test_different_dimensions(self, observation_dim, action_dim):
        """Test agent with different observation and action dimensions."""
        agent = PPOAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            d_model=64,  # Smaller for faster testing
            nhead=2,
            num_layers=1,
        )

        batch_size = 4
        board_dim = 16
        observations = torch.randn(batch_size, board_dim, observation_dim)

        action_logits, values = agent.forward(observations)
        assert action_logits.shape == (batch_size, action_dim)
        assert values.shape == (batch_size, 1)

    def test_numerical_stability(self, agent):
        """Test numerical stability with extreme inputs."""
        batch_size = 4
        board_dim = 16
        observation_dim = OBS_DIM

        # Test with very large values
        large_obs = torch.full((batch_size, board_dim, observation_dim), 1e3)
        action_logits, values = agent.forward(large_obs)
        assert torch.all(torch.isfinite(action_logits))
        assert torch.all(torch.isfinite(values))

        # Test with very small values
        small_obs = torch.full((batch_size, board_dim, observation_dim), 1e-6)
        action_logits, values = agent.forward(small_obs)
        assert torch.all(torch.isfinite(action_logits))
        assert torch.all(torch.isfinite(values))

        # Test with zeros
        zero_obs = torch.zeros(batch_size, board_dim, observation_dim)
        action_logits, values = agent.forward(zero_obs)
        assert torch.all(torch.isfinite(action_logits))
        assert torch.all(torch.isfinite(values))
