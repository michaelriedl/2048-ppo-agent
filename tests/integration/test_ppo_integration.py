"""
Integration tests for PPO agent, torch action wrapper, and batch runner.

This module tests the integration between the key components of the PPO training pipeline:
- PPOAgent: The neural network policy and value function
- TorchActionFunction: Wrapper that converts PyTorch models to JAX-compatible functions
- BatchRunner: Environment runner that executes batched 2048 games

These tests verify that the components work together correctly, similar to how they
are integrated in the PPO trainer during actual training.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from src.ppo.ppo_agent import PPOAgent
from src.ppo.torch_action_wrapper import TorchActionFunction
from src.runs.batch_runner import BatchRunner


class TestPPOIntegration:
    """Integration tests for PPO agent, torch action wrapper, and batch runner."""

    @pytest.fixture
    def agent(self):
        """Create a PPOAgent instance for testing."""
        return PPOAgent(
            observation_dim=31,
            action_dim=4,
            hidden_dim=128,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
        )

    @pytest.fixture
    def batch_runner(self):
        """Create a BatchRunner instance for testing."""
        return BatchRunner(init_seed=42)

    @pytest.fixture
    def torch_action_wrapper(self, agent):
        """Create a TorchActionFunction wrapper for testing."""
        return TorchActionFunction(agent, use_mask=True, device=torch.device("cpu"))

    def test_torch_action_wrapper_creation(self, agent):
        """Test that TorchActionWrapper can be created with a PPO agent."""
        wrapper = TorchActionFunction(agent, device=torch.device("cpu"))

        assert wrapper is not None
        assert wrapper.device == torch.device("cpu")
        assert wrapper._agent_state is not None
        assert isinstance(wrapper._agent_state, dict)

    def test_torch_action_wrapper_call(self, torch_action_wrapper):
        """Test that the torch action wrapper can be called with JAX inputs."""
        # Create sample inputs
        rng_key = jax.random.key(0)
        obs = jax.random.uniform(rng_key, (4, 4, 31))  # Board shape with features
        mask = jax.numpy.ones((4,))  # All actions are legal

        # Call the wrapper
        action, log_prob, value = torch_action_wrapper(rng_key, obs, mask)

        # Verify output
        assert isinstance(action, jax.Array)
        assert isinstance(log_prob, jax.Array)
        assert isinstance(value, jax.Array)
        assert action.shape == ()  # Scalar action
        assert log_prob.shape == ()  # Scalar log probability
        assert value.shape == ()  # Scalar value
        assert 0 <= action < 4  # Valid action range
        assert jnp.isfinite(log_prob)  # Log prob should be finite
        assert jnp.isfinite(value)  # Value should be finite

    def test_torch_action_wrapper_with_mask(self, torch_action_wrapper):
        """Test that the torch action wrapper respects action masks."""
        rng_key = jax.random.key(42)
        obs = jax.random.uniform(rng_key, (4, 4, 31))

        # Test with only one legal action
        mask = jax.numpy.array([1.0, 0.0, 0.0, 0.0])  # Only action 0 is legal

        # Run multiple times to check consistency
        actions = []
        for i in range(10):
            key = jax.random.key(i)
            action, log_prob, value = torch_action_wrapper(key, obs, mask)
            actions.append(int(action))

        # All actions should be 0 (the only legal action)
        assert all(action == 0 for action in actions)

    def test_batch_runner_with_torch_action_wrapper(
        self, batch_runner, torch_action_wrapper
    ):
        """Test integration between BatchRunner and TorchActionWrapper."""
        # Set the action function in batch runner
        batch_runner.act_fn = torch_action_wrapper

        # Run a small batch
        batch_size = 2
        (
            observations,
            actions,
            action_masks,
            log_probs,
            values,
            rewards,
            terminations,
        ) = batch_runner.run_actions_batch(batch_size)

        # Verify outputs have correct shapes and types
        assert isinstance(observations, np.ndarray)
        assert isinstance(actions, np.ndarray)
        assert isinstance(action_masks, np.ndarray)
        assert isinstance(log_probs, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(terminations, np.ndarray)

        # Check shapes
        assert observations.shape[0] == batch_size  # Batch dimension
        assert actions.shape[0] == batch_size
        assert action_masks.shape[0] == batch_size
        assert log_probs.shape[0] == batch_size
        assert values.shape[0] == batch_size
        assert rewards.shape[0] == batch_size
        assert terminations.shape[0] == batch_size

        # Check that actions are valid
        assert np.all((actions >= 0) & (actions < 4))

        # Check that observations have the right dimensions (batch, steps, 4, 4, 31)
        assert observations.shape[2] == 4  # Board height
        assert observations.shape[3] == 4  # Board width
        assert observations.shape[4] == 31  # Feature dimension

        # Check that all episodes terminated
        assert np.all(terminations[:, -1])

    def test_ppo_agent_forward_compatibility(self, agent):
        """Test that PPO agent forward pass is compatible with the wrapper."""
        # Create sample batch
        batch_size = 4
        board_size = 16
        obs_dim = 31

        observations = torch.randn(batch_size, board_size, obs_dim)
        action_mask = torch.ones(batch_size, 4)  # All actions legal

        # Forward pass
        action_logits, values = agent.forward(observations, action_mask)

        # Verify outputs
        assert action_logits.shape == (batch_size, 4)
        assert values.shape == (batch_size, 1)

        # Test get_action method
        actions, log_probs, values_get = agent.get_action(observations, action_mask)

        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert values_get.shape == (batch_size, 1)
        assert torch.all((actions >= 0) & (actions < 4))

    def test_full_integration_workflow(self, agent, batch_runner):
        """Test the complete integration workflow similar to PPO trainer."""
        # Create torch action wrapper
        torch_action_fn = TorchActionFunction(
            agent, use_mask=True, device=torch.device("cpu")
        )

        # Set action function in batch runner
        batch_runner.act_fn = torch_action_fn

        # Run environments
        batch_size = 3
        (
            observations,
            actions,
            action_masks,
            log_probs,
            values,
            rewards,
            terminations,
        ) = batch_runner.run_actions_batch(batch_size)

        # Verify the complete pipeline worked
        assert observations.shape[0] == batch_size
        assert actions.shape[0] == batch_size
        assert log_probs.shape[0] == batch_size
        assert values.shape[0] == batch_size
        assert rewards.shape[0] == batch_size
        assert terminations.shape[0] == batch_size

        # Check that we got meaningful data
        assert observations.shape[1] > 0  # At least one step
        assert np.any(rewards > 0)  # At least some positive rewards
        assert np.all(terminations[:, -1])  # All episodes ended

        # Check observation values are reasonable (not all zeros)
        assert not np.allclose(observations, 0)

        # Test that we can get policy outputs for the collected observations
        with torch.no_grad():
            # Convert first step observations to torch
            obs_tensor = torch.from_numpy(observations[:, 0, :, :, :]).float()
            obs_tensor = obs_tensor.reshape(
                batch_size, 16, 31
            )  # Reshape to (batch, board_size, features)

            # Get action logits and values
            action_logits, values = agent.forward(obs_tensor)

            assert action_logits.shape == (batch_size, 4)
            assert values.shape == (batch_size, 1)

    def test_torch_action_wrapper_jax_compatibility(self, torch_action_wrapper):
        """Test that the wrapper is fully JAX-compatible for JIT compilation."""
        rng_key = jax.random.key(123)
        obs = jax.random.uniform(rng_key, (4, 4, 31))
        mask = jax.numpy.ones((4,))

        # Test that the function can be JIT compiled
        jitted_action_fn = jax.jit(torch_action_wrapper)
        action, log_prob, value = jitted_action_fn(rng_key, obs, mask)

        assert isinstance(action, jax.Array)
        assert isinstance(log_prob, jax.Array)
        assert isinstance(value, jax.Array)
        assert action.shape == ()
        assert log_prob.shape == ()
        assert value.shape == ()
        assert 0 <= action < 4
        assert jnp.isfinite(log_prob)
        assert jnp.isfinite(value)

    def test_batch_runner_determinism(self, batch_runner, agent):
        """Test that batch runner produces deterministic results with same seed."""
        # Create two identical wrappers
        wrapper1 = TorchActionFunction(agent, device=torch.device("cpu"))
        wrapper2 = TorchActionFunction(agent, device=torch.device("cpu"))

        # Run with same initial conditions
        batch_runner.act_fn = wrapper1
        obs1, actions1, action_masks1, log_probs1, values1, rewards1, term1 = (
            batch_runner.run_actions_batch(2)
        )

        # Reset batch runner with same seed
        batch_runner.__init__(init_seed=42)
        batch_runner.act_fn = wrapper2
        obs2, actions2, action_masks2, log_probs2, values2, rewards2, term2 = (
            batch_runner.run_actions_batch(2)
        )

        # Results should be identical (within numerical precision)
        np.testing.assert_array_equal(actions1, actions2)
        np.testing.assert_array_equal(action_masks1, action_masks2)
        np.testing.assert_array_equal(term1, term2)
        np.testing.assert_allclose(obs1, obs2, rtol=1e-6)
        np.testing.assert_allclose(log_probs1, log_probs2, rtol=1e-6)
        np.testing.assert_allclose(values1, values2, rtol=1e-6)
        np.testing.assert_allclose(rewards1, rewards2, rtol=1e-6)
        np.testing.assert_allclose(rewards1, rewards2, rtol=1e-6)

    def test_integration_with_different_batch_sizes(self, agent, batch_runner):
        """Test integration with various batch sizes."""
        torch_action_fn = TorchActionFunction(agent, device=torch.device("cpu"))
        batch_runner.act_fn = torch_action_fn

        batch_sizes = [1, 3, 8]

        for batch_size in batch_sizes:
            # Reset batch runner for each test
            batch_runner.__init__(init_seed=42)
            batch_runner.act_fn = torch_action_fn

            (
                observations,
                actions,
                action_masks,
                log_probs,
                values,
                rewards,
                terminations,
            ) = batch_runner.run_actions_batch(batch_size)

            # Verify correct batch size
            assert observations.shape[0] == batch_size
            assert actions.shape[0] == batch_size
            assert log_probs.shape[0] == batch_size
            assert values.shape[0] == batch_size
            assert rewards.shape[0] == batch_size
            assert terminations.shape[0] == batch_size

            # Verify all episodes completed
            assert np.all(terminations[:, -1])

    def test_action_wrapper_gradient_isolation(self, agent):
        """Test that the action wrapper doesn't interfere with gradients."""
        # Enable gradients on the original agent
        agent.train()

        # Create wrapper (this will call .eval() on the agent)
        wrapper = TorchActionFunction(agent, device=torch.device("cpu"))

        # Set agent back to training mode after wrapper creation
        agent.train()

        # Original agent should now have gradients enabled again
        assert agent.training
        for param in agent.parameters():
            assert param.requires_grad

        # Test that we can still compute gradients on the original agent
        obs = torch.randn(1, 16, 31)
        mask = torch.ones(1, 4)

        action_logits, values = agent.forward(obs, mask)
        loss = action_logits.sum() + values.sum()
        loss.backward()

        # Check that gradients were computed
        for param in agent.parameters():
            assert param.grad is not None

    def test_action_selection_distribution(self, agent, batch_runner):
        """Test that action selection follows a reasonable distribution."""
        torch_action_fn = TorchActionFunction(agent, device=torch.device("cpu"))
        batch_runner.act_fn = torch_action_fn

        # Run multiple small batches to collect action statistics
        all_actions = []
        for _ in range(5):
            batch_runner.__init__(init_seed=np.random.randint(0, 1000))
            batch_runner.act_fn = torch_action_fn

            _, actions, _, _, _, _, _ = batch_runner.run_actions_batch(4)
            all_actions.extend(actions.flatten())

        # Convert to numpy array
        all_actions = np.array(all_actions)

        # Check that all actions are valid
        assert np.all((all_actions >= 0) & (all_actions < 4))

        # Check that we see some variety in actions (not all the same)
        unique_actions = np.unique(all_actions)
        assert len(unique_actions) > 1, "Actions should show some variety"

    @pytest.mark.parametrize("device_str", ["cpu"])
    def test_device_compatibility(self, agent, batch_runner, device_str):
        """Test integration across different devices."""
        device = torch.device(device_str)

        # Move agent to device
        agent = agent.to(device)

        # Create wrapper with specific device
        torch_action_fn = TorchActionFunction(agent, device=device)
        batch_runner.act_fn = torch_action_fn

        # Run integration test
        (
            observations,
            actions,
            action_masks,
            log_probs,
            values,
            rewards,
            terminations,
        ) = batch_runner.run_actions_batch(2)

        # Verify successful execution
        assert observations.shape[0] == 2
        assert np.all(terminations[:, -1])

    def test_error_handling_invalid_inputs(self, batch_runner):
        """Test error handling with invalid inputs."""
        # Test with no action function set
        with pytest.raises(ValueError, match="The action function is not set"):
            batch_runner.run_actions_batch(2)

        # Test with invalid batch size
        agent = PPOAgent()
        torch_action_fn = TorchActionFunction(agent)
        batch_runner.act_fn = torch_action_fn

        with pytest.raises((ValueError, RuntimeError)):
            batch_runner.run_actions_batch(0)  # Invalid batch size

    def test_sample_actions_parameter_integration(self, agent, batch_runner):
        """Test integration of the sample_actions parameter in TorchActionFunction."""
        # Create two wrappers: one with sampling, one with argmax
        wrapper_sampling = TorchActionFunction(
            agent, use_mask=True, sample_actions=True, device=torch.device("cpu")
        )
        wrapper_argmax = TorchActionFunction(
            agent, use_mask=True, sample_actions=False, device=torch.device("cpu")
        )

        # Test with identical inputs
        rng_key = jax.random.key(42)
        obs = jax.random.uniform(rng_key, (4, 4, 31))
        mask = jax.numpy.ones((4,))

        # Get actions from both wrappers multiple times
        sampling_actions = []
        argmax_actions = []

        for i in range(10):
            key = jax.random.key(i)

            # Sampling wrapper
            action_sample, log_prob_sample, value_sample = wrapper_sampling(
                key, obs, mask
            )
            sampling_actions.append(int(action_sample))

            # Argmax wrapper
            action_argmax, log_prob_argmax, value_argmax = wrapper_argmax(
                key, obs, mask
            )
            argmax_actions.append(int(action_argmax))

        # Argmax wrapper should always return the same action (deterministic)
        assert len(set(argmax_actions)) == 1, "Argmax wrapper should be deterministic"

        # Sampling wrapper might show variety (though not guaranteed with small sample)
        # At minimum, both should return valid actions
        assert all(0 <= action < 4 for action in sampling_actions)
        assert all(0 <= action < 4 for action in argmax_actions)

    def test_sample_actions_with_batch_runner_integration(self, agent, batch_runner):
        """Test sample_actions parameter integration with BatchRunner."""
        # Test with sampling enabled
        wrapper_sampling = TorchActionFunction(
            agent, use_mask=True, sample_actions=True, device=torch.device("cpu")
        )
        batch_runner.act_fn = wrapper_sampling

        obs1, actions1, masks1, log_probs1, values1, rewards1, term1 = (
            batch_runner.run_actions_batch(3)
        )

        # Reset and test with argmax
        batch_runner.__init__(init_seed=42)  # Same seed for comparison
        wrapper_argmax = TorchActionFunction(
            agent, use_mask=True, sample_actions=False, device=torch.device("cpu")
        )
        batch_runner.act_fn = wrapper_argmax

        obs2, actions2, masks2, log_probs2, values2, rewards2, term2 = (
            batch_runner.run_actions_batch(3)
        )

        # Both should produce valid results with correct batch size
        assert obs1.shape[0] == 3  # Batch size
        assert obs2.shape[0] == 3  # Batch size
        assert actions1.shape[0] == 3
        assert actions2.shape[0] == 3
        assert masks1.shape[0] == 3
        assert masks2.shape[0] == 3
        assert log_probs1.shape[0] == 3
        assert log_probs2.shape[0] == 3
        assert values1.shape[0] == 3
        assert values2.shape[0] == 3
        assert rewards1.shape[0] == 3
        assert rewards2.shape[0] == 3
        assert term1.shape[0] == 3
        assert term2.shape[0] == 3

        # Both should have proper observation shape (batch, steps, 4, 4, 31)
        assert obs1.shape[2:] == (4, 4, 31)
        assert obs2.shape[2:] == (4, 4, 31)

        # All actions should be valid
        assert np.all((actions1 >= 0) & (actions1 < 4))
        assert np.all((actions2 >= 0) & (actions2 < 4))

        # All episodes should terminate
        assert np.all(term1[:, -1])
        assert np.all(term2[:, -1])

        # Values should be finite
        assert np.all(np.isfinite(values1))
        assert np.all(np.isfinite(values2))

        # Log probabilities should be finite
        assert np.all(np.isfinite(log_probs1))
        assert np.all(np.isfinite(log_probs2))

        # Both should have at least one step per episode
        assert obs1.shape[1] > 0
        assert obs2.shape[1] > 0

    def test_sample_actions_determinism_with_masks(self, agent):
        """Test that sample_actions parameter respects action masks correctly."""
        # Create wrappers
        wrapper_sampling = TorchActionFunction(
            agent, use_mask=True, sample_actions=True, device=torch.device("cpu")
        )
        wrapper_argmax = TorchActionFunction(
            agent, use_mask=True, sample_actions=False, device=torch.device("cpu")
        )

        rng_key = jax.random.key(123)
        obs = jax.random.uniform(rng_key, (4, 4, 31))

        # Test with restricted action space (only action 2 is legal)
        mask = jax.numpy.array([0.0, 0.0, 1.0, 0.0])

        # Test sampling wrapper multiple times
        sampling_actions = []
        for i in range(10):
            key = jax.random.key(i)
            action, log_prob, value = wrapper_sampling(key, obs, mask)
            sampling_actions.append(int(action))

            # All outputs should be valid
            assert isinstance(action, jax.Array)
            assert isinstance(log_prob, jax.Array)
            assert isinstance(value, jax.Array)
            assert jnp.isfinite(log_prob)
            assert jnp.isfinite(value)

        # Test argmax wrapper multiple times
        argmax_actions = []
        for i in range(10):
            key = jax.random.key(i)
            action, log_prob, value = wrapper_argmax(key, obs, mask)
            argmax_actions.append(int(action))

            # All outputs should be valid
            assert isinstance(action, jax.Array)
            assert isinstance(log_prob, jax.Array)
            assert isinstance(value, jax.Array)
            assert jnp.isfinite(log_prob)
            assert jnp.isfinite(value)

        # Both wrappers should only select the legal action (action 2)
        assert all(action == 2 for action in sampling_actions)
        assert all(action == 2 for action in argmax_actions)

    def test_sample_actions_jit_compatibility(self, agent):
        """Test that both sample_actions modes are JIT compatible."""
        # Create wrappers
        wrapper_sampling = TorchActionFunction(
            agent, use_mask=True, sample_actions=True, device=torch.device("cpu")
        )
        wrapper_argmax = TorchActionFunction(
            agent, use_mask=True, sample_actions=False, device=torch.device("cpu")
        )

        # JIT compile both
        jitted_sampling = jax.jit(wrapper_sampling)
        jitted_argmax = jax.jit(wrapper_argmax)

        # Test inputs
        rng_key = jax.random.key(456)
        obs = jax.random.uniform(rng_key, (4, 4, 31))
        mask = jax.numpy.ones((4,))

        # Test jitted sampling wrapper
        action1, log_prob1, value1 = jitted_sampling(rng_key, obs, mask)
        assert isinstance(action1, jax.Array)
        assert isinstance(log_prob1, jax.Array)
        assert isinstance(value1, jax.Array)
        assert action1.shape == ()
        assert log_prob1.shape == ()
        assert value1.shape == ()
        assert 0 <= action1 < 4
        assert jnp.isfinite(log_prob1)
        assert jnp.isfinite(value1)

        # Test jitted argmax wrapper
        action2, log_prob2, value2 = jitted_argmax(rng_key, obs, mask)
        assert isinstance(action2, jax.Array)
        assert isinstance(log_prob2, jax.Array)
        assert isinstance(value2, jax.Array)
        assert action2.shape == ()
        assert log_prob2.shape == ()
        assert value2.shape == ()
        assert 0 <= action2 < 4
        assert jnp.isfinite(log_prob2)
        assert jnp.isfinite(value2)
