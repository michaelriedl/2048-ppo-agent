import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from src.env_definitions import ACTION_DIM, BOARD_FLAT_DIM, OBS_DIM
from src.ppo.ppo_agent import PPOAgent
from src.ppo.torch_action_wrapper import TorchActionFunction


class TestLogProbConsistency:
    """Test consistency between TorchActionFunction and PPOAgent log probabilities."""

    @pytest.fixture
    def agent(self):
        """Create a PPO agent for testing."""
        return PPOAgent(
            observation_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=128,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
        )

    @pytest.fixture
    def torch_action_function(self, agent):
        """Create TorchActionFunction wrapper."""
        return TorchActionFunction(agent, device=torch.device("cpu"))

    @pytest.fixture
    def sample_observation(self):
        """Create a sample observation."""
        # Random observation with proper shape
        obs = np.random.randn(4, 4, OBS_DIM).astype(np.float32)
        return obs

    @pytest.fixture
    def sample_mask(self):
        """Create a sample action mask."""
        # Random binary mask for legal actions
        mask = np.random.choice([0, 1], size=(ACTION_DIM,)).astype(np.float32)
        # Ensure at least one action is legal
        if mask.sum() == 0:
            mask[0] = 1.0
        return mask

    def test_log_prob_consistency_single_observation(
        self, agent, torch_action_function, sample_observation, sample_mask
    ):
        """Test log probability consistency for a single observation."""
        # Set agent to eval mode and disable gradients
        agent.eval()

        # Convert to appropriate formats
        obs_torch = torch.from_numpy(sample_observation).reshape(
            1, BOARD_FLAT_DIM, OBS_DIM
        )
        mask_torch = torch.from_numpy(sample_mask).reshape(1, -1)

        obs_jax = jnp.array(sample_observation)
        mask_jax = jnp.array(sample_mask)

        # Get action and log_prob from TorchActionFunction
        rng_key = jax.random.PRNGKey(42)
        action_jax, log_prob_jax, _ = torch_action_function(rng_key, obs_jax, mask_jax)

        # Convert action to torch tensor for evaluation
        action_torch = torch.tensor([int(action_jax)], dtype=torch.long)

        # Get log_prob from PPOAgent.evaluate_actions
        with torch.no_grad():
            log_prob_torch, _, _ = agent.evaluate_actions(
                obs_torch, action_torch, mask_torch
            )

        # Compare log probabilities (convert to same type for comparison)
        log_prob_torch_val = float(log_prob_torch.squeeze())
        log_prob_jax_val = float(log_prob_jax)

        # Assert they are approximately equal (allowing for small numerical differences)
        assert abs(log_prob_torch_val - log_prob_jax_val) < 1e-5, (
            f"Log probabilities don't match: "
            f"torch={log_prob_torch_val}, jax={log_prob_jax_val}"
        )

    def test_log_prob_consistency_multiple_actions(
        self, agent, torch_action_function, sample_observation, sample_mask
    ):
        """Test log probability consistency for multiple specific actions."""
        agent.eval()

        # Convert to appropriate formats
        obs_torch = torch.from_numpy(sample_observation).reshape(
            1, BOARD_FLAT_DIM, OBS_DIM
        )
        mask_torch = torch.from_numpy(sample_mask).reshape(1, -1)

        obs_jax = jnp.array(sample_observation)
        mask_jax = jnp.array(sample_mask)

        # Test all legal actions
        legal_actions = np.where(sample_mask > 0)[0]

        for action_idx in legal_actions:
            # Get action logits from both methods
            with torch.no_grad():
                action_logits_torch, _ = agent.forward(obs_torch, mask_torch)

            # Calculate log_prob using torch agent
            action_torch = torch.tensor([action_idx], dtype=torch.long)
            with torch.no_grad():
                log_prob_torch, _, _ = agent.evaluate_actions(
                    obs_torch, action_torch, mask_torch
                )

            # For JAX version, we need to manually calculate log_prob from logits
            # since TorchActionFunction samples randomly
            action_logits_jax, _ = torch_action_function.agent(
                obs_jax.reshape(1, BOARD_FLAT_DIM, OBS_DIM),
                mask_jax.reshape(1, -1),
                state_dict=torch_action_function._agent_state,
            )
            action_logits_jax = action_logits_jax.squeeze()

            # Calculate log probability manually
            log_prob_jax_manual = action_logits_jax[
                action_idx
            ] - jax.scipy.special.logsumexp(action_logits_jax)

            # Compare log probabilities
            log_prob_torch_val = float(log_prob_torch.squeeze())
            log_prob_jax_val = float(log_prob_jax_manual)

            assert abs(log_prob_torch_val - log_prob_jax_val) < 1e-5, (
                f"Log probabilities don't match for action {action_idx}: "
                f"torch={log_prob_torch_val}, jax={log_prob_jax_val}"
            )

    def test_action_logits_consistency(
        self, agent, torch_action_function, sample_observation, sample_mask
    ):
        """Test that action logits are consistent between both implementations."""
        agent.eval()

        # Convert to appropriate formats
        obs_torch = torch.from_numpy(sample_observation).reshape(
            1, BOARD_FLAT_DIM, OBS_DIM
        )
        mask_torch = torch.from_numpy(sample_mask).reshape(1, -1)

        obs_jax = jnp.array(sample_observation)
        mask_jax = jnp.array(sample_mask)

        # Get action logits from torch agent
        with torch.no_grad():
            action_logits_torch, _ = agent.forward(obs_torch, mask_torch)
            action_logits_torch = action_logits_torch.squeeze().numpy()

        # Get action logits from jax wrapper
        action_logits_jax, _ = torch_action_function.agent(
            obs_jax.reshape(1, BOARD_FLAT_DIM, OBS_DIM),
            mask_jax.reshape(1, -1),
            state_dict=torch_action_function._agent_state,
        )
        action_logits_jax = np.array(action_logits_jax.squeeze())

        # Compare action logits
        np.testing.assert_allclose(
            action_logits_torch,
            action_logits_jax,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Action logits don't match between torch and jax implementations",
        )


if __name__ == "__main__":
    # Run a quick test
    test_instance = TestLogProbConsistency()
    agent = test_instance.agent()
    torch_action_function = test_instance.torch_action_function(agent)
    sample_obs = test_instance.sample_observation()
    sample_mask = test_instance.sample_mask()

    print("Running log probability consistency test...")
    try:
        test_instance.test_log_prob_consistency_single_observation(
            agent, torch_action_function, sample_obs, sample_mask
        )
        test_instance.test_action_logits_consistency(
            agent, torch_action_function, sample_obs, sample_mask
        )
        print("✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
