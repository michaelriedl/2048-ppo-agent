import jax
import jax.numpy as jnp
import torch
from torch2jax import t2j

from .ppo_agent import PPOAgent

OBS_DIM = 31
BOARD_DIM = 16


class TorchActionFunction:
    """
    Wrapper to convert PyTorch PPO agent to JAX-compatible action function.
    """

    def __init__(self, agent: PPOAgent, device: torch.device = torch.device("cpu")):
        """
        Initialize the wrapper.

        Parameters
        ----------
        agent : PPOAgent
            The trained PPO agent
        device : torch.device
            Device to run inference on
        """
        self.agent = t2j(agent.to(device).eval())
        # Store the agent parameters as a dictionary of JAX arrays
        self._agent_params = {k: t2j(v) for k, v in agent.named_parameters()}
        # Add the named buffers
        self._agent_buffers = {k: t2j(v) for k, v in agent.named_buffers()}
        # Combine parameters and buffers
        self._agent_state = {**self._agent_params, **self._agent_buffers}
        self.device = device

    def __call__(
        self, rng_key: jax.Array, obs: jax.Array, mask: jax.Array
    ) -> jax.Array:
        """
        JAX-compatible action function.

        Parameters
        ----------
        rng_key : jax.Array
            JAX random key (unused, but required for compatibility)
        obs : jax.Array
            Observation with shape (4, 4, 31)
        mask : jax.Array
            Legal action mask with shape (4,)

        Returns
        -------
        jax.Array
            Selected action
        """
        with torch.no_grad():
            # Reshape the observation
            obs = obs.reshape(BOARD_DIM, OBS_DIM)
            # If there is no batch dimension, add one to the obs and mask
            if obs.ndim == 2:
                obs = obs.reshape(1, BOARD_DIM, OBS_DIM)
            if mask.ndim == 1:
                mask = mask.reshape(1, -1)
            # Get action from agent
            action_logits, _ = jax.jit(self.agent)(
                obs, mask, state_dict=self._agent_state
            )
            # Clip action logits
            action_logits = jnp.maximum(
                action_logits, jnp.finfo(action_logits.dtype).min
            )

        return jax.random.categorical(rng_key, logits=action_logits)
