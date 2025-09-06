import jax
import jax.numpy as jnp
import torch
from torch2jax import t2j

from ..env_definitions import BOARD_FLAT_DIM, OBS_DIM
from .ppo_agent import PPOAgent


class TorchActionFunction:
    """
    Wrapper to convert PyTorch PPO agent to JAX-compatible action function.
    """

    def __init__(
        self,
        agent: PPOAgent,
        use_mask: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the wrapper.

        Parameters
        ----------
        agent : PPOAgent
            The trained PPO agent
        use_mask : bool
            Whether to use action masking
        device : torch.device
            Device to run inference on
        """
        self.agent = t2j(agent.to(device).eval())
        self.use_mask = use_mask
        # Store the agent parameters as a dictionary of JAX arrays
        self._agent_params = {k: t2j(v) for k, v in agent.named_parameters()}
        # Add the named buffers
        self._agent_buffers = {k: t2j(v) for k, v in agent.named_buffers()}
        # Combine parameters and buffers
        self._agent_state = {**self._agent_params, **self._agent_buffers}
        self.device = device

    def __call__(
        self, rng_key: jax.Array, obs: jax.Array, mask: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        JAX-compatible action function.

        Parameters
        ----------
        rng_key : jax.Array
            JAX random key for action sampling
        obs : jax.Array
            Observation with shape (4, 4, 31)
        mask : jax.Array
            Legal action mask with shape (4,)

        Returns
        -------
        action : jax.Array
            Selected action
        log_prob : jax.Array
            Log probability of the selected action
        value : jax.Array
            Value estimate from the critic
        """
        with torch.no_grad():
            # Reshape the observation
            obs = obs.reshape(BOARD_FLAT_DIM, OBS_DIM)
            # If there is no batch dimension, add one to the obs and mask
            if obs.ndim == 2:
                obs = obs.reshape(1, BOARD_FLAT_DIM, OBS_DIM)
            if mask.ndim == 1:
                mask = mask.reshape(1, -1)
            # Get action from agent
            action_logits, values = jax.jit(self.agent)(
                obs, mask if self.use_mask else None, state_dict=self._agent_state
            )
            # Clip action logits
            action_logits = jnp.maximum(
                action_logits, jnp.finfo(action_logits.dtype).min
            )
            action_logits = action_logits.squeeze()

            # Sample action and compute log probability
            action = jax.random.categorical(rng_key, logits=action_logits)
            log_prob = action_logits[action] - jax.scipy.special.logsumexp(
                action_logits
            )

            # Extract scalar value
            value = values.squeeze()

        return action, log_prob, value
