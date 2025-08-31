from typing import Callable

import jax
import numpy as np
import pgx

ENV_ID = "2048"


class BatchRunner:
    """
    BatchRunner class for running batched environments with an action function.
    """

    def __init__(self, init_seed: int, act_fn: Callable = None):
        """
        Initialize the BatchRunner class.

        Parameters
        ----------
        init_seed : int
            Initial random seed.
        act_fn : Callable
            Function to choose actions. Defaults to None.
        """
        self.key = jax.random.key(seed=init_seed)
        self.env = pgx.make(ENV_ID)
        self.init_fn = jax.jit(jax.vmap(self.env.init))
        self.step_fn = jax.jit(jax.vmap(self.env.step))
        if act_fn is not None:
            self._act_fn = jax.jit(jax.vmap(act_fn))
        else:
            self._act_fn = None

    @property
    def act_fn(self):
        """
        Get the action function.
        """
        return self._act_fn

    @act_fn.setter
    def act_fn(self, act_fn: Callable):
        """
        Set the action function.

        Parameters
        ----------
        act_fn : Callable
            Function to choose actions.
        """
        self._act_fn = jax.jit(jax.vmap(act_fn))

    def run_actions_batch(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run actions in the 2048 environment in parallel environments with a
        batch size of batch_size.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        observations : np.ndarray
            Array of observations from the environments. Expected shape (batch_size, num_steps, 16).
        actions : np.ndarray
            Array of actions taken in the environments. Expected shape (batch_size, num_steps).
        rewards : np.ndarray
            Array of rewards received from the environments. Expected shape (batch_size, num_steps).
        terminations : np.ndarray
            Array of termination flags from the environments. Expected shape (batch_size, num_steps).
        """
        # Check if the action function is set
        if self.act_fn is None:
            raise ValueError("The action function is not set.")
        # Initialize the states
        self.key, subkey = jax.random.split(self.key)
        subkey = jax.random.split(subkey, batch_size)
        state = self.init_fn(subkey)

        # Run the environment
        observations = []
        actions = []
        rewards = []
        terminations = []
        while not (state.terminated | state.truncated).all():
            self.key, subkey = jax.random.split(self.key)
            subkey = jax.random.split(subkey, batch_size)
            action = self.act_fn(subkey, state.observation, state.legal_action_mask)
            self.key, subkey = jax.random.split(self.key)
            subkey = jax.random.split(subkey, batch_size)
            state = self.step_fn(state, action, subkey)
            # Store the observations, actions, rewards, and terminations
            observations.append(state.observation)
            actions.append(action)
            rewards.append(state.rewards)
            terminations.append(state.terminated | state.truncated)
        # Convert lists to arrays
        observations = np.stack(observations, axis=1)
        actions = np.stack(actions, axis=1)
        rewards = np.concatenate(rewards, axis=1)
        terminations = np.stack(terminations, axis=1)

        return observations, actions, rewards, terminations
