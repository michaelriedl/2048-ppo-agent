from typing import Callable

import jax
import numpy as np
import pgx

ENV_ID = "2048"


class BatchRunner:
    """
    BatchRunner class for running batched 2048 environments with an action function.

    This class provides functionality to run multiple 2048 game environments in parallel,
    collecting observations, actions, log probabilities, values, rewards, and termination
    flags for reinforcement learning training.
    """

    def __init__(self, init_seed: int, act_fn: Callable = None):
        """
        Initialize the BatchRunner class.

        Parameters
        ----------
        init_seed : int
            Initial random seed for JAX random number generation.
        act_fn : Callable, optional
            Function to choose actions. Should accept (key, observation, legal_action_mask)
            and return (action, log_prob, value). If None, must be set before running.
            Defaults to None.
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

        Returns
        -------
        Callable or None
            The current action function, or None if not set.
        """
        return self._act_fn

    @act_fn.setter
    def act_fn(self, act_fn: Callable):
        """
        Set the action function.

        Parameters
        ----------
        act_fn : Callable
            Function to choose actions. Should accept (key, observation, legal_action_mask)
            and return (action, log_prob, value). The function will be JIT-compiled
            and vectorized for batch processing.
        """
        self._act_fn = jax.jit(jax.vmap(act_fn))

    def run_actions_batch(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run actions in the 2048 environment in parallel environments with a
        batch size of batch_size.

        Parameters
        ----------
        batch_size : int
            Number of parallel environments to run.

        Returns
        -------
        observations : np.ndarray
            Array of observations from the environments. Shape (batch_size, num_steps, 16).
        actions : np.ndarray
            Array of actions taken in the environments. Shape (batch_size, num_steps).
        log_probs : np.ndarray
            Array of log probabilities of the actions taken. Shape (batch_size, num_steps).
        values : np.ndarray
            Array of value estimates from the policy. Shape (batch_size, num_steps).
        rewards : np.ndarray
            Array of rewards received from the environments. Shape (batch_size, num_steps).
        terminations : np.ndarray
            Array of termination flags from the environments. Shape (batch_size, num_steps).

        Raises
        ------
        ValueError
            If the action function is not set.
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
        log_probs = []
        values = []
        rewards = []
        terminations = []
        while not (state.terminated | state.truncated).all():
            self.key, subkey = jax.random.split(self.key)
            subkey = jax.random.split(subkey, batch_size)
            # Store the observation
            observation = state.observation
            action, log_prob, value = self.act_fn(
                subkey, observation, state.legal_action_mask
            )
            self.key, subkey = jax.random.split(self.key)
            subkey = jax.random.split(subkey, batch_size)
            state = self.step_fn(state, action, subkey)
            # Store the observations, actions, rewards, and terminations
            observations.append(observation)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(state.rewards)
            terminations.append(state.terminated | state.truncated)
        # Convert lists to arrays
        observations = np.stack(observations, axis=1)
        actions = np.stack(actions, axis=1)
        log_probs = np.stack(log_probs, axis=1)
        values = np.stack(values, axis=1)
        rewards = np.concatenate(rewards, axis=1)
        terminations = np.stack(terminations, axis=1)

        return observations, actions, log_probs, values, rewards, terminations

    def run_rollout_batch(self, batch_size: int) -> list:
        """
        Run actions in the 2048 environment in parallel environments with a
        batch size of batch_size.

        Parameters
        ----------
        batch_size : int
            Number of parallel environments to run.

        Returns
        -------
        list
            A list containing all the states.
        """
        # Check if the action function is set
        if self.act_fn is None:
            raise ValueError("The action function is not set.")
        # Initialize the states
        self.key, subkey = jax.random.split(self.key)
        subkey = jax.random.split(subkey, batch_size)
        state = self.init_fn(subkey)

        # Run the environment
        states = [state]
        while not (state.terminated | state.truncated).all():
            self.key, subkey = jax.random.split(self.key)
            subkey = jax.random.split(subkey, batch_size)
            # Store the observation
            observation = state.observation
            action, log_prob, value = self.act_fn(
                subkey, observation, state.legal_action_mask
            )
            self.key, subkey = jax.random.split(self.key)
            subkey = jax.random.split(subkey, batch_size)
            state = self.step_fn(state, action, subkey)
            # Store the state
            states.append(state)

        return states
