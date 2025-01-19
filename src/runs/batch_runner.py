from typing import Callable

import jax
import pgx
from pgx import State

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

    def run_actions_batch(self, batch_size: int) -> list[State]:
        """
        Run actions in the 2048 environment in parallel environments with a
        batch size of batch_size.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        states : list[State]
            List of states.
        """
        # Check if the action function is set
        if self.act_fn is None:
            raise ValueError("The action function is not set.")
        # Initialize the states
        self.key, subkey = jax.random.split(self.key)
        subkey = jax.random.split(subkey, batch_size)
        state = self.init_fn(subkey)

        # Run the environment
        states = []
        while not (state.terminated | state.truncated).all():
            self.key, subkey = jax.random.split(self.key)
            subkey = jax.random.split(subkey, batch_size)
            action = self.act_fn(subkey, state.observation, state.legal_action_mask)
            self.key, subkey = jax.random.split(self.key)
            subkey = jax.random.split(subkey, batch_size)
            state = self.step_fn(state, action, subkey)
            states.append(state)

        return states
