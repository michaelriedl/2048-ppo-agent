import jax
import pgx
from typing import Callable
from pgx import State

ENV_ID = "2048"


def run_actions_batch(init_seed: int, batch_size: int, act_fn: Callable) -> list[State]:
    """
    Run actions in the 2048 environment in parallel environments with a
    batch size of batch_size.

    Parameters
    ----------
    init_seed : int
        Initial random seed.
    batch_size : int
        Batch size.
    act_fn : Callable
        Function to choose actions.

    Returns
    -------
    states : list[State]
        List of states.

    """
    # Create the random key
    key = jax.random.key(seed=init_seed)

    # Create a new environment
    env = pgx.make(ENV_ID)

    # Create the parallel functions
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))
    act_fn = jax.jit(jax.vmap(act_fn))

    # Initialize the states
    key, subkey = jax.random.split(key)
    subkey = jax.random.split(subkey, batch_size)
    state = init_fn(subkey)

    # Run the environment
    states = []
    while not (state.terminated | state.truncated).all():
        key, subkey = jax.random.split(key)
        subkey = jax.random.split(subkey, batch_size)
        action = act_fn(subkey, state.observation, state.legal_action_mask)
        key, subkey = jax.random.split(key)
        subkey = jax.random.split(subkey, batch_size)
        state = step_fn(state, action, subkey)
        states.append(state)

    return states
