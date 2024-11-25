import jax
import pgx
import numpy as np
from typing import Callable
from ..running_stats_vec import RunningStatsVec


def run_actions(
    init_seed: int, batch_size: int, num_sims: int, act_fn: Callable
) -> RunningStatsVec:
    """
    Run actions in the 2048 environment.

    Parameters
    ----------
    init_seed : int
        Initial random seed.
    batch_size : int
        Batch size.
    num_sims : int
        Number of simulations. The number of simulations must be divisible by the batch size.
        If not, the number of simulations will be adjusted to be divisible by the batch size.
    act_fn : Callable
        Function to choose actions.

    Returns
    -------
    RunningStatsVec
        Running statistics for the maximum tile.

    """
    # Warn if the number of simulations is not divisible by the batch size
    if num_sims % batch_size != 0:
        Warning(
            f"The number of simulations ({num_sims}) is not divisible by the batch size ({batch_size})."
            "The number of simulations will be adjusted to be divisible by the batch size."
        )

    # Set the environment id
    env_id = "2048"

    # Create the random key
    key = jax.random.key(seed=init_seed)

    # Create a new environment
    env = pgx.make(env_id)

    # Create the parallel functions
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))
    # act_fn = jax.jit(jax.vmap(act_fn))

    # Run the simulations
    running_stats_max_tile = RunningStatsVec()
    for _ in range(num_sims // batch_size):

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

        # Convert the boards to numpy
        boards = [s._board for s in states]
        boards = np.stack(boards)

        # Update the running stats
        max_tiles = (2**boards).max(-1)
        running_stats_max_tile.push(max_tiles)

    return running_stats_max_tile
