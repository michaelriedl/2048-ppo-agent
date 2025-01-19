from typing import Callable

import numpy as np

from ..running_stats_vec import RunningStatsVec
from .batch_runner import BatchRunner


def run_actions_max_tile(
    init_seed: int, batch_size: int, num_envs: int, act_fn: Callable
) -> RunningStatsVec:
    """
    Run actions in the 2048 environment and return the maximum tile statistics.

    Parameters
    ----------
    init_seed : int
        Initial random seed.
    batch_size : int
        Batch size.
    num_envs : int
        Number of environments. The number of environments must be divisible by the batch size.
        If not, the number of environments will be adjusted to be divisible by the batch size.
    act_fn : Callable
        Function to choose actions.

    Returns
    -------
    RunningStatsVec
        Running statistics for the maximum tile.

    """
    # Warn if the number of environments is not divisible by the batch size
    if num_envs % batch_size != 0:
        Warning(
            f"The number of environments ({num_envs}) is not divisible by the batch size ({batch_size})."
            "The number of environments will be adjusted to be divisible by the batch size."
        )

    # Create the batch runner
    runner = BatchRunner(init_seed=init_seed, act_fn=act_fn)

    # Run the environments
    running_stats_max_tile = RunningStatsVec()
    for _ in range(num_envs // batch_size):

        # Run the batch of environments
        states = runner.run_actions_batch(batch_size)

        # Convert the boards to numpy
        boards = [s._board for s in states]
        boards = np.stack(boards)

        # Update the running stats
        max_tiles = (2**boards).max(-1)
        running_stats_max_tile.push(max_tiles)

    return running_stats_max_tile
