import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import jax
import pytest
from pgx import State

from src.runs.run_actions_batch import run_actions_batch


def act_fn(key, obs, mask):
    return jax.random.randint(key, (), 0, 4)


@pytest.mark.parametrize("batch_size", [2, 5, 10])
def test_call(batch_size):
    states = run_actions_batch(0, batch_size, act_fn)
    assert states[0].observation.shape[0] == batch_size
    assert all([isinstance(s, State) for s in states])
