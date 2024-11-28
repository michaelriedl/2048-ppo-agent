import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import pytest
import jax
from src.runs.run_actions_max_tile import run_actions_max_tile


def act_fn(key, obs, mask):
    return jax.random.randint(key, (), 0, 4)


@pytest.mark.parametrize("batch_size", [2, 5])
def test_call(batch_size):
    num_sims = 10
    running_stats = run_actions_max_tile(0, batch_size, num_sims, act_fn)
    assert running_stats.num_samples[0] == num_sims


@pytest.mark.parametrize("batch_size", [2, 5])
@pytest.mark.parametrize("num_sims", [9, 15])
def test_mismatch_batch_size(batch_size, num_sims):
    running_stats = run_actions_max_tile(0, batch_size, num_sims, act_fn)
    assert running_stats.num_samples[0] == (num_sims // batch_size) * batch_size
