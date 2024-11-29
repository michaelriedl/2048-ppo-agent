import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import pytest
import jax
from pgx import State
from src.runs.batch_runner import BatchRunner


def act_fn(key, obs, mask):
    return jax.random.randint(key, (), 0, 4)


@pytest.mark.parametrize("action_fn", [None, act_fn])
def test_init(action_fn):
    runner = BatchRunner(init_seed=0, act_fn=action_fn)
    if action_fn is not None:
        assert runner.act_fn is not None
    else:
        assert runner.act_fn is None


@pytest.mark.parametrize("action_fn", [None, act_fn])
def test_set_act_fn(action_fn):
    runner = BatchRunner(init_seed=0, act_fn=action_fn)
    runner.act_fn = act_fn
    assert runner.act_fn is not None


@pytest.mark.parametrize("batch_size", [2, 5, 10])
def test_run_actions_batch(batch_size):
    runner = BatchRunner(init_seed=0, act_fn=act_fn)
    states = runner.run_actions_batch(batch_size)
    assert states[0].observation.shape[0] == batch_size
    assert all([isinstance(s, State) for s in states])
