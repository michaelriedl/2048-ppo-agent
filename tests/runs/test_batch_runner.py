import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import jax
import numpy as np
import pytest

from src.runs.batch_runner import BatchRunner


def act_fn(key, obs, mask):
    action = jax.random.randint(key, (), 0, 4)
    log_prob = jax.random.normal(key, ())  # Mock log probability
    value = jax.random.normal(key, ())  # Mock value estimate
    return action, log_prob, value


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
    observations, actions, action_masks, log_probs, values, rewards, terminations = (
        runner.run_actions_batch(batch_size)
    )

    # Check that we get 7 arrays back
    assert isinstance(observations, np.ndarray)
    assert isinstance(actions, np.ndarray)
    assert isinstance(action_masks, np.ndarray)
    assert isinstance(log_probs, np.ndarray)
    assert isinstance(values, np.ndarray)
    assert isinstance(rewards, np.ndarray)
    assert isinstance(terminations, np.ndarray)

    # Check batch dimensions
    assert observations.shape[0] == batch_size
    assert actions.shape[0] == batch_size
    assert action_masks.shape[0] == batch_size
    assert log_probs.shape[0] == batch_size
    assert values.shape[0] == batch_size
    assert rewards.shape[0] == batch_size
    assert terminations.shape[0] == batch_size

    # Check that all arrays have the same number of time steps
    num_steps = observations.shape[1]
    assert actions.shape[1] == num_steps
    assert action_masks.shape[1] == num_steps
    assert log_probs.shape[1] == num_steps
    assert values.shape[1] == num_steps
    assert rewards.shape[1] == num_steps
    assert terminations.shape[1] == num_steps

    # Check that observations have correct shape (batch_size, num_steps, 4, 4, 31)
    assert observations.shape[2] == 4
    assert observations.shape[3] == 4
    assert observations.shape[4] == 31

    # Check that action_masks have correct shape (batch_size, num_steps, 4)
    assert action_masks.shape[2] == 4

    # Check that all episodes terminated
    assert terminations[:, -1].all()


def test_run_actions_batch_no_act_fn():
    """Test that run_actions_batch raises ValueError when no action function is set."""
    runner = BatchRunner(init_seed=0, act_fn=None)

    with pytest.raises(ValueError, match="The action function is not set"):
        runner.run_actions_batch(batch_size=2)


def test_run_actions_batch_deterministic():
    """Test that running with the same seed produces deterministic results."""
    runner1 = BatchRunner(init_seed=42, act_fn=act_fn)
    runner2 = BatchRunner(init_seed=42, act_fn=act_fn)

    obs1, actions1, action_masks1, log_probs1, values1, rewards1, terms1 = (
        runner1.run_actions_batch(batch_size=3)
    )
    obs2, actions2, action_masks2, log_probs2, values2, rewards2, terms2 = (
        runner2.run_actions_batch(batch_size=3)
    )

    # Results should be identical with same seed
    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(actions1, actions2)
    np.testing.assert_array_equal(action_masks1, action_masks2)
    np.testing.assert_array_equal(log_probs1, log_probs2)
    np.testing.assert_array_equal(values1, values2)
    np.testing.assert_array_equal(rewards1, rewards2)
    np.testing.assert_array_equal(terms1, terms2)


def test_run_actions_batch_return_values():
    """Test that log_probs and values have expected properties."""
    runner = BatchRunner(init_seed=0, act_fn=act_fn)
    observations, actions, action_masks, log_probs, values, rewards, terminations = (
        runner.run_actions_batch(batch_size=3)
    )

    # Check that log_probs and values are finite numbers
    assert np.isfinite(log_probs).all(), "Log probabilities should be finite"
    assert np.isfinite(values).all(), "Values should be finite"

    # Check that actions are in valid range [0, 3]
    assert (actions >= 0).all() and (
        actions <= 3
    ).all(), "Actions should be in range [0, 3]"

    # Check that action_masks are boolean-like (0 or 1)
    assert np.isin(action_masks, [0, 1]).all(), "Action masks should be 0 or 1"

    # Check that terminations are boolean-like (0 or 1)
    assert np.isin(terminations, [0, 1]).all(), "Terminations should be 0 or 1"
