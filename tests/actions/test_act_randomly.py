import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import jax
import pytest

from src.actions.act_randomly import act_randomly


def test_call():
    rng_key = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng_key, (4, 4, 31))
    mask = jax.random.uniform(rng_key, (4,)) > 0.5
    mask = mask.astype(jax.numpy.bool)
    action, log_prob, value = act_randomly(rng_key, obs, mask)
    assert action.shape == ()
    assert (action >= 0).all()
    assert (action < 4).all()
    assert log_prob.shape == ()
    assert jax.numpy.isfinite(log_prob)
    assert value is None


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_call_vmap(batch_size):
    rng_key = jax.random.PRNGKey(0)
    subkey = jax.random.split(rng_key, batch_size)
    obs = jax.random.uniform(rng_key, (batch_size, 4, 4, 31))
    mask = jax.random.uniform(rng_key, (batch_size, 4)) > 0.5
    mask = mask.astype(jax.numpy.bool)
    act_fn = jax.jit(jax.vmap(act_randomly))
    actions, log_probs, values = act_fn(subkey, obs, mask)
    assert actions.shape == (batch_size,)
    assert (actions >= 0).all()
    assert (actions < 4).all()
    assert log_probs.shape == (batch_size,)
    assert jax.numpy.isfinite(log_probs).all()
    assert values is None


def test_legal_actions():
    rng_key = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng_key, (4, 4, 31))
    mask = jax.numpy.array([True, False, True, False])
    action, log_prob, value = act_randomly(rng_key, obs, mask)
    assert action.shape == ()
    assert (action == 0) or (action == 2)  # Should choose from legal actions only
    assert log_prob.shape == ()
    assert jax.numpy.isclose(
        log_prob, jax.numpy.log(0.5)
    )  # Equal probability for 2 legal actions
    assert value is None
