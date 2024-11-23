import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import pytest
import jax
from src.actions.act_drul import act_drul


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_call(batch_size):
    rng_key = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng_key, (batch_size, 4, 4, 31))
    mask = jax.random.uniform(rng_key, (batch_size, 4)) > 0.5
    mask = mask.astype(jax.numpy.bool)
    act = act_drul(rng_key, obs, mask)
    assert act.shape == (batch_size,)
    assert (act >= 0).all()
    assert (act < 4).all()


def test_legal_actions():
    rng_key = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng_key, (1, 4, 4, 31))
    mask = jax.numpy.array([[True, False, True, False]])
    act = act_drul(rng_key, obs, mask)
    assert act.shape == (1,)
    assert act == 2
