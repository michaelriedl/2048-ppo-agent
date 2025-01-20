import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import pytest

from src.ppo.rollout_buffer import RolloutBuffer


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("obs_dim", [1, 10, 100])
@pytest.mark.parametrize("act_dim", [1, 10, 100])
@pytest.mark.parametrize("max_steps", [1, 10, 100])
def test_rollout_buffer(batch_size, obs_dim, act_dim, max_steps):
    buffer = RolloutBuffer(batch_size, obs_dim, act_dim, max_steps)
    assert buffer.obs_buffer.shape == (batch_size, max_steps, obs_dim)
    assert buffer.act_buffer.shape == (batch_size, max_steps, act_dim)
    assert buffer.adv_buffer.shape == (batch_size, max_steps)
    assert buffer.rew_buffer.shape == (batch_size, max_steps)
    assert buffer.ret_buffer.shape == (batch_size, max_steps)
    assert buffer.val_buffer.shape == (batch_size, max_steps)
    assert buffer.logp_buffer.shape == (batch_size, max_steps)
    assert buffer.gamma == 0.99
    assert buffer.lam == 0.95
    assert buffer.obs_buffer.dtype == "float32"
    assert buffer.act_buffer.dtype == "float32"
    assert buffer.adv_buffer.dtype == "float32"
    assert buffer.rew_buffer.dtype == "float32"
    assert buffer.ret_buffer.dtype == "float32"
    assert buffer.val_buffer.dtype == "float32"
    assert buffer.logp_buffer.dtype == "float32"
