import jax
import jax.numpy as jnp


def act_drul(rng_key: jax.Array, obs: jax.Array, mask: jax.Array):
    """
    Ignore observation and prioritize actions in the order of down, right, up, left.
    This function does not work with batched inputs.

    Parameters
    ----------
    rng_key : jax.Array
        Random key.
    obs : jax.Array
        Observation. The shape of the observation should be (4, 4, 31).
    mask : jax.Array
        Legal action mask. The mask should be a one-hot vector of shape (4,).

    Returns
    -------
    jax.Array
        Action.

    """
    # Check shapes
    assert obs.shape == (4, 4, 31)
    assert mask.shape == (4,)
    # Prioritize actions in the order of down, right, up, left
    action_order = jnp.array([3, 2, 1, 0], dtype=jnp.int32)
    # Reorder the mask
    mask = mask[action_order]
    # Choose the first legal action
    action = action_order[mask.argmax()]

    return action
