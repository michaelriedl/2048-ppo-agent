import jax
import jax.numpy as jnp


def act_drul(rng_key: jax.Array, obs: jax.Array, mask: jax.Array):
    """
    Ignore observation and prioritize actions in the order of down, right, up, left.

    Parameters
    ----------
    rng_key : jax.Array
        Random key.
    obs : jax.Array
        Observation.
    mask : jax.Array
        Legal action mask.

    Returns
    -------
    jax.Array
        Action.

    """
    # Prioritize actions in the order of down, right, up, left
    action_order = jnp.array([3, 2, 1, 0], dtype=jnp.int32)

    # Reorder the mask
    mask = mask[:, action_order]

    # Choose the first legal action
    action = action_order[mask.argmax(axis=-1)]

    return action
