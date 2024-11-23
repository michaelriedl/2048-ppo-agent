import jax
import jax.numpy as jnp


def act_randomly(rng_key: jax.Array, obs: jax.Array, mask: jax.Array):
    """
    Ignore observation and choose randomly from legal actions.

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
        Randomly chosen action.

    """
    probs = mask / mask.sum(axis=-1, keepdims=True)
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)

    return jax.random.categorical(rng_key, logits=logits, axis=-1)
