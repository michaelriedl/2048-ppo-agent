import jax
import jax.numpy as jnp


def act_randomly(rng_key: jax.Array, obs: jax.Array, mask: jax.Array):
    """
    Ignore observation and choose randomly from legal actions. This function does not
    work with batched inputs.

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
        Randomly chosen action.

    """
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)

    return jax.random.categorical(rng_key, logits=logits)
