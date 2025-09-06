import jax
import jax.numpy as jnp


def act_randomly(rng_key: jax.Array, obs: jax.Array, mask: jax.Array):
    """
    Random action selection strategy that chooses uniformly from legal actions.

    This is a simple baseline policy that ignores the game state observation and
    randomly selects from available legal actions with equal probability.
    This function does not work with batched inputs.

    Parameters
    ----------
    rng_key : jax.Array
        Random key used for action sampling.
    obs : jax.Array
        Game state observation with shape (4, 4, 31). Not used in action selection.
    mask : jax.Array
        Legal action mask with shape (4,). Boolean array where True indicates
        a legal action. Actions correspond to: [Up, Left, Right, Down].

    Returns
    -------
    tuple[jax.Array, jax.Array, None]
        A tuple containing:
        - action : jax.Array (scalar)
            Selected action index (0=Up, 1=Left, 2=Right, 3=Down).
        - log_prob : jax.Array (scalar)
            Log probability of the selected action.
        - value : None
            State value (None for this simple heuristic).

    """
    # Check shapes
    assert obs.shape == (4, 4, 31)
    assert mask.shape == (4,)

    # Calculate probabilities for legal actions
    num_legal_actions = mask.sum()
    # Handle edge case where no actions are legal
    probs = jnp.where(
        num_legal_actions > 0, mask / num_legal_actions, jnp.ones_like(mask) / 4
    )
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)

    # Sample action
    action = jax.random.categorical(rng_key, logits=logits)

    # Calculate log probability of the selected action
    log_prob = jnp.log(probs[action])

    # Set value to None for this simple heuristic
    value = None

    return action, log_prob, value
