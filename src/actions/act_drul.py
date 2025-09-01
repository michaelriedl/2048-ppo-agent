import jax
import jax.numpy as jnp


def act_drul(rng_key: jax.Array, obs: jax.Array, mask: jax.Array):
    """
    Deterministic action selection strategy that prioritizes actions in the order:
    Down (3), Right (2), Up (1), Left (0).

    This is a simple heuristic policy that ignores the game state observation and
    always selects the first legal action from the priority order. This function
    does not work with batched inputs.

    Parameters
    ----------
    rng_key : jax.Array
        Random key (unused in this deterministic strategy, included for API consistency).
    obs : jax.Array
        Game state observation with shape (4, 4, 31). Not used in action selection.
    mask : jax.Array
        Legal action mask with shape (4,). Boolean array where True indicates
        a legal action. Actions correspond to: [Up, Left, Right, Down].

    Returns
    -------
    tuple[jax.Array, None, None]
        A tuple containing:
        - action : jax.Array (scalar)
            Selected action index (0=Up, 1=Left, 2=Right, 3=Down).
        - log_prob : None
            Log probability (None for deterministic actions).
        - value : None
            State value (None for this simple heuristic).

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
    # Set the log probability and value to adhere to the action pattern
    log_prob = None
    value = None

    return action, log_prob, value
