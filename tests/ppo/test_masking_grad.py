import pytest
import torch
from torch.distributions.categorical import Categorical


def test_invalid_action_masking_gradient():
    """Test that gradients flow correctly through invalid action masking.

    This test verifies that when actions are masked out (set to invalid),
    the gradient computation still works correctly for valid actions.
    """
    action = 0
    advantage = torch.tensor(1.0)

    # Setup target logits with gradient tracking
    target_logits = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        requires_grad=True,
    )

    # Create mask where action 2 is invalid
    invalid_action_masks = torch.tensor(
        [
            1.0,
            1.0,
            0.0,
            1.0,
        ]
    )
    invalid_action_masks = invalid_action_masks.type(torch.BoolTensor)

    # Apply masking by subtracting large value from invalid actions
    adjusted_logits = target_logits - (
        1e8 * (1 - invalid_action_masks.type(torch.FloatTensor))
    )

    # Create categorical distribution and compute log probability
    adjusted_probs = Categorical(logits=adjusted_logits)
    adjusted_log_prob = adjusted_probs.log_prob(torch.tensor(action))

    # Compute gradients
    (adjusted_log_prob * advantage).backward()

    # Verify gradients exist and are reasonable
    assert target_logits.grad is not None, "Gradients should be computed"
    assert not torch.isnan(target_logits.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(target_logits.grad).any(), "Gradients should not contain Inf"

    # Verify probability distribution properties
    assert torch.allclose(
        adjusted_probs.probs.sum(), torch.tensor(1.0), atol=1e-6
    ), "Probabilities should sum to 1"
    assert (adjusted_probs.probs >= 0).all(), "All probabilities should be non-negative"

    # Verify that masked action has near-zero probability
    assert (
        adjusted_probs.probs[2] < 1e-6
    ), "Masked action should have near-zero probability"


def test_masking_with_multiple_invalid_actions():
    """Test masking when multiple actions are invalid."""
    target_logits = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    # Mask out actions 1 and 3
    invalid_action_masks = torch.tensor([1.0, 0.0, 1.0, 0.0])
    invalid_action_masks = invalid_action_masks.type(torch.BoolTensor)

    adjusted_logits = target_logits - (
        1e8 * (1 - invalid_action_masks.type(torch.FloatTensor))
    )

    adjusted_probs = Categorical(logits=adjusted_logits)

    # Check that masked actions have near-zero probability
    assert (
        adjusted_probs.probs[1] < 1e-6
    ), "Masked action 1 should have near-zero probability"
    assert (
        adjusted_probs.probs[3] < 1e-6
    ), "Masked action 3 should have near-zero probability"

    # Check that valid actions have non-zero probability
    assert (
        adjusted_probs.probs[0] > 0.1
    ), "Valid action 0 should have non-zero probability"
    assert (
        adjusted_probs.probs[2] > 0.1
    ), "Valid action 2 should have non-zero probability"

    # Probabilities should still sum to 1
    assert torch.allclose(
        adjusted_probs.probs.sum(), torch.tensor(1.0), atol=1e-6
    ), "Probabilities should sum to 1"


def test_masking_all_but_one_action():
    """Test edge case where only one action is valid."""
    target_logits = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    # Only action 2 is valid
    invalid_action_masks = torch.tensor([0.0, 0.0, 1.0, 0.0])
    invalid_action_masks = invalid_action_masks.type(torch.BoolTensor)

    adjusted_logits = target_logits - (
        1e8 * (1 - invalid_action_masks.type(torch.FloatTensor))
    )

    adjusted_probs = Categorical(logits=adjusted_logits)

    # The only valid action should have probability ~1
    assert torch.allclose(
        adjusted_probs.probs[2], torch.tensor(1.0), atol=1e-6
    ), "Single valid action should have probability ~1"

    # All other actions should have near-zero probability
    assert adjusted_probs.probs[0] < 1e-6
    assert adjusted_probs.probs[1] < 1e-6
    assert adjusted_probs.probs[3] < 1e-6
