import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np

from src.ppo.rollout_buffer import RolloutBuffer


class TestRolloutBuffer:
    """Test class for RolloutBuffer functionality."""

    def test_initialization(self):
        """Test that RolloutBuffer initializes correctly."""
        buffer = RolloutBuffer(
            total_buffer_size=100,
            observation_dim=16,
            observation_length=10,
            action_dim=4,
        )

        assert buffer.total_buffer_size == 100
        assert buffer.observation_dim == 16
        assert buffer.observation_length == 10
        assert buffer.action_dim == 4
        assert buffer.buffer_size == 0

        # Check buffer shapes
        assert buffer.observation_buffer.shape == (100, 10, 16)
        assert buffer.action_buffer.shape == (100, 4)
        assert buffer.reward_buffer.shape == (100,)
        assert buffer.value_buffer.shape == (100,)
        assert buffer.log_prob_buffer.shape == (100,)
        assert buffer.termination_buffer.shape == (100,)

    def test_reset(self):
        """Test that reset method clears all buffers."""
        buffer = RolloutBuffer(
            total_buffer_size=10, observation_dim=4, observation_length=5, action_dim=2
        )

        # Fill buffers with some data
        buffer.observation_buffer.fill(1.0)
        buffer.action_buffer.fill(2.0)
        buffer.reward_buffer.fill(3.0)
        buffer.buffer_size = 5

        # Reset and verify
        buffer.reset()

        assert buffer.buffer_size == 0
        assert np.all(buffer.observation_buffer == 0)
        assert np.all(buffer.action_buffer == 0)
        assert np.all(buffer.reward_buffer == 0)
        assert np.all(buffer.value_buffer == 0)
        assert np.all(buffer.log_prob_buffer == 0)
        assert np.all(buffer.termination_buffer == 0)

    def test_store_batch_single_episode(self):
        """Test storing a batch with single episodes that terminate."""
        buffer = RolloutBuffer(
            total_buffer_size=50, observation_dim=4, observation_length=5, action_dim=2
        )

        # Create test data for 2 environments, each with 5 timesteps
        batch_size = 2
        time_steps = 5

        observations = np.random.randn(batch_size, time_steps, 5, 4).astype(np.float32)
        actions = np.random.randn(batch_size, time_steps, 2).astype(np.float32)
        rewards = np.random.randn(batch_size, time_steps).astype(np.float32)
        values = np.random.randn(batch_size, time_steps).astype(np.float32)
        log_probs = np.random.randn(batch_size, time_steps).astype(np.float32)

        # Create terminations - first env terminates at step 3, second at step 4
        terminations = np.zeros((batch_size, time_steps), dtype=bool)
        terminations[0, 3] = True
        terminations[1, 4] = True

        buffer.store_batch(
            observations, actions, rewards, values, log_probs, terminations
        )

        # Should store 4 steps from first env + 5 steps from second env = 9 total
        assert buffer.buffer_size == 9

        # Check that data was stored correctly
        data = buffer.get_buffer_data()
        assert data["observations"].shape == (9, 5, 4)
        assert data["actions"].shape == (9, 2)
        assert data["rewards"].shape == (9,)
        assert data["terminations"].shape == (9,)

    def test_store_batch_no_termination(self):
        """Test storing a batch where no episodes terminate."""
        buffer = RolloutBuffer(
            total_buffer_size=50, observation_dim=4, observation_length=5, action_dim=2
        )

        batch_size = 2
        time_steps = 5

        observations = np.random.randn(batch_size, time_steps, 5, 4).astype(np.float32)
        actions = np.random.randn(batch_size, time_steps, 2).astype(np.float32)
        rewards = np.random.randn(batch_size, time_steps).astype(np.float32)
        values = np.random.randn(batch_size, time_steps).astype(np.float32)
        log_probs = np.random.randn(batch_size, time_steps).astype(np.float32)
        terminations = np.zeros((batch_size, time_steps), dtype=bool)

        buffer.store_batch(
            observations, actions, rewards, values, log_probs, terminations
        )

        # Should store 0 steps since no terminations
        assert buffer.buffer_size == 0

    def test_store_batch_buffer_overflow(self):
        """Test that store_batch handles buffer overflow correctly."""
        buffer = RolloutBuffer(
            total_buffer_size=5,
            observation_dim=2,
            observation_length=3,
            action_dim=1,  # Small buffer
        )

        batch_size = 1
        time_steps = 10  # More steps than buffer can hold

        observations = np.random.randn(batch_size, time_steps, 3, 2).astype(np.float32)
        actions = np.random.randn(batch_size, time_steps, 1).astype(np.float32)
        rewards = np.random.randn(batch_size, time_steps).astype(np.float32)
        values = np.random.randn(batch_size, time_steps).astype(np.float32)
        log_probs = np.random.randn(batch_size, time_steps).astype(np.float32)

        # Terminate at step 8 (would be 9 steps total)
        terminations = np.zeros((batch_size, time_steps), dtype=bool)
        terminations[0, 8] = True

        buffer.store_batch(
            observations, actions, rewards, values, log_probs, terminations
        )

        # Should only store 5 steps (buffer capacity)
        assert buffer.buffer_size == 5

    def test_get_buffer_data(self):
        """Test that get_buffer_data returns correct data structure."""
        buffer = RolloutBuffer(
            total_buffer_size=10, observation_dim=3, observation_length=4, action_dim=2
        )

        # Simulate some stored data
        buffer.buffer_size = 5
        buffer.observation_buffer[:5] = np.ones((5, 4, 3))
        buffer.action_buffer[:5] = np.ones((5, 2)) * 2
        buffer.reward_buffer[:5] = np.ones(5) * 3
        buffer.value_buffer[:5] = np.ones(5) * 4
        buffer.log_prob_buffer[:5] = np.ones(5) * 5
        buffer.termination_buffer[:5] = True

        data = buffer.get_buffer_data()

        # Check structure and content
        expected_keys = {
            "observations",
            "actions",
            "rewards",
            "values",
            "log_probs",
            "terminations",
        }
        assert set(data.keys()) == expected_keys

        assert data["observations"].shape == (5, 4, 3)
        assert data["actions"].shape == (5, 2)
        assert data["rewards"].shape == (5,)
        assert np.all(data["observations"] == 1)
        assert np.all(data["actions"] == 2)
        assert np.all(data["rewards"] == 3)
        assert np.all(data["values"] == 4)
        assert np.all(data["log_probs"] == 5)
        assert np.all(data["terminations"] == True)

    def test_multiple_store_calls(self):
        """Test that multiple calls to store_batch accumulate data correctly."""
        buffer = RolloutBuffer(
            total_buffer_size=20, observation_dim=2, observation_length=3, action_dim=1
        )

        # First batch
        batch_size = 1
        time_steps = 3

        observations1 = np.ones((batch_size, time_steps, 3, 2), dtype=np.float32)
        actions1 = np.ones((batch_size, time_steps, 1), dtype=np.float32)
        rewards1 = np.ones((batch_size, time_steps), dtype=np.float32)
        values1 = np.ones((batch_size, time_steps), dtype=np.float32)
        log_probs1 = np.ones((batch_size, time_steps), dtype=np.float32)
        terminations1 = np.zeros((batch_size, time_steps), dtype=bool)
        terminations1[0, 2] = True  # Terminate at step 2

        buffer.store_batch(
            observations1, actions1, rewards1, values1, log_probs1, terminations1
        )

        assert buffer.buffer_size == 3

        # Second batch
        observations2 = np.ones((batch_size, time_steps, 3, 2), dtype=np.float32) * 2
        actions2 = np.ones((batch_size, time_steps, 1), dtype=np.float32) * 2
        rewards2 = np.ones((batch_size, time_steps), dtype=np.float32) * 2
        values2 = np.ones((batch_size, time_steps), dtype=np.float32) * 2
        log_probs2 = np.ones((batch_size, time_steps), dtype=np.float32) * 2
        terminations2 = np.zeros((batch_size, time_steps), dtype=bool)
        terminations2[0, 1] = True  # Terminate at step 1

        buffer.store_batch(
            observations2, actions2, rewards2, values2, log_probs2, terminations2
        )

        assert buffer.buffer_size == 5  # 3 + 2

        data = buffer.get_buffer_data()

        # Check that first batch data is preserved
        assert np.all(data["observations"][:3] == 1)
        assert np.all(data["rewards"][:3] == 1)

        # Check that second batch data is appended
        assert np.all(data["observations"][3:5] == 2)
        assert np.all(data["rewards"][3:5] == 2)

    def test_empty_batch_store(self):
        """Test storing empty batch (batch_size=0)."""
        buffer = RolloutBuffer(
            total_buffer_size=10, observation_dim=2, observation_length=5, action_dim=1
        )

        # Create empty batch
        observations = np.empty((0, 5, 5, 2), dtype=np.float32)
        actions = np.empty((0, 5, 1), dtype=np.float32)
        rewards = np.empty((0, 5), dtype=np.float32)
        values = np.empty((0, 5), dtype=np.float32)
        log_probs = np.empty((0, 5), dtype=np.float32)
        terminations = np.empty((0, 5), dtype=bool)

        buffer.store_batch(
            observations, actions, rewards, values, log_probs, terminations
        )

        # Buffer should remain empty
        assert buffer.buffer_size == 0

    def test_termination_at_first_step(self):
        """Test behavior when episode terminates at the very first step."""
        buffer = RolloutBuffer(
            total_buffer_size=10, observation_dim=2, observation_length=5, action_dim=1
        )

        batch_size = 1
        time_steps = 5

        observations = np.ones((batch_size, time_steps, 5, 2), dtype=np.float32)
        actions = np.ones((batch_size, time_steps, 1), dtype=np.float32)
        rewards = np.ones((batch_size, time_steps), dtype=np.float32)
        values = np.ones((batch_size, time_steps), dtype=np.float32)
        log_probs = np.ones((batch_size, time_steps), dtype=np.float32)

        # Terminate at step 0
        terminations = np.zeros((batch_size, time_steps), dtype=bool)
        terminations[0, 0] = True

        buffer.store_batch(
            observations, actions, rewards, values, log_probs, terminations
        )

        # Should store 1 step (the termination step)
        assert buffer.buffer_size == 1

    def test_multiple_terminations_in_episode(self):
        """Test behavior when there are multiple terminations in one episode."""
        buffer = RolloutBuffer(
            total_buffer_size=10, observation_dim=2, observation_length=5, action_dim=1
        )

        batch_size = 1
        time_steps = 5

        observations = np.ones((batch_size, time_steps, 5, 2), dtype=np.float32)
        actions = np.ones((batch_size, time_steps, 1), dtype=np.float32)
        rewards = np.ones((batch_size, time_steps), dtype=np.float32)
        values = np.ones((batch_size, time_steps), dtype=np.float32)
        log_probs = np.ones((batch_size, time_steps), dtype=np.float32)

        # Multiple terminations - should only use the first one
        terminations = np.zeros((batch_size, time_steps), dtype=bool)
        terminations[0, 2] = True
        terminations[0, 4] = True

        buffer.store_batch(
            observations, actions, rewards, values, log_probs, terminations
        )

        # Should store only up to first termination (3 steps: 0, 1, 2)
        assert buffer.buffer_size == 3

    def test_dtype_preservation(self):
        """Test that data types are preserved correctly."""
        buffer = RolloutBuffer(
            total_buffer_size=10, observation_dim=2, observation_length=3, action_dim=1
        )

        batch_size = 1
        time_steps = 3

        # Create data with specific dtypes
        observations = np.ones((batch_size, time_steps, 3, 2), dtype=np.float32)
        actions = np.ones((batch_size, time_steps, 1), dtype=np.float32)
        rewards = np.ones((batch_size, time_steps), dtype=np.float32)
        values = np.ones((batch_size, time_steps), dtype=np.float32)
        log_probs = np.ones((batch_size, time_steps), dtype=np.float32)
        terminations = np.zeros((batch_size, time_steps), dtype=bool)
        terminations[0, 2] = True

        buffer.store_batch(
            observations, actions, rewards, values, log_probs, terminations
        )

        data = buffer.get_buffer_data()

        # Check that dtypes are preserved
        assert data["observations"].dtype == np.float32
        assert data["actions"].dtype == np.float32
        assert data["rewards"].dtype == np.float32
        assert data["values"].dtype == np.float32
        assert data["log_probs"].dtype == np.float32
        assert data["terminations"].dtype == bool

    def test_large_batch_processing(self):
        """Test processing a large batch to ensure performance is reasonable."""
        buffer = RolloutBuffer(
            total_buffer_size=10000,
            observation_dim=64,
            observation_length=50,
            action_dim=4,
        )

        # Large batch
        batch_size = 100
        time_steps = 50

        observations = np.random.randn(batch_size, time_steps, 50, 64).astype(
            np.float32
        )
        actions = np.random.randn(batch_size, time_steps, 4).astype(np.float32)
        rewards = np.random.randn(batch_size, time_steps).astype(np.float32)
        values = np.random.randn(batch_size, time_steps).astype(np.float32)
        log_probs = np.random.randn(batch_size, time_steps).astype(np.float32)

        # Random terminations
        terminations = np.zeros((batch_size, time_steps), dtype=bool)
        for i in range(batch_size):
            # Random termination between steps 10-40
            term_step = np.random.randint(10, 40)
            terminations[i, term_step] = True

        # This should complete without error
        buffer.store_batch(
            observations, actions, rewards, values, log_probs, terminations
        )

        # Verify some data was stored
        assert buffer.buffer_size > 0
        assert buffer.buffer_size <= buffer.total_buffer_size

        data = buffer.get_buffer_data()
        assert len(data) == 6  # All expected keys

    def test_is_full_property(self):
        """Test that is_full property works correctly."""
        buffer = RolloutBuffer(
            total_buffer_size=5, observation_dim=2, observation_length=3, action_dim=1
        )

        # Initially buffer should not be full
        assert not buffer.is_full
        assert buffer.buffer_size == 0

        # Manually set buffer size to test property
        buffer.buffer_size = 3
        assert not buffer.is_full

        buffer.buffer_size = 5
        assert buffer.is_full

        # Test with actual data storage
        buffer.reset()
        assert not buffer.is_full

        # Store data that fills the buffer
        batch_size = 1
        time_steps = 10

        observations = np.ones((batch_size, time_steps, 3, 2), dtype=np.float32)
        actions = np.ones((batch_size, time_steps, 1), dtype=np.float32)
        rewards = np.ones((batch_size, time_steps), dtype=np.float32)
        values = np.ones((batch_size, time_steps), dtype=np.float32)
        log_probs = np.ones((batch_size, time_steps), dtype=np.float32)

        # Terminate at step 4 to fill the buffer exactly
        terminations = np.zeros((batch_size, time_steps), dtype=bool)
        terminations[0, 4] = True  # This will store 5 steps (0-4)

        buffer.store_batch(
            observations, actions, rewards, values, log_probs, terminations
        )

        assert buffer.is_full
        assert buffer.buffer_size == 5

    def test_initialization_with_tuple_observation_length(self):
        """Test that RolloutBuffer initializes correctly with tuple observation_length."""
        buffer = RolloutBuffer(
            total_buffer_size=50,
            observation_dim=8,
            observation_length=(4, 4),  # 2D observation space
            action_dim=2,
        )

        assert buffer.total_buffer_size == 50
        assert buffer.observation_dim == 8
        assert buffer.observation_length == (4, 4)
        assert buffer.action_dim == 2
        assert buffer.buffer_size == 0

        # Check buffer shapes - should be (50, 4, 4, 8)
        assert buffer.observation_buffer.shape == (50, 4, 4, 8)
        assert buffer.action_buffer.shape == (50, 2)
        assert buffer.reward_buffer.shape == (50,)
        assert buffer.value_buffer.shape == (50,)
        assert buffer.log_prob_buffer.shape == (50,)
        assert buffer.termination_buffer.shape == (50,)

    def test_initialization_with_list_observation_length(self):
        """Test that RolloutBuffer initializes correctly with list observation_length."""
        buffer = RolloutBuffer(
            total_buffer_size=30,
            observation_dim=4,
            observation_length=[2, 3, 5],  # 3D observation space
            action_dim=3,
        )

        assert buffer.total_buffer_size == 30
        assert buffer.observation_dim == 4
        assert buffer.observation_length == [2, 3, 5]
        assert buffer.action_dim == 3
        assert buffer.buffer_size == 0

        # Check buffer shapes - should be (30, 2, 3, 5, 4)
        assert buffer.observation_buffer.shape == (30, 2, 3, 5, 4)
        assert buffer.action_buffer.shape == (30, 3)
        assert buffer.reward_buffer.shape == (30,)
        assert buffer.value_buffer.shape == (30,)
        assert buffer.log_prob_buffer.shape == (30,)
        assert buffer.termination_buffer.shape == (30,)

    def test_validate_and_reshape_observations_correct_shape(self):
        """Test that observations with correct shape pass through unchanged."""
        buffer = RolloutBuffer(
            total_buffer_size=10,
            observation_dim=4,
            observation_length=5,
            action_dim=2,
        )

        # Create observations with correct shape: (batch_size, time_steps, 5, 4)
        batch_size = 3
        time_steps = 7
        observations = np.random.randn(batch_size, time_steps, 5, 4).astype(np.float32)

        result = buffer._validate_and_reshape_observations(observations)

        # Should return the same array unchanged
        assert np.array_equal(result, observations)
        assert result.shape == (batch_size, time_steps, 5, 4)

    def test_validate_and_reshape_observations_flattened_input(self):
        """Test that flattened observations can be reshaped correctly."""
        buffer = RolloutBuffer(
            total_buffer_size=10,
            observation_dim=4,
            observation_length=5,
            action_dim=2,
        )

        # Create flattened observations: (batch_size, time_steps, 20) instead of (batch_size, time_steps, 5, 4)
        batch_size = 3
        time_steps = 6
        flattened_observations = np.random.randn(batch_size, time_steps, 20).astype(
            np.float32
        )

        result = buffer._validate_and_reshape_observations(flattened_observations)

        # Should be reshaped to (batch_size, time_steps, 5, 4)
        assert result.shape == (batch_size, time_steps, 5, 4)

        # Verify data is preserved (flatten result should match input)
        assert np.array_equal(
            result.reshape(batch_size, time_steps, -1), flattened_observations
        )

    def test_validate_and_reshape_observations_2d_to_3d(self):
        """Test reshaping observations with 2D observation_length when flattened."""
        buffer = RolloutBuffer(
            total_buffer_size=10,
            observation_dim=3,
            observation_length=(4, 4),  # 2D spatial observation
            action_dim=2,
        )

        # Create observations with shape (batch_size, time_steps, 48) -> should become (batch_size, time_steps, 4, 4, 3)
        batch_size = 2
        time_steps = 5
        flattened_observations = np.random.randn(batch_size, time_steps, 48).astype(
            np.float32
        )

        result = buffer._validate_and_reshape_observations(flattened_observations)

        # Should be reshaped to (batch_size, time_steps, 4, 4, 3)
        assert result.shape == (batch_size, time_steps, 4, 4, 3)

        # Verify data is preserved
        assert np.array_equal(
            result.reshape(batch_size, time_steps, -1), flattened_observations
        )

    def test_validate_and_reshape_observations_incompatible_shape(self):
        """Test that incompatible observation shapes raise ValueError."""
        buffer = RolloutBuffer(
            total_buffer_size=10,
            observation_dim=4,
            observation_length=5,
            action_dim=2,
        )

        # Create observations with wrong number of elements: (batch_size, time_steps, 15) instead of 20
        batch_size = 3
        time_steps = 4
        wrong_observations = np.random.randn(batch_size, time_steps, 15).astype(
            np.float32
        )

        try:
            buffer._validate_and_reshape_observations(wrong_observations)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Cannot reshape observations" in str(e)
            assert "15 elements per timestep but expected 20 elements" in str(e)

    def test_validate_and_reshape_observations_complex_shape_mismatch(self):
        """Test error handling with complex shape mismatches."""
        buffer = RolloutBuffer(
            total_buffer_size=10,
            observation_dim=2,
            observation_length=(
                3,
                4,
            ),  # Expected: (batch_size, time_steps, 3, 4, 2) = 24 elements per timestep
            action_dim=1,
        )

        # Create observations with wrong shape that can't be reshaped
        batch_size = 2
        time_steps = 3
        wrong_observations = np.random.randn(batch_size, time_steps, 49).astype(
            np.float32
        )  # 49 elements per timestep ≠ 24

        try:
            buffer._validate_and_reshape_observations(wrong_observations)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Cannot reshape observations" in str(e)

    def test_validate_and_reshape_observations_insufficient_dimensions(self):
        """Test that observations with insufficient dimensions raise ValueError."""
        buffer = RolloutBuffer(
            total_buffer_size=10,
            observation_dim=4,
            observation_length=5,
            action_dim=2,
        )

        # Create observations with only 1 dimension (should have at least 2)
        wrong_observations = np.random.randn(10).astype(np.float32)

        try:
            buffer._validate_and_reshape_observations(wrong_observations)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Observations must have at least 2 dimensions" in str(e)
            assert "but got shape (10,)" in str(e)

    def test_validate_and_reshape_observations_tuple_observation_length(self):
        """Test validation and reshaping with tuple observation_length."""
        buffer = RolloutBuffer(
            total_buffer_size=10,
            observation_dim=3,
            observation_length=(4, 4),  # 2D spatial observation
            action_dim=2,
        )

        # Test with correct shape: (batch_size, time_steps, 4, 4, 3)
        batch_size = 2
        time_steps = 6
        correct_observations = np.random.randn(batch_size, time_steps, 4, 4, 3).astype(
            np.float32
        )

        result = buffer._validate_and_reshape_observations(correct_observations)
        assert np.array_equal(result, correct_observations)
        assert result.shape == (batch_size, time_steps, 4, 4, 3)

        # Test with flattened input: (batch_size, time_steps, 48) -> (batch_size, time_steps, 4, 4, 3)
        flattened_observations = np.random.randn(batch_size, time_steps, 48).astype(
            np.float32
        )
        result = buffer._validate_and_reshape_observations(flattened_observations)
        assert result.shape == (batch_size, time_steps, 4, 4, 3)
        assert np.array_equal(
            result.reshape(batch_size, time_steps, -1), flattened_observations
        )
