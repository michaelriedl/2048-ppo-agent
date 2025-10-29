import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.ppo.data_loader import PPODataset, create_ppo_dataloader
from src.ppo.rollout_buffer import RolloutBuffer


class TestPPODataset:
    """Test class for PPODataset functionality."""

    @pytest.fixture
    def sample_buffer_data(self):
        """Create sample buffer data for testing."""
        buffer_size = 100
        obs_length = 20
        obs_dim = 16
        action_dim = 4

        np.random.seed(42)

        return {
            "observations": np.random.randn(buffer_size, obs_length, obs_dim).astype(
                np.float32
            ),
            "actions": np.random.randn(buffer_size, action_dim).astype(np.float32),
            "action_masks": np.random.choice(
                [True, False], (buffer_size, action_dim), p=[0.8, 0.2]
            ),
            "rewards": np.random.randn(buffer_size).astype(np.float32),
            "values": np.random.randn(buffer_size).astype(np.float32),
            "log_probs": np.random.randn(buffer_size).astype(np.float32),
            "terminations": np.random.choice([True, False], buffer_size, p=[0.1, 0.9]),
            "advantages": np.random.randn(buffer_size).astype(
                np.float32
            ),  # Will be overwritten
            "returns": np.random.randn(buffer_size).astype(
                np.float32
            ),  # Will be overwritten
        }

    def test_initialization(self, sample_buffer_data):
        """Test that PPODataset initializes correctly."""
        dataset = PPODataset(
            buffer_data=sample_buffer_data, gamma=0.99, lambda_gae=0.95
        )

        assert dataset.gamma == 0.99
        assert dataset.lambda_gae == 0.95
        assert len(dataset) == 100

        # Check that tensors are created correctly
        assert isinstance(dataset.observations, torch.Tensor)
        assert isinstance(dataset.actions, torch.Tensor)
        assert isinstance(dataset.action_masks, torch.Tensor)
        assert isinstance(dataset.rewards, torch.Tensor)
        assert isinstance(dataset.values, torch.Tensor)
        assert isinstance(dataset.log_probs, torch.Tensor)
        assert isinstance(dataset.terminations, torch.Tensor)

        # Check tensor shapes
        assert dataset.observations.shape == (100, 20, 16)
        assert dataset.actions.shape == (100, 4)
        assert dataset.action_masks.shape == (100, 4)
        assert dataset.rewards.shape == (100,)
        assert dataset.values.shape == (100,)
        assert dataset.log_probs.shape == (100,)
        assert dataset.terminations.shape == (100,)

    def test_gae_computation(self, sample_buffer_data):
        """Test GAE computation functionality."""
        dataset = PPODataset(
            buffer_data=sample_buffer_data, gamma=0.99, lambda_gae=0.95
        )

        # Check that advantages and returns are computed
        assert hasattr(dataset, "advantages")
        assert hasattr(dataset, "returns")
        assert isinstance(dataset.advantages, torch.Tensor)
        assert isinstance(dataset.returns, torch.Tensor)
        assert dataset.advantages.shape == (100,)
        assert dataset.returns.shape == (100,)

    def test_getitem(self, sample_buffer_data):
        """Test dataset item retrieval."""
        dataset = PPODataset(
            buffer_data=sample_buffer_data, gamma=0.99, lambda_gae=0.95
        )

        item = dataset[0]

        # Check that item is a dictionary with correct keys
        expected_keys = {
            "observations",
            "actions",
            "action_masks",
            "rewards",
            "values",
            "log_probs",
            "terminations",
            "advantages",
            "returns",
        }
        assert set(item.keys()) == expected_keys

        # Check that all values are tensors
        for key, value in item.items():
            assert isinstance(value, torch.Tensor)

        # Check shapes for individual items
        assert item["observations"].shape == (20, 16)
        assert item["actions"].shape == (4,)
        assert item["action_masks"].shape == (4,)
        assert item["rewards"].shape == ()
        assert item["values"].shape == ()
        assert item["log_probs"].shape == ()
        assert item["terminations"].shape == ()
        assert item["advantages"].shape == ()
        assert item["returns"].shape == ()

    def test_different_gamma_lambda(self, sample_buffer_data):
        """Test dataset with different gamma and lambda values."""
        dataset1 = PPODataset(
            buffer_data=sample_buffer_data, gamma=0.95, lambda_gae=0.9
        )

        dataset2 = PPODataset(
            buffer_data=sample_buffer_data, gamma=0.99, lambda_gae=0.95
        )

        # GAE computation should give different results with different parameters
        assert not torch.allclose(dataset1.advantages, dataset2.advantages)
        assert not torch.allclose(dataset1.returns, dataset2.returns)

    def test_action_masks_handling(self, sample_buffer_data):
        """Test that action masks are properly handled."""
        dataset = PPODataset(
            buffer_data=sample_buffer_data, gamma=0.99, lambda_gae=0.95
        )

        # Check that action masks are boolean tensors
        assert dataset.action_masks.dtype == torch.bool
        assert dataset.action_masks.shape == (100, 4)

        # Check that action masks in individual items are boolean
        item = dataset[0]
        assert item["action_masks"].dtype == torch.bool
        assert item["action_masks"].shape == (4,)

        # Check that action masks contain both True and False values
        # (with our random generation, this should be the case)
        assert torch.any(dataset.action_masks)  # At least one True
        assert not torch.all(dataset.action_masks)  # At least one False


class TestFactoryFunction:
    """Test class for the create_ppo_dataloader factory function."""

    @pytest.fixture
    def sample_buffer_data(self):
        """Create sample buffer data using RolloutBuffer."""
        buffer = RolloutBuffer(
            observation_dim=16,
            observation_length=20,
            action_dim=4,
        )

        # Add some sample data
        np.random.seed(42)
        batch_size = 5
        time_steps = 20

        observations = np.random.randn(batch_size, time_steps, 20, 16).astype(
            np.float32
        )
        actions = np.random.randn(batch_size, time_steps, 4).astype(np.float32)
        action_masks = (
            np.random.rand(batch_size, time_steps, 4) > 0.3
        )  # Random action masks
        rewards = np.random.randn(batch_size, time_steps).astype(np.float32)
        values = np.random.randn(batch_size, time_steps).astype(np.float32)
        log_probs = np.random.randn(batch_size, time_steps).astype(np.float32)

        # Create terminations with some True values
        terminations = np.zeros((batch_size, time_steps), dtype=bool)
        for i in range(batch_size):
            # Set termination at random step (ensuring at least one termination per trajectory)
            term_step = np.random.randint(5, time_steps)
            terminations[i, term_step] = True

        buffer.store_batch(
            observations=observations,
            actions=actions,
            action_masks=action_masks,
            rewards=rewards,
            values=values,
            log_probs=log_probs,
            terminations=terminations,
        )

        return buffer.get_buffer_data()

    def test_create_ppo_dataloader(self, sample_buffer_data):
        """Test the factory function creates a proper dataloader."""
        dataloader = create_ppo_dataloader(
            buffer_data=sample_buffer_data,
            gamma=0.95,
            lambda_gae=0.9,
            batch_size=16,
            shuffle=True,
            num_workers=0,
        )

        assert isinstance(dataloader, DataLoader)
        assert isinstance(dataloader.dataset, PPODataset)
        assert dataloader.dataset.gamma == 0.95
        assert dataloader.dataset.lambda_gae == 0.9
        assert dataloader.batch_size == 16

        # Test that we can iterate over the dataloader
        for batch in dataloader:
            assert isinstance(batch, dict)
            break  # Just check first batch

    def test_dataloader_iteration(self, sample_buffer_data):
        """Test that the created dataloader can be iterated over."""
        dataloader = create_ppo_dataloader(
            buffer_data=sample_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            batch_size=32,
            shuffle=False,  # Don't shuffle for predictable testing
            num_workers=0,
        )

        batches = list(dataloader)

        # Check that we have batches
        assert len(batches) > 0

        # Check batch shapes (at least for the first batch)
        if len(batches) > 0:
            batch = batches[0]
            batch_size = batch["observations"].shape[0]

            assert batch["observations"].shape[1:] == (
                20,
                16,
            )  # observation_length, observation_dim
            assert batch["actions"].shape == (batch_size, 4)  # action dim
            assert batch["action_masks"].shape == (batch_size, 4)  # action masks
            assert batch["rewards"].shape == (batch_size,)
            assert batch["values"].shape == (batch_size,)
            assert batch["log_probs"].shape == (batch_size,)
            assert batch["terminations"].shape == (batch_size,)
            assert batch["advantages"].shape == (batch_size,)
            assert batch["returns"].shape == (batch_size,)

    def test_integration_with_rollout_buffer(self, sample_buffer_data):
        """Test integration between RolloutBuffer and DataLoader."""
        # Create dataloader from buffer data
        dataloader = create_ppo_dataloader(
            buffer_data=sample_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            batch_size=32,
            shuffle=False,
            drop_last=False,
        )

        # Check that we have valid data
        assert len(dataloader.dataset) > 0

        # Test iteration
        total_samples = 0
        for batch in dataloader:
            batch_size = batch["observations"].shape[0]
            total_samples += batch_size

            # Check that all tensors have the same batch size
            for key, tensor in batch.items():
                assert tensor.shape[0] == batch_size

        # Total samples should match buffer size
        assert total_samples == len(dataloader.dataset)


class TestDatasetLimiting:
    """Test class for dataset limiting and subset selection functionality."""

    @pytest.fixture
    def large_buffer_data(self):
        """Create a large buffer dataset for testing limiting functionality."""
        buffer_size = 200
        obs_length = 20
        obs_dim = 16
        action_dim = 4

        np.random.seed(42)

        return {
            "observations": np.random.randn(buffer_size, obs_length, obs_dim).astype(
                np.float32
            ),
            "actions": np.random.randn(buffer_size, action_dim).astype(np.float32),
            "action_masks": np.random.choice(
                [True, False], (buffer_size, action_dim), p=[0.8, 0.2]
            ),
            "rewards": np.random.randn(buffer_size).astype(np.float32),
            "values": np.random.randn(buffer_size).astype(np.float32),
            "log_probs": np.random.randn(buffer_size).astype(np.float32),
            "terminations": np.random.choice([True, False], buffer_size, p=[0.1, 0.9]),
        }

    def test_no_limit_uses_all_data(self, large_buffer_data):
        """Test that without a limit, all data is used."""
        dataset = PPODataset(buffer_data=large_buffer_data, gamma=0.99, lambda_gae=0.95)

        assert len(dataset) == 200
        assert dataset.total_length == 200
        assert dataset.active_indices is None

    def test_limit_smaller_than_dataset(self, large_buffer_data):
        """Test that max_samples_per_epoch limits the dataset length."""
        dataset = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=50,
        )

        assert len(dataset) == 50
        assert dataset.total_length == 200
        assert dataset.active_indices is not None
        assert len(dataset.active_indices) == 50

    def test_limit_larger_than_dataset(self, large_buffer_data):
        """Test that a limit larger than dataset size uses all data."""
        dataset = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=300,
        )

        assert len(dataset) == 200
        assert dataset.total_length == 200
        assert dataset.active_indices is None

    def test_limit_equal_to_dataset(self, large_buffer_data):
        """Test that a limit equal to dataset size uses all data."""
        dataset = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=200,
        )

        assert len(dataset) == 200
        assert dataset.total_length == 200
        assert dataset.active_indices is None

    def test_getitem_with_limit(self, large_buffer_data):
        """Test that __getitem__ works correctly with a limit."""
        dataset = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=50,
        )

        # Access all items in the limited dataset
        for idx in range(len(dataset)):
            item = dataset[idx]
            assert isinstance(item, dict)
            assert "observations" in item
            assert item["observations"].shape == (20, 16)

    def test_subset_selection_is_random(self, large_buffer_data):
        """Test that different datasets get different random subsets."""
        dataset1 = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=50,
        )

        dataset2 = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=50,
        )

        # The two datasets should likely have different subsets
        # (extremely unlikely to be the same with random selection)
        assert not torch.equal(dataset1.active_indices, dataset2.active_indices)

    def test_shuffle_on_reset_disabled(self, large_buffer_data):
        """Test that subset stays the same when shuffle_on_reset is False."""
        dataset = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=50,
            shuffle_on_reset=False,
        )

        # Store the initial indices
        initial_indices = dataset.active_indices.clone()

        # Reset the epoch
        dataset.reset_epoch()

        # Indices should remain the same
        assert torch.equal(dataset.active_indices, initial_indices)

    def test_shuffle_on_reset_enabled(self, large_buffer_data):
        """Test that subset changes when shuffle_on_reset is True."""
        dataset = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=50,
            shuffle_on_reset=True,
        )

        # Store the initial indices
        initial_indices = dataset.active_indices.clone()

        # Reset the epoch
        dataset.reset_epoch()

        # Indices should be different (extremely unlikely to be the same)
        assert not torch.equal(dataset.active_indices, initial_indices)

    def test_reset_epoch_with_no_limit(self, large_buffer_data):
        """Test that reset_epoch does nothing when there's no limit."""
        dataset = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            shuffle_on_reset=True,  # Even with shuffle enabled
        )

        # Should have no active_indices
        assert dataset.active_indices is None

        # Reset should do nothing
        dataset.reset_epoch()

        # Still no active_indices
        assert dataset.active_indices is None

    def test_dataloader_with_limit(self, large_buffer_data):
        """Test that DataLoader works correctly with limited dataset."""
        dataloader = create_ppo_dataloader(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            batch_size=16,
            shuffle=True,
            max_samples_per_epoch=50,
        )

        # Dataset should be limited
        assert len(dataloader.dataset) == 50

        # Should be able to iterate
        batches = list(dataloader)
        assert len(batches) > 0

        # Total samples should be 50 (or close if drop_last=True)
        total_samples = sum(batch["observations"].shape[0] for batch in batches)
        assert total_samples <= 50

    def test_dataloader_with_limit_and_shuffle_on_reset(self, large_buffer_data):
        """Test DataLoader with both limiting and shuffle_on_reset."""
        dataloader = create_ppo_dataloader(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            batch_size=16,
            shuffle=False,  # DataLoader shuffle off to see dataset subset changes
            max_samples_per_epoch=50,
            shuffle_on_reset=True,
        )

        # Get first batch from first epoch
        first_epoch_first_batch = next(iter(dataloader))

        # Reset the dataset
        dataloader.dataset.reset_epoch()

        # Get first batch from second epoch
        second_epoch_first_batch = next(iter(dataloader))

        # The batches should likely be different due to subset resampling
        # Note: This test has a small probability of false positives
        assert not torch.equal(
            first_epoch_first_batch["observations"],
            second_epoch_first_batch["observations"],
        )

    def test_all_data_accessible_over_multiple_epochs(self, large_buffer_data):
        """Test that with shuffle_on_reset, different data is accessed each epoch."""
        dataset = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=50,
            shuffle_on_reset=True,
        )

        # Collect indices from multiple epochs
        all_indices = set()
        for _ in range(10):  # 10 epochs with 50 samples each
            all_indices.update(dataset.active_indices.tolist())
            dataset.reset_epoch()

        # We should have seen a good portion of the dataset
        # (likely more than just 50 unique indices)
        assert len(all_indices) > 50

    def test_indices_are_within_bounds(self, large_buffer_data):
        """Test that sampled indices are always within valid range."""
        dataset = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=50,
            shuffle_on_reset=True,
        )

        # Check initial indices
        assert torch.all(dataset.active_indices >= 0)
        assert torch.all(dataset.active_indices < 200)

        # Check after multiple resets
        for _ in range(5):
            dataset.reset_epoch()
            assert torch.all(dataset.active_indices >= 0)
            assert torch.all(dataset.active_indices < 200)

    def test_indices_are_unique(self, large_buffer_data):
        """Test that sampled indices don't contain duplicates."""
        dataset = PPODataset(
            buffer_data=large_buffer_data,
            gamma=0.99,
            lambda_gae=0.95,
            max_samples_per_epoch=50,
        )

        # Check that all indices are unique
        unique_indices = torch.unique(dataset.active_indices)
        assert len(unique_indices) == len(dataset.active_indices)

        # Check after reset
        dataset.shuffle_on_reset = True
        dataset.reset_epoch()
        unique_indices = torch.unique(dataset.active_indices)
        assert len(unique_indices) == len(dataset.active_indices)
