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
        obs_dim = 16
        action_dim = 4

        np.random.seed(42)

        return {
            "observations": np.random.randn(buffer_size, obs_dim).astype(np.float32),
            "actions": np.random.randn(buffer_size, action_dim).astype(np.float32),
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
        assert isinstance(dataset.rewards, torch.Tensor)
        assert isinstance(dataset.values, torch.Tensor)
        assert isinstance(dataset.log_probs, torch.Tensor)
        assert isinstance(dataset.terminations, torch.Tensor)

        # Check tensor shapes
        assert dataset.observations.shape == (100, 16)
        assert dataset.actions.shape == (100, 4)
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
        assert item["observations"].shape == (16,)
        assert item["actions"].shape == (4,)
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


class TestFactoryFunction:
    """Test class for the create_ppo_dataloader factory function."""

    @pytest.fixture
    def sample_buffer_data(self):
        """Create sample buffer data using RolloutBuffer."""
        buffer = RolloutBuffer(total_buffer_size=100, observation_dim=16, action_dim=4)

        # Add some sample data
        np.random.seed(42)
        batch_size = 5
        time_steps = 20

        observations = np.random.randn(batch_size, time_steps, 16).astype(np.float32)
        actions = np.random.randn(batch_size, time_steps, 4).astype(np.float32)
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

            assert batch["observations"].shape[1] == 16  # observation dim
            assert batch["actions"].shape == (batch_size, 4)  # action dim
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
