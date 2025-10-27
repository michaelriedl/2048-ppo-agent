from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PPODataset(Dataset):
    """
    PyTorch Dataset for PPO training data.

    This dataset takes the output from a RolloutBuffer and prepares it for PPO training,
    including computation of advantages and returns using GAE (Generalized Advantage Estimation).
    """

    def __init__(
        self,
        buffer_data: Dict[str, np.ndarray],
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        max_samples_per_epoch: int = None,
        shuffle_on_reset: bool = False,
    ):
        """
        Initialize the PPO dataset.

        Parameters
        ----------
        buffer_data : Dict[str, np.ndarray]
            Dictionary containing rollout buffer data with keys:
            - observations, actions, rewards, values, log_probs, terminations
        gamma : float, default=0.99
            Discount factor for returns computation
        lambda_gae : float, default=0.95
            Lambda parameter for GAE computation
        max_samples_per_epoch : int, optional
            Maximum number of samples to use per epoch. If None, all samples are used.
            If less than the total dataset size, only a subset is used each epoch.
        shuffle_on_reset : bool, default=False
            If True and max_samples_per_epoch is set, select a different random subset
            each time the dataset is exhausted. If False, use the same subset each epoch.
        """
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.max_samples_per_epoch = max_samples_per_epoch
        self.shuffle_on_reset = shuffle_on_reset

        # Convert numpy arrays to torch tensors
        self.observations = torch.from_numpy(buffer_data["observations"]).float()
        self.actions = torch.from_numpy(buffer_data["actions"]).float()
        self.action_masks = torch.from_numpy(buffer_data["action_masks"]).bool()
        self.rewards = torch.from_numpy(buffer_data["rewards"]).float()
        self.values = torch.from_numpy(buffer_data["values"]).float()
        self.log_probs = torch.from_numpy(buffer_data["log_probs"]).float()
        self.terminations = torch.from_numpy(buffer_data["terminations"]).bool()

        # Compute advantages and returns
        self.advantages, self.returns = self._compute_gae_returns()

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )
        # Normalize returns
        self.returns = (self.returns - self.returns.mean()) / (
            self.returns.std() + 1e-8
        )

        # Store the total dataset size
        self.total_length = len(self.observations)

        # Set the effective length (limited by max_samples_per_epoch if specified)
        if (
            self.max_samples_per_epoch is None
            or self.max_samples_per_epoch >= self.total_length
        ):
            self.length = self.total_length
            self.active_indices = None  # Use all data
        else:
            self.length = self.max_samples_per_epoch
            self.active_indices = self._sample_indices()

    def _sample_indices(self) -> torch.Tensor:
        """
        Sample random indices for the current epoch subset.

        Returns
        -------
        torch.Tensor
            Indices to use for the current epoch
        """
        indices = torch.randperm(self.total_length)[: self.length]
        return indices

    def reset_epoch(self):
        """
        Reset the dataset for a new epoch. If shuffle_on_reset is True,
        select a new random subset of data.
        """
        if self.shuffle_on_reset and self.active_indices is not None:
            self.active_indices = self._sample_indices()

    def _compute_gae_returns(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) and returns.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            advantages, returns
        """
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)

        # Work backwards through the trajectory to compute advantages and returns
        last_gae = 0
        last_value = 0
        for step in reversed(range(len(self.rewards))):
            if self.terminations[step]:
                last_value = 0
                last_gae = 0
            # Update the GAE
            delta = self.rewards[step] + self.gamma * last_value - self.values[step]
            last_gae = delta + self.gamma * self.lambda_gae * last_gae

            advantages[step] = last_gae
            returns[step] = last_gae + self.values[step]
            last_value = self.values[step]

        return advantages, returns

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve (0 to len(self)-1)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing training data for one step
        """
        # Map the requested index to the actual data index
        if self.active_indices is not None:
            actual_idx = self.active_indices[idx]
        else:
            actual_idx = idx

        return {
            "observations": self.observations[actual_idx],
            "actions": self.actions[actual_idx],
            "action_masks": self.action_masks[actual_idx],
            "rewards": self.rewards[actual_idx],
            "values": self.values[actual_idx],
            "log_probs": self.log_probs[actual_idx],
            "terminations": self.terminations[actual_idx],
            "advantages": self.advantages[actual_idx],
            "returns": self.returns[actual_idx],
        }


def create_ppo_dataloader(
    buffer_data: Dict[str, np.ndarray],
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    max_samples_per_epoch: int = None,
    shuffle_on_reset: bool = False,
) -> DataLoader:
    """
    Factory function to create a PyTorch DataLoader for PPO training from buffer data.

    Parameters
    ----------
    buffer_data : Dict[str, np.ndarray]
        Dictionary containing rollout buffer data
    gamma : float, default=0.99
        Discount factor for returns computation
    lambda_gae : float, default=0.95
        Lambda parameter for GAE computation
    batch_size : int, default=32
        Size of batches to create
    shuffle : bool, default=True
        Whether to shuffle the data
    drop_last : bool, default=True
        Whether to drop the last incomplete batch
    num_workers : int, default=0
        Number of worker processes for loading
    max_samples_per_epoch : int, optional
        Maximum number of samples to use per epoch. If None, all samples are used.
    shuffle_on_reset : bool, default=False
        If True and max_samples_per_epoch is set, select a different random subset
        each time the dataset is exhausted.

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader with PPO dataset
    """
    dataset = PPODataset(
        buffer_data,
        gamma=gamma,
        lambda_gae=lambda_gae,
        max_samples_per_epoch=max_samples_per_epoch,
        shuffle_on_reset=shuffle_on_reset,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
