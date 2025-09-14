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
        return_norm_scale: float = 120,
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
        return_norm_scale : float, default=20
            Scale factor for return normalization
        """
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.return_norm_scale = return_norm_scale

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
        self.returns = self.returns / self.return_norm_scale

        self.length = len(self.observations)

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
            Index of the item to retrieve

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing training data for one step
        """
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "action_masks": self.action_masks[idx],
            "rewards": self.rewards[idx],
            "values": self.values[idx],
            "log_probs": self.log_probs[idx],
            "terminations": self.terminations[idx],
            "advantages": self.advantages[idx],
            "returns": self.returns[idx],
        }


def create_ppo_dataloader(
    buffer_data: Dict[str, np.ndarray],
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
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

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader with PPO dataset
    """
    dataset = PPODataset(buffer_data, gamma=gamma, lambda_gae=lambda_gae)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
