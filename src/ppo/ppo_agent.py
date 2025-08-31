from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .transformer_encoder import TransformerEncoder


class PPOAgent(nn.Module):
    """
    PPO Agent with actor-critic architecture using transformer encoder.
    """

    def __init__(
        self,
        observation_dim: int = 31,
        action_dim: int = 4,  # up, down, left, right
        hidden_dim: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Initialize PPO Agent.

        Parameters
        ----------
        observation_dim : int
            Dimension of observation space (31 for 2048 game)
        action_dim : int
            Dimension of action space (4 for 2048 game)
        hidden_dim : int
            Hidden dimension for linear layers
        d_model : int
            Model dimension for transformer
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer layers
        dim_feedforward : int
            Feedforward dimension in transformer
        dropout : float
            Dropout rate
        """
        super(PPOAgent, self).__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input embedding layer
        self.input_embedding = nn.Linear(observation_dim, d_model)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, observations: torch.Tensor, action_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Parameters
        ----------
        observations : torch.Tensor
            Batch of observations, shape (batch_size, board_size, observation_dim)
        action_mask : torch.Tensor, optional
            Legal action mask, shape (batch_size, action_dim)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            action_logits, values
        """
        # Embed observations
        embedded = self.input_embedding(observations)

        # Pass through transformer
        transformer_out = self.transformer(embedded)

        # Reduce the sequence to a single embedding
        features = transformer_out.mean(dim=1)

        # Actor output (action logits)
        action_logits = self.actor(features)

        # Apply action mask if provided
        if action_mask is not None:
            # Set masked actions to very negative value
            action_logits = action_logits - (1e8 * (1 - action_mask))

        # Critic output (state values)
        values = self.critic(features)

        return action_logits, values

    def get_action(
        self, observations: torch.Tensor, action_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.

        Parameters
        ----------
        observations : torch.Tensor
            Batch of observations
        action_mask : torch.Tensor, optional
            Legal action mask

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            actions, log_probs, values
        """
        action_logits, values = self.forward(observations, action_mask)

        # Create categorical distribution
        dist = Categorical(logits=action_logits)

        # Sample actions
        actions = dist.sample()

        # Get log probabilities
        log_probs = dist.log_prob(actions)

        return actions, log_probs, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.

        Parameters
        ----------
        observations : torch.Tensor
            Batch of observations
        actions : torch.Tensor
            Batch of actions to evaluate
        action_mask : torch.Tensor, optional
            Legal action mask

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            log_probs, values, entropy
        """
        action_logits, values = self.forward(observations, action_mask)

        # Create categorical distribution
        dist = Categorical(logits=action_logits)

        # Get log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy
