import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ..optim import configure_bert_optimizers
from ..runs.batch_runner import BatchRunner
from .data_loader import create_ppo_dataloader
from .ppo_agent import PPOAgent
from .rollout_buffer import RolloutBuffer
from .torch_action_wrapper import TorchActionFunction

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO Trainer for training the 2048 agent.
    """

    def __init__(
        self,
        agent: PPOAgent,
        batch_runner: BatchRunner,
        rollout_buffer: RolloutBuffer,
        optimizer_param_dict: Dict,
        max_steps: int,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        use_action_mask: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize PPO Trainer.

        Parameters
        ----------
        agent : PPOAgent
            The PPO agent to train
        batch_runner : BatchRunner
            Environment batch runner
        rollout_buffer : RolloutBuffer
            Buffer for storing rollout data
        learning_rate : float
            Learning rate for optimizer
        gamma : float
            Discount factor
        lambda_gae : float
            GAE lambda parameter
        clip_epsilon : float
            PPO clipping parameter
        value_loss_coef : float
            Value loss coefficient
        entropy_coef : float
            Entropy coefficient
        max_grad_norm : float
            Maximum gradient norm for clipping
        target_kl : float
            Target KL divergence for early stopping
        use_action_mask : bool
            Whether to use action masking
        device : torch.device
            Device to run on
        output_dir : str
            Output directory for logs and checkpoints
        """
        self.agent = agent.to(device)
        self.batch_runner = batch_runner
        self.rollout_buffer = rollout_buffer

        # Hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.use_action_mask = use_action_mask
        self.device = device

        # Optimizer
        opt_dict = configure_bert_optimizers(
            self.agent, steps=max_steps, **optimizer_param_dict
        )
        self.optimizer = opt_dict["optimizer"]
        self.lr_scheduler = opt_dict["lr_scheduler"]

        # Logging
        self.writer = SummaryWriter("logs")

        # Training metrics
        self.total_timesteps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_save_timestep = 0

        # Track loading of checkpoints
        self.load_checkpoint_path = None

    def collect_rollouts(self, batch_size: int, num_batches: int) -> None:
        """
        Collect rollout data using the current policy.

        Parameters
        ----------
        batch_size : int
            Number of parallel environments
        num_batches : int
            Number of batches to collect
        """
        self.rollout_buffer.reset()

        # Set agent to evaluation mode
        self.agent.eval()

        # Create action function for BatchRunner
        torch_action_fn = TorchActionFunction(
            self.agent, use_mask=self.use_action_mask, device=self.device
        )

        # Set the action function in batch runner
        self.batch_runner.act_fn = torch_action_fn

        total_episodes = 0

        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Run environments to get trajectory data
                (
                    observations,
                    actions,
                    action_masks,
                    log_probs,
                    values,
                    rewards,
                    terminations,
                ) = self.batch_runner.run_actions_batch(batch_size)

                # Convert to torch tensors and move to device if needed
                batch_size_actual, num_steps = observations.shape[:2]

                # Convert actions to proper format for buffer (need to expand dims for action_dim)
                actions_expanded = np.zeros(
                    (batch_size_actual, num_steps, 4), dtype=np.float32
                )
                for i in range(batch_size_actual):
                    for j in range(num_steps):
                        actions_expanded[i, j, actions[i, j]] = 1.0  # One-hot encoding

                # Store in buffer
                self.rollout_buffer.store_batch(
                    observations=observations,
                    actions=actions_expanded,
                    action_masks=action_masks,
                    rewards=rewards,
                    values=values,
                    log_probs=log_probs,
                    terminations=terminations,
                )

                # Count episodes
                total_episodes += batch_size_actual

                # Track episode statistics
                for env_idx in range(batch_size_actual):
                    episode_reward = np.max(rewards[env_idx])
                    episode_length = (
                        np.argmax(terminations[env_idx]) + 1
                        if np.any(terminations[env_idx])
                        else num_steps
                    )
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)

        self.total_timesteps += self.rollout_buffer.buffer_size

        logger.info(
            f"Collected {self.rollout_buffer.buffer_size} timesteps from {total_episodes} episodes"
        )

        # Log episode statistics
        if self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-total_episodes:])
            max_reward = np.max(self.episode_rewards[-total_episodes:])
            mean_length = np.mean(self.episode_lengths[-total_episodes:])

            self.writer.add_scalar(
                "rollout/mean_max_episode_reward", mean_reward, self.total_timesteps
            )
            self.writer.add_scalar(
                "rollout/max_episode_reward", max_reward, self.total_timesteps
            )
            self.writer.add_scalar(
                "rollout/mean_episode_length", mean_length, self.total_timesteps
            )

    def update_policy(
        self,
        batch_size: int = 64,
        n_epochs: int = 4,
    ) -> Dict[str, float]:
        """
        Update the policy using collected rollout data.

        Parameters
        ----------
        batch_size : int
            Batch size for data loader
        n_epochs : int
            Number of epochs to train

        Returns
        -------
        Dict[str, float]
            Training metrics
        """
        if self.rollout_buffer.buffer_size == 0:
            logger.warning("No data in rollout buffer")
            return {}

        # Get buffer data
        buffer_data = self.rollout_buffer.get_buffer_data()

        # Create data loader
        dataloader = create_ppo_dataloader(
            buffer_data=buffer_data,
            gamma=self.gamma,
            lambda_gae=self.lambda_gae,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Set agent to training mode
        self.agent.train()

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        n_updates = 0

        for epoch in range(n_epochs):
            epoch_kl_div = 0.0
            epoch_batches = 0

            for batch in dataloader:
                # Move batch to device
                observations = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)
                action_masks = batch["action_masks"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                # Convert one-hot actions back to action indices
                action_indices = actions.argmax(dim=-1)

                # Get current policy outputs
                new_log_probs, values, entropy = self.agent.evaluate_actions(
                    observations,
                    action_indices,
                    action_mask=action_masks if self.use_action_mask else None,
                )

                # Calculate ratio (pi_theta / pi_theta_old)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Surrogate loss
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2)

                # Value loss
                value_loss = F.mse_loss(values.flatten(), returns, reduction="none")

                # Entropy loss
                entropy_loss = -entropy

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                ).mean()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.max_grad_norm
                )

                self.optimizer.step()

                # Update metrics
                total_policy_loss += policy_loss.mean().item()
                total_value_loss += value_loss.mean().item()
                total_entropy_loss += entropy_loss.mean().item()
                total_loss += loss.item()
                n_updates += 1

                # Calculate KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (old_log_probs - new_log_probs).mean().item()
                    epoch_kl_div += kl_div
                    epoch_batches += 1

            # Check for early stopping
            mean_kl_div = epoch_kl_div / epoch_batches if epoch_batches > 0 else 0
            if mean_kl_div > self.target_kl:
                logger.info(
                    f"Early stopping at epoch {epoch} due to high KL divergence: {mean_kl_div:.6f}"
                )
                break

        # Calculate averages
        metrics = {
            "policy_loss": total_policy_loss / n_updates if n_updates > 0 else 0,
            "value_loss": total_value_loss / n_updates if n_updates > 0 else 0,
            "entropy_loss": total_entropy_loss / n_updates if n_updates > 0 else 0,
            "total_loss": total_loss / n_updates if n_updates > 0 else 0,
            "kl_divergence": mean_kl_div,
            "n_updates": n_updates,
        }

        # Log metrics
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.total_timesteps)

        # Log the distribution of the magnitude of the model parameters
        for name, param in self.agent.named_parameters():
            self.writer.add_histogram(
                f"train/param_magnitude/{name}", param.data, self.total_timesteps
            )

        return metrics

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Parameters
        ----------
        filename : str
            Filename for the checkpoint
        """
        checkpoint = {
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_timesteps": self.total_timesteps,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "last_save_timestep": self.last_save_timestep,
        }

        torch.save(checkpoint, filename)
        logger.info(
            f"Checkpoint saved to {filename} (timesteps: {self.total_timesteps})"
        )

    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.

        Parameters
        ----------
        filename : str
            Filename of the checkpoint
        """
        # Store the checkpoint path
        self.load_checkpoint_path = filename
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            logger.info(f"Loading checkpoint from {filename}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {filename}: {e}")
            raise

        # Validate checkpoint contents
        required_keys = ["agent_state_dict", "optimizer_state_dict"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

        # Load agent state
        try:
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            logger.info("Successfully loaded agent state")
        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")
            raise

        # Load optimizer state
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Successfully loaded optimizer state")
        except Exception as e:
            logger.warning(
                f"Failed to load optimizer state: {e}. Training will continue with fresh optimizer state."
            )

        # Load training statistics
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.episode_lengths = checkpoint.get("episode_lengths", [])
        self.last_save_timestep = checkpoint.get("last_save_timestep", 0)

        logger.info(f"Checkpoint loaded successfully:")
        logger.info(f"  - Total timesteps: {self.total_timesteps}")
        logger.info(
            f"  - Episode rewards history: {len(self.episode_rewards)} episodes"
        )
        logger.info(
            f"  - Episode lengths history: {len(self.episode_lengths)} episodes"
        )

        if self.episode_rewards:
            recent_reward = np.mean(
                self.episode_rewards[-min(100, len(self.episode_rewards)) :]
            )
            logger.info(f"  - Recent mean reward: {recent_reward:.2f}")

    def train(
        self,
        total_timesteps: int,
        rollout_batch_size: int = 32,
        rollout_batches: int = 4,
        update_epochs: int = 4,
        train_batch_size: int = 64,
        save_freq: int = 10000,
        resume_extend_steps: bool = True,
    ) -> None:
        """
        Main training loop.

        Parameters
        ----------
        total_timesteps : int
            Total timesteps to train for
        rollout_batch_size : int
            Batch size for environment rollouts
        rollout_batches : int
            Number of rollout batches to collect per iteration
        update_epochs : int
            Number of epochs per policy update
        train_batch_size : int
            Minibatch size for policy updates
        save_freq : int
            Frequency to save checkpoints
        resume_extend_steps : bool
            If True, extend training by total_timesteps from current state.
            If False, train until total_timesteps from start.
        """
        # Calculate target timesteps based on resume mode
        starting_timesteps = self.total_timesteps
        if self.load_checkpoint_path is not None:
            starting_text = "Resuming"
        else:
            starting_text = "Starting"
        if resume_extend_steps:
            target_timesteps = starting_timesteps + total_timesteps
            logger.info(
                f"{starting_text} training from {starting_timesteps} timesteps, extending by {total_timesteps} to reach {target_timesteps}"
            )
        else:
            target_timesteps = total_timesteps
            logger.info(
                f"{starting_text} training from {starting_timesteps} timesteps to reach {target_timesteps}"
            )

        if starting_timesteps >= target_timesteps:
            logger.warning(
                f"Already trained for {starting_timesteps} timesteps, target is {target_timesteps}. No training needed."
            )
            return

        logger.info(
            f"Starting PPO training for {target_timesteps - starting_timesteps} timesteps"
        )

        iteration = 0

        while self.total_timesteps < target_timesteps:
            iteration += 1

            # Collect rollouts
            self.collect_rollouts(rollout_batch_size, rollout_batches)

            # Update policy
            metrics = self.update_policy(
                batch_size=train_batch_size,
                n_epochs=update_epochs,
            )

            # Log metrics
            logger.info(
                f"Iteration {iteration}, Timesteps: {self.total_timesteps}/{target_timesteps}"
            )
            if metrics:
                logger.info(
                    f"Policy Loss: {metrics['policy_loss']:.4f}, "
                    f"Value Loss: {metrics['value_loss']:.4f}, "
                    f"Entropy: {metrics['entropy_loss']:.4f}"
                )

            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-100:]
                logger.info(
                    f"Mean Episode Reward (last 100): {np.mean(recent_rewards):.2f}"
                )

            # Save checkpoint
            if self.total_timesteps - self.last_save_timestep >= save_freq:
                self.save_checkpoint(f"checkpoint_{iteration}.pt")
                self.last_save_timestep = self.total_timesteps

        logger.info("Training completed!")
        self.save_checkpoint("final_model.pt")
        self.writer.close()
