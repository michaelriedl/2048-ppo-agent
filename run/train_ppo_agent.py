import logging
import sys
from typing import Optional

sys.path.append("..")  # Ensure parent directory is in sys.path for imports
import hydra
import torch
from omegaconf import DictConfig

from src.ppo import PPOAgent, PPOTrainer, RolloutBuffer
from src.runs import BatchRunner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train_ppo_agent.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main training function for PPO agent.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration

    Returns
    -------
    Optional[float]
        Final mean episode reward
    """
    logger.info("Starting PPO training for 2048")
    logger.info(f"Configuration: {cfg}")

    # Set seed
    seed = cfg.get(
        "seed", cfg.data.seed if "data" in cfg and "seed" in cfg.data else None
    )
    if seed is not None:
        torch.manual_seed(seed)
        logger.info(f"Set random seed to {seed}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create PPO agent
    agent = PPOAgent(
        observation_dim=cfg.model.observation_dim,
        action_dim=cfg.model.action_dim,
        hidden_dim=cfg.model.hidden_dim,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
    )

    logger.info(
        f"Created PPO agent with {sum(p.numel() for p in agent.parameters())} parameters"
    )

    # Create batch runner for environment simulations
    batch_runner = BatchRunner(
        init_seed=seed if seed is not None else 0, act_fn=None  # Will be set by trainer
    )

    # Create rollout buffer
    rollout_buffer = RolloutBuffer(
        total_buffer_size=cfg.trainer.buffer_size,
        observation_dim=cfg.model.observation_dim,
        observation_length=cfg.model.observation_length,
        action_dim=cfg.model.action_dim,
    )

    # Create trainer
    trainer = PPOTrainer(
        agent=agent,
        batch_runner=batch_runner,
        rollout_buffer=rollout_buffer,
        learning_rate=cfg.trainer.learning_rate,
        gamma=cfg.trainer.gamma,
        lambda_gae=cfg.trainer.lambda_gae,
        clip_epsilon=cfg.trainer.clip_epsilon,
        value_loss_coef=cfg.trainer.value_loss_coef,
        entropy_coef=cfg.trainer.entropy_coef,
        max_grad_norm=cfg.trainer.max_grad_norm,
        target_kl=cfg.trainer.target_kl,
        use_action_mask=cfg.trainer.use_action_mask,
        device=device,
    )

    logger.info("Starting training loop")

    # Train the agent
    trainer.train(
        total_timesteps=cfg.trainer.total_timesteps,
        rollout_batch_size=cfg.trainer.rollout_batch_size,
        rollout_batches=cfg.trainer.rollout_batches,
        update_epochs=cfg.trainer.update_epochs,
        train_batch_size=cfg.trainer.train_batch_size,
        save_freq=cfg.trainer.save_freq,
    )

    # Return final performance metric
    if trainer.episode_rewards:
        final_mean_reward = sum(trainer.episode_rewards[-100:]) / min(
            100, len(trainer.episode_rewards)
        )
        logger.info(
            f"Final mean episode reward (last 100 episodes): {final_mean_reward:.2f}"
        )
        return final_mean_reward

    return None


if __name__ == "__main__":
    main()
