#!/usr/bin/env python3
"""
Visualization script for trained PPO agent playing 2048.

This script loads a trained PPO model and visualizes gameplay rollouts,
creating SVG animations of the agent playing the game.
"""

import logging
import os
import sys
from pathlib import Path

import jax
import pgx
import torch
from IPython.display import SVG, display

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.env_definitions import ACTION_DIM, OBS_DIM
from src.ppo import PPOAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_agent(checkpoint_path: str, device: torch.device) -> PPOAgent:
    """
    Load a trained PPO agent from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint
    device : torch.device
        Device to load the model on

    Returns
    -------
    PPOAgent
        Loaded PPO agent
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Create agent with same configuration as training
    # These parameters should match the config used during training
    agent = PPOAgent(
        observation_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=512,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    ).to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"])

    # Set to evaluation mode
    agent.eval()

    logger.info("Model loaded successfully")
    return agent


def visualize_rollouts(
    agent: PPOAgent,
    device: torch.device,
    num_rollouts: int = 4,
    output_dir: str = "visualizations",
    seed: int = 42,
):
    """
    Create and save SVG visualizations of the agent playing.

    Parameters
    ----------
    agent : PPOAgent
        Trained PPO agent
    device : torch.device
        Device for inference
    num_rollouts : int
        Number of rollouts to visualize
    output_dir : str
        Directory to save visualizations
    seed : int
        Random seed for reproducible visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating {num_rollouts} rollout visualizations...")

    for rollout_idx in range(num_rollouts):
        logger.info(f"Creating visualization {rollout_idx + 1}/{num_rollouts}")

        # Create a custom simulation loop for this rollout
        # This avoids the JAX/PyTorch interaction issues
        rollout_seed = seed + rollout_idx * 1000

        # Initialize environment
        key = jax.random.key(seed=rollout_seed)
        env = pgx.make("2048")

        # Initialize single environment state
        key, subkey = jax.random.split(key)
        state = env.init(subkey)

        states = []
        step_count = 0
        max_steps = 1000  # Safety limit

        while not (state.terminated or state.truncated) and step_count < max_steps:
            states.append(state)

            # Get observation and legal action mask
            obs = state.observation  # Shape: (4, 4, 31)
            mask = state.legal_action_mask  # Shape: (4,)

            # Convert to PyTorch and get action from agent
            obs_reshaped = obs.reshape(
                16, 31
            )  # Shape: (16, 31) - board_size x observation_dim
            obs_np = jax.device_get(obs_reshaped)
            mask_np = jax.device_get(mask)

            # Make a copy to ensure it's writable
            obs_np = obs_np.copy()
            mask_np = mask_np.copy()

            # Convert to torch tensors
            obs_tensor = (
                torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
            )  # Shape: (1, 16, 31)
            mask_tensor = (
                torch.from_numpy(mask_np).float().unsqueeze(0).to(device)
            )  # Shape: (1, 4)

            # Get action from agent
            with torch.no_grad():
                action_logits, _ = agent.forward(obs_tensor, mask_tensor)

                # Apply legal action mask
                masked_logits = action_logits.clone()
                masked_logits[0, mask_tensor[0] == 0] = float("-inf")

                # Get the action with highest probability (greedy)
                action_idx = torch.argmax(masked_logits, dim=-1)[0].cpu().numpy()

            # Take step in environment
            key, subkey = jax.random.split(key)
            state = env.step(state, int(action_idx), subkey)
            step_count += 1

        # Add final state
        if states:
            states.append(state)

            # Create SVG animation
            output_path = os.path.join(
                output_dir, f"ppo_agent_rollout_{rollout_idx + 1}.svg"
            )
            pgx.save_svg_animation(
                states,
                output_path,
                frame_duration_seconds=0.8,
            )

            # Calculate some statistics about this rollout
            final_state = state
            # Extract board from final observation
            final_obs = final_state.observation  # Shape: (4, 4, 31)
            # Convert from one-hot to board values
            board_values = final_obs.argmax(axis=-1)  # Shape: (4, 4)
            max_tile = (2**board_values).max()
            num_moves = len(states) - 1  # Subtract initial state

            logger.info(
                f"  Rollout {rollout_idx + 1}: {num_moves} moves, max tile: {max_tile}"
            )
            logger.info(f"  Saved visualization to: {output_path}")
        else:
            logger.warning(f"  Rollout {rollout_idx + 1}: No valid states collected")

    logger.info(f"All visualizations saved to: {output_dir}/")


def find_latest_checkpoint(results_dir: str = "results") -> str:
    """
    Find the most recent checkpoint file in the results directory.

    Parameters
    ----------
    results_dir : str
        Results directory to search

    Returns
    -------
    str
        Path to the latest checkpoint
    """
    # Look for checkpoint files in the results directory
    results_path = Path(results_dir)

    # Find all .pt files
    checkpoint_files = list(results_path.glob("**/*.pt"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files (.pt) found in {results_dir}")

    # Sort by modification time and get the latest
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)

    return str(latest_checkpoint)


def main():
    """Main visualization function."""
    logger.info("Starting PPO Agent Visualization")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Find the latest trained model
    try:
        checkpoint_path = find_latest_checkpoint()
        logger.info(f"Found latest checkpoint: {checkpoint_path}")
    except FileNotFoundError as e:
        logger.error(f"Error finding checkpoint: {e}")
        logger.error(
            "Please ensure you have trained a model first by running train_ppo_agent.py"
        )
        return

    # Load the trained agent
    try:
        agent = load_trained_agent(checkpoint_path, device)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Create output directory for visualizations
    output_dir = "visualizations"

    # Generate visualizations
    try:
        visualize_rollouts(
            agent=agent, device=device, num_rollouts=4, output_dir=output_dir, seed=42
        )

        logger.info(
            f"""
        ✅ Visualization complete! 
        
        Check the '{output_dir}' directory for SVG animation files:
        - ppo_agent_rollout_1.svg
        - ppo_agent_rollout_2.svg  
        - ppo_agent_rollout_3.svg
        - ppo_agent_rollout_4.svg
        
        You can open these files in a web browser to see the agent playing 2048!
        """
        )

    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        return


if __name__ == "__main__":
    main()
