"""
Visualization and statistical analysis script for trained PPO agent playing 2048.

This script loads a trained PPO model and provides two types of analysis:
1. SVG animations of gameplay rollouts for visual inspection
2. Statistical analysis with bar graphs showing performance distribution

The script automatically reads the Hydra configuration used during training
to properly instantiate the model before loading weights.
"""

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pgx
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ppo import PPOAgent
from src.ppo.torch_action_wrapper import TorchActionFunction
from src.runs import BatchRunner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_hydra_config(results_dir: str) -> DictConfig:
    """
    Load the Hydra configuration from a results directory.

    Parameters
    ----------
    results_dir : str
        Path to the results directory containing .hydra folder

    Returns
    -------
    DictConfig
        Loaded Hydra configuration
    """
    config_path = Path(results_dir) / ".hydra" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Hydra config not found at {config_path}. "
            f"Make sure {results_dir} is a valid results directory from a training run."
        )

    logger.info(f"Loading Hydra config from {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert to OmegaConf DictConfig for compatibility
    cfg = OmegaConf.create(config_dict)

    return cfg


def find_checkpoint_in_dir(results_dir: str) -> str:
    """
    Find the checkpoint file in the specified results directory.
    Prioritizes final_model.pt, then the highest numbered checkpoint_#.pt.

    Parameters
    ----------
    results_dir : str
        Path to the results directory

    Returns
    -------
    str
        Path to the checkpoint file
    """
    results_path = Path(results_dir)

    # First, check for final_model.pt
    final_model_path = results_path / "final_model.pt"
    if final_model_path.exists():
        logger.info(f"Found final_model.pt, using: {final_model_path}")
        return str(final_model_path)

    # Look for checkpoint_#.pt files
    checkpoint_files = list(results_path.glob("checkpoint_*.pt"))

    if checkpoint_files:
        # Extract numbers from checkpoint files and sort by number (highest first)
        def extract_checkpoint_number(filepath: Path) -> int:
            try:
                # Extract number from checkpoint_#.pt pattern
                stem = filepath.stem  # gets "checkpoint_#"
                return int(stem.split("_")[1])
            except (IndexError, ValueError):
                return -1  # fallback for malformed names

        # Sort by checkpoint number in descending order
        checkpoint_files.sort(key=extract_checkpoint_number, reverse=True)
        selected_checkpoint = checkpoint_files[0]
        logger.info(f"Using highest numbered checkpoint: {selected_checkpoint}")
        return str(selected_checkpoint)

    # Look for any other .pt files as fallback
    other_pt_files = list(results_path.glob("*.pt"))

    if not other_pt_files:
        raise FileNotFoundError(f"No checkpoint files (.pt) found in {results_dir}")

    if len(other_pt_files) > 1:
        logger.warning(
            f"Multiple non-standard checkpoint files found, using: {other_pt_files[0]}"
        )

    return str(other_pt_files[0])


def load_trained_agent(results_dir: str, device: torch.device) -> PPOAgent:
    """
    Load a trained PPO agent from results directory.
    Automatically reads the Hydra config to instantiate the model correctly.

    Parameters
    ----------
    results_dir : str
        Path to the results directory containing both config and checkpoint
    device : torch.device
        Device to load the model on

    Returns
    -------
    PPOAgent
        Loaded PPO agent
    """
    # Load the Hydra configuration
    cfg = load_hydra_config(results_dir)

    # Find the checkpoint file
    checkpoint_path = find_checkpoint_in_dir(results_dir)

    logger.info(f"Loading model from {checkpoint_path}")

    # Create agent with configuration from the training run
    agent = PPOAgent(
        observation_dim=cfg.model.observation_dim,
        action_dim=cfg.model.action_dim,
        hidden_dim=cfg.model.hidden_dim,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
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
    num_rollouts_viz: int = 4,
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
    num_rollouts_viz : int
        Number of rollouts to visualize
    output_dir : str
        Directory to save visualizations
    seed : int
        Random seed for reproducible visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating {num_rollouts_viz} rollout visualizations...")

    # Create action function for BatchRunner
    torch_action_fn = TorchActionFunction(
        agent, use_mask=True, sample_actions=False, device=device
    )

    # Create batch runner for environment simulations
    batch_runner = BatchRunner(init_seed=seed, act_fn=torch_action_fn)
    # Run environments to get trajectory data
    states = batch_runner.run_rollout_batch(num_rollouts_viz)

    # Calculate some statistics about this rollout
    final_state = states[-1]
    for rollout_idx in range(num_rollouts_viz):
        # Extract board from final observation
        final_obs = final_state.observation[rollout_idx]  # Shape: (4, 4, 31)
        # Convert from one-hot to board values
        board_values = final_obs.argmax(axis=-1)  # Shape: (4, 4)
        max_tile = (2**board_values).max()
        num_moves = len(states) - 1  # Subtract initial state

        logger.info(
            f"  Rollout {rollout_idx + 1}: {num_moves} moves, max tile: {max_tile}"
        )

    # Create SVG animation
    output_path = os.path.join(output_dir, "ppo_agent_rollout.svg")
    pgx.save_svg_animation(
        states,
        output_path,
        frame_duration_seconds=0.25,
    )
    logger.info(f"  Saved visualization to: {output_path}")


def generate_rollout_statistics(
    agent: PPOAgent,
    device: torch.device,
    num_rollouts_stats: int = 1000,
    output_dir: str = "visualizations",
    seed: int = 42,
):
    """
    Generate statistical analysis of agent performance across many rollouts.
    Creates a bar graph showing the percentage of episodes that reach certain max tile values.

    Parameters
    ----------
    agent : PPOAgent
        Trained PPO agent
    device : torch.device
        Device for inference
    num_rollouts_stats : int
        Number of rollouts for statistical analysis
    output_dir : str
        Directory to save statistics plots
    seed : int
        Random seed for reproducible statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating statistics from {num_rollouts_stats} rollouts...")

    # Create action function for BatchRunner
    torch_action_fn = TorchActionFunction(
        agent, use_mask=True, sample_actions=False, device=device
    )

    # Track max tiles achieved
    max_tiles = []

    # Run rollouts in batches to avoid memory issues
    batch_size = min(100, num_rollouts_stats)  # Process in batches of 100 or less
    num_batches = (num_rollouts_stats + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(
            batch_size, num_rollouts_stats - batch_idx * batch_size
        )

        # Create batch runner for this batch
        batch_seed = seed + batch_idx * batch_size
        batch_runner = BatchRunner(init_seed=batch_seed, act_fn=torch_action_fn)

        # Run environments to get trajectory data
        states = batch_runner.run_rollout_batch(current_batch_size)

        # Extract max tiles from final states
        final_state = states[-1]
        for rollout_idx in range(current_batch_size):
            # Extract board from final observation
            final_obs = final_state.observation[rollout_idx]  # Shape: (4, 4, 31)
            # Convert from one-hot to board values
            board_values = final_obs.argmax(axis=-1)  # Shape: (4, 4)
            max_tile = (2**board_values).max()
            max_tiles.append(max_tile.item())

        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            logger.info(
                f"  Processed {(batch_idx + 1) * batch_size}/{num_rollouts_stats} rollouts..."
            )

    # Create statistics
    tile_counts = Counter(max_tiles)

    # Define common tile values to show (powers of 2)
    common_tiles = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    # Calculate percentages for each tile value
    tile_percentages = {}
    for tile in common_tiles:
        if tile in tile_counts:
            percentage = (tile_counts[tile] / num_rollouts_stats) * 100
            tile_percentages[tile] = percentage
        else:
            tile_percentages[tile] = 0.0

    # Also include any other tiles that were achieved
    for tile in tile_counts:
        if tile not in tile_percentages:
            percentage = (tile_counts[tile] / num_rollouts_stats) * 100
            tile_percentages[tile] = percentage

    # Sort tiles by value
    sorted_tiles = sorted(tile_percentages.keys())
    percentages = [tile_percentages[tile] for tile in sorted_tiles]

    # Create bar graph
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(sorted_tiles)), percentages)

    # Customize the plot
    plt.xlabel("Max Tile Value Reached")
    plt.ylabel("Percentage of Episodes (%)")
    plt.title(f"PPO Agent Performance Distribution\n({num_rollouts_stats} episodes)")

    # Set x-axis labels
    plt.xticks(range(len(sorted_tiles)), sorted_tiles, rotation=45)

    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        if pct > 0:  # Only show labels for non-zero bars
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.1,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
            )

    # Color bars differently for major milestones
    for i, tile in enumerate(sorted_tiles):
        if tile >= 2048:
            bars[i].set_color("gold")
        elif tile >= 1024:
            bars[i].set_color("orange")
        elif tile >= 512:
            bars[i].set_color("lightcoral")
        else:
            bars[i].set_color("lightblue")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    stats_path = os.path.join(output_dir, "ppo_agent_statistics.png")
    plt.savefig(stats_path, dpi=300, bbox_inches="tight")
    logger.info(f"  Saved statistics plot to: {stats_path}")

    # Print summary statistics
    logger.info("Performance Summary:")
    logger.info(f"  Total episodes: {num_rollouts_stats}")
    logger.info(f"  Max tile achieved: {max(max_tiles)}")
    logger.info(f"  Average max tile: {np.mean(max_tiles):.1f}")
    logger.info(
        f"  Episodes reaching 2048+: {sum(1 for t in max_tiles if t >= 2048)} ({(sum(1 for t in max_tiles if t >= 2048) / num_rollouts_stats * 100):.1f}%)"
    )
    logger.info(
        f"  Episodes reaching 1024+: {sum(1 for t in max_tiles if t >= 1024)} ({(sum(1 for t in max_tiles if t >= 1024) / num_rollouts_stats * 100):.1f}%)"
    )
    logger.info(
        f"  Episodes reaching 512+: {sum(1 for t in max_tiles if t >= 512)} ({(sum(1 for t in max_tiles if t >= 512) / num_rollouts_stats * 100):.1f}%)"
    )

    plt.close()  # Close the plot to free memory


def find_latest_results_dir(results_root: str = "results") -> str:
    """
    Find the most recent results directory in the results root.

    Parameters
    ----------
    results_root : str
        Root results directory to search

    Returns
    -------
    str
        Path to the latest results directory
    """
    results_path = Path(results_root)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory {results_root} does not exist")

    # Look for subdirectories that contain .hydra folders (indicating a training run)
    run_dirs = []
    for item in results_path.rglob(".hydra"):
        if item.is_dir():
            run_dirs.append(item.parent)

    if not run_dirs:
        raise FileNotFoundError(f"No training run directories found in {results_root}")

    # Sort by directory name timestamp and get the latest
    # Directory names are expected to be in format: YYYY-MM-DD_HH-MM-SS
    # This format is naturally sortable as strings
    latest_dir = max(run_dirs, key=lambda path: path.name)

    return str(latest_dir)


def main():
    """Main analysis function for visualization and statistics."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze trained PPO agent playing 2048 with visualizations and statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to specific results directory. If not provided, will find the latest one.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="../results",
        help="Root directory to search for results (used when --results-dir is not specified)",
    )
    parser.add_argument(
        "--num-rollouts-viz",
        type=int,
        default=4,
        help="Number of rollouts to create for SVG visualization",
    )
    parser.add_argument(
        "--num-rollouts-stats",
        type=int,
        default=1000,
        help="Number of rollouts to run for statistical analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible visualizations",
    )

    args = parser.parse_args()

    logger.info("Starting PPO Agent Visualization")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Determine results directory to use
    if args.results_dir:
        results_dir = args.results_dir
        logger.info(f"Using specified results directory: {results_dir}")
    else:
        try:
            results_dir = find_latest_results_dir(args.results_root)
            logger.info(f"Found latest results directory: {results_dir}")
        except FileNotFoundError as e:
            logger.error(f"Error finding results directory: {e}")
            logger.error(
                "Please ensure you have trained a model first by running train_ppo_agent.py, "
                "or specify a results directory with --results-dir"
            )
            return

    # Load the trained agent
    try:
        agent = load_trained_agent(results_dir, device)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Generate visualizations and statistics
    try:
        # Generate SVG visualization
        visualize_rollouts(
            agent=agent,
            device=device,
            num_rollouts_viz=args.num_rollouts_viz,
            output_dir=args.output_dir,
            seed=args.seed,
        )

        # Generate statistical analysis
        generate_rollout_statistics(
            agent=agent,
            device=device,
            num_rollouts_stats=args.num_rollouts_stats,
            output_dir=args.output_dir,
            seed=args.seed,
        )

        logger.info(
            f"""
        ✅ Analysis complete! 
        
        Check the '{args.output_dir}' directory for:
        - ppo_agent_rollout.svg (SVG animation of {args.num_rollouts_viz} games)
        - ppo_agent_statistics.png (Performance statistics from {args.num_rollouts_stats} games)

        You can open the SVG file in a web browser to see the agent playing 2048!

        Model loaded from: {results_dir}
        """
        )

    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        return


if __name__ == "__main__":
    main()
