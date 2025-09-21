"""
Visualization and statistical analysis script for naive strategies playing 2048.

This script provides analysis for two naive strategies:
1. Random action selection (act_randomly)
2. DRUL action selection (act_drul - Down, Right, Up, Left priority)

The script provides two types of analysis:
1. SVG animations of gameplay rollouts for visual inspection
2. Statistical analysis with bar graphs showing performance distribution

Usage examples:
    python viz_naive_strategies.py --strategy random
    python viz_naive_strategies.py --strategy drul
    python viz_naive_strategies.py --strategy both --num-rollouts-stats 500
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

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.actions.act_drul import act_drul
from src.actions.act_randomly import act_randomly
from src.runs import BatchRunner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_strategy_function(strategy_name: str):
    """
    Get the action function for the specified strategy.

    Parameters
    ----------
    strategy_name : str
        Name of the strategy ('random' or 'drul')

    Returns
    -------
    callable
        Action function for the strategy
    """
    if strategy_name == "random":
        return act_randomly
    elif strategy_name == "drul":
        return act_drul
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Use 'random' or 'drul'.")


def visualize_rollouts(
    strategy_name: str,
    num_rollouts_viz: int = 4,
    output_dir: str = "visualizations",
    seed: int = 42,
):
    """
    Create and save SVG visualizations of the strategy playing.

    Parameters
    ----------
    strategy_name : str
        Name of the strategy ('random' or 'drul')
    num_rollouts_viz : int
        Number of rollouts to visualize
    output_dir : str
        Directory to save visualizations
    seed : int
        Random seed for reproducible visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        f"Generating {num_rollouts_viz} rollout visualizations for {strategy_name} strategy..."
    )

    # Get the action function
    action_fn = get_strategy_function(strategy_name)

    # Create batch runner for environment simulations
    batch_runner = BatchRunner(init_seed=seed, act_fn=action_fn)

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
    output_path = os.path.join(output_dir, f"{strategy_name}_strategy_rollout.svg")
    pgx.save_svg_animation(
        states,
        output_path,
        frame_duration_seconds=0.25,
    )
    logger.info(f"  Saved visualization to: {output_path}")


def generate_rollout_statistics(
    strategy_name: str,
    num_rollouts_stats: int = 1000,
    output_dir: str = "visualizations",
    seed: int = 42,
):
    """
    Generate statistical analysis of strategy performance across many rollouts.
    Creates a bar graph showing the percentage of episodes that reach certain max tile values.

    Parameters
    ----------
    strategy_name : str
        Name of the strategy ('random' or 'drul')
    num_rollouts_stats : int
        Number of rollouts for statistical analysis
    output_dir : str
        Directory to save statistics plots
    seed : int
        Random seed for reproducible statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        f"Generating statistics from {num_rollouts_stats} rollouts for {strategy_name} strategy..."
    )

    # Get the action function
    action_fn = get_strategy_function(strategy_name)

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
        batch_runner = BatchRunner(init_seed=batch_seed, act_fn=action_fn)

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
    strategy_display_name = (
        strategy_name.upper() if strategy_name == "drul" else strategy_name.capitalize()
    )
    plt.xlabel("Max Tile Value Reached")
    plt.ylabel("Percentage of Episodes (%)")
    plt.title(
        f"{strategy_display_name} Strategy Performance Distribution\n({num_rollouts_stats} episodes)"
    )

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
    stats_path = os.path.join(output_dir, f"{strategy_name}_strategy_statistics.png")
    plt.savefig(stats_path, dpi=300, bbox_inches="tight")
    logger.info(f"  Saved statistics plot to: {stats_path}")

    # Print summary statistics
    logger.info(f"Performance Summary for {strategy_display_name} Strategy:")
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

    return {
        "max_tiles": max_tiles,
        "tile_counts": tile_counts,
        "avg_max_tile": np.mean(max_tiles),
        "max_achieved": max(max_tiles),
    }


def compare_strategies(
    strategies: list,
    num_rollouts_stats: int = 1000,
    output_dir: str = "visualizations",
    seed: int = 42,
):
    """
    Compare multiple strategies side by side.

    Parameters
    ----------
    strategies : list
        List of strategy names to compare
    num_rollouts_stats : int
        Number of rollouts for statistical analysis per strategy
    output_dir : str
        Directory to save comparison plots
    seed : int
        Random seed for reproducible statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Comparing strategies: {', '.join(strategies)}")

    strategy_results = {}

    # Collect data for each strategy
    for strategy in strategies:
        logger.info(f"\nAnalyzing {strategy} strategy...")
        results = generate_rollout_statistics(
            strategy_name=strategy,
            num_rollouts_stats=num_rollouts_stats,
            output_dir=output_dir,
            seed=seed,
        )
        strategy_results[strategy] = results

    # Create comparison plot
    plt.figure(figsize=(14, 10))

    # Define common tile values to compare
    common_tiles = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    # Prepare data for comparison
    strategy_names = list(strategy_results.keys())
    x = np.arange(len(common_tiles))
    width = 0.35  # Width of bars

    # Calculate percentages for each strategy
    strategy_percentages = {}
    for strategy in strategy_names:
        tile_counts = strategy_results[strategy]["tile_counts"]
        percentages = []
        for tile in common_tiles:
            if tile in tile_counts:
                percentage = (tile_counts[tile] / num_rollouts_stats) * 100
            else:
                percentage = 0.0
            percentages.append(percentage)
        strategy_percentages[strategy] = percentages

    # Create grouped bar chart
    if len(strategy_names) == 2:
        bars1 = plt.bar(
            x - width / 2,
            strategy_percentages[strategy_names[0]],
            width,
            label=(
                strategy_names[0].upper()
                if strategy_names[0] == "drul"
                else strategy_names[0].capitalize()
            ),
            alpha=0.8,
        )
        bars2 = plt.bar(
            x + width / 2,
            strategy_percentages[strategy_names[1]],
            width,
            label=(
                strategy_names[1].upper()
                if strategy_names[1] == "drul"
                else strategy_names[1].capitalize()
            ),
            alpha=0.8,
        )

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.1,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
    else:
        # Single strategy case
        bars = plt.bar(
            x,
            strategy_percentages[strategy_names[0]],
            width,
            label=(
                strategy_names[0].upper()
                if strategy_names[0] == "drul"
                else strategy_names[0].capitalize()
            ),
            alpha=0.8,
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                )

    # Customize the plot
    plt.xlabel("Max Tile Value Reached")
    plt.ylabel("Percentage of Episodes (%)")
    plt.title(
        f"Strategy Performance Comparison\n({num_rollouts_stats} episodes per strategy)"
    )
    plt.xticks(x, common_tiles, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save comparison plot
    comparison_path = os.path.join(
        output_dir, f"strategy_comparison_{'+'.join(strategy_names)}.png"
    )
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved comparison plot to: {comparison_path}")

    # Print comparison summary
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY COMPARISON SUMMARY")
    logger.info("=" * 60)

    for strategy in strategy_names:
        results = strategy_results[strategy]
        strategy_display = (
            strategy.upper() if strategy == "drul" else strategy.capitalize()
        )
        logger.info(f"\n{strategy_display} Strategy:")
        logger.info(f"  Average max tile: {results['avg_max_tile']:.1f}")
        logger.info(f"  Best performance: {results['max_achieved']}")
        logger.info(
            f"  Reached 1024+: {sum(1 for t in results['max_tiles'] if t >= 1024)}/{num_rollouts_stats} ({(sum(1 for t in results['max_tiles'] if t >= 1024) / num_rollouts_stats * 100):.1f}%)"
        )
        logger.info(
            f"  Reached 512+: {sum(1 for t in results['max_tiles'] if t >= 512)}/{num_rollouts_stats} ({(sum(1 for t in results['max_tiles'] if t >= 512) / num_rollouts_stats * 100):.1f}%)"
        )

    plt.close()


def main():
    """Main analysis function for visualization and statistics."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze naive strategies playing 2048 with visualizations and statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["random", "drul", "both"],
        default="both",
        help="Strategy to analyze: 'random', 'drul', or 'both' for comparison",
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

    logger.info("Starting Naive Strategy Visualization")

    # Determine which strategies to analyze
    if args.strategy == "both":
        strategies = ["random", "drul"]
    else:
        strategies = [args.strategy]

    # Generate visualizations and statistics
    try:
        # Generate SVG visualizations for each strategy
        for strategy in strategies:
            visualize_rollouts(
                strategy_name=strategy,
                num_rollouts_viz=args.num_rollouts_viz,
                output_dir=args.output_dir,
                seed=args.seed,
            )

        # Generate statistical analysis
        if len(strategies) > 1:
            # Compare strategies
            compare_strategies(
                strategies=strategies,
                num_rollouts_stats=args.num_rollouts_stats,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        else:
            # Single strategy analysis
            generate_rollout_statistics(
                strategy_name=strategies[0],
                num_rollouts_stats=args.num_rollouts_stats,
                output_dir=args.output_dir,
                seed=args.seed,
            )

        # Summary message
        strategy_display = " and ".join(
            [s.upper() if s == "drul" else s.capitalize() for s in strategies]
        )
        output_files = []

        for strategy in strategies:
            output_files.append(
                f"- {strategy}_strategy_rollout.svg (SVG animation of {args.num_rollouts_viz} games)"
            )
            output_files.append(
                f"- {strategy}_strategy_statistics.png (Performance statistics from {args.num_rollouts_stats} games)"
            )

        if len(strategies) > 1:
            output_files.append(
                f"- strategy_comparison_{'+'.join(strategies)}.png (Side-by-side comparison)"
            )

        logger.info(
            f"""
        ✅ Analysis complete for {strategy_display} strategy/strategies!
        
        Check the '{args.output_dir}' directory for:
        """
            + "\n        ".join(output_files)
            + f"""

        You can open the SVG files in a web browser to see the strategies playing 2048!
        """
        )

    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise


if __name__ == "__main__":
    main()
