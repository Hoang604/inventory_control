"""
Reward Distribution Analysis

Visualizes the distribution of rewards in the training dataset to understand
the reward landscape and identify potential issues with reward scaling.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot_reward_distribution(dataset_path: str, output_path: str | None = None) -> None:
    """
    Loads a dataset and plots the reward distribution.

    Args:
        dataset_path: Path to the .pt dataset file.
        output_path: Optional path to save the figure. If None, displays interactively.
    """
    dataset = torch.load(dataset_path, weights_only=False)
    rewards = dataset['rewards'].numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Reward Distribution Analysis\n{Path(dataset_path).name}', fontsize=14)

    # 1. Histogram
    ax1 = axes[0]
    ax1.hist(rewards, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(rewards.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rewards.mean():.2f}')
    ax1.axvline(np.median(rewards), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Histogram')
    ax1.legend()

    # 2. Box Plot
    ax2 = axes[1]
    bp = ax2.boxplot(rewards, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Box Plot')
    ax2.set_xticklabels(['Rewards'])

    # 3. Time Series (rewards over episodes)
    ax3 = axes[2]
    episode_length = 30  # Based on your environment config
    num_complete_episodes = len(rewards) // episode_length
    
    if num_complete_episodes > 0:
        episode_rewards = rewards[:num_complete_episodes * episode_length].reshape(-1, episode_length)
        episode_totals = episode_rewards.sum(axis=1)
        
        ax3.plot(episode_totals, alpha=0.6, color='steelblue')
        ax3.axhline(episode_totals.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {episode_totals.mean():.2f}')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Total Episode Reward')
        ax3.set_title(f'Episode Returns ({num_complete_episodes} episodes)')
        ax3.legend()

    plt.tight_layout()

    # Print statistics
    print("=" * 50)
    print("REWARD STATISTICS")
    print("=" * 50)
    print(f"Total transitions: {len(rewards)}")
    print(f"Mean reward:       {rewards.mean():.4f}")
    print(f"Std reward:        {rewards.std():.4f}")
    print(f"Min reward:        {rewards.min():.4f}")
    print(f"Max reward:        {rewards.max():.4f}")
    print(f"Median reward:     {np.median(rewards):.4f}")
    
    if num_complete_episodes > 0:
        print("-" * 50)
        print(f"Episodes:          {num_complete_episodes}")
        print(f"Mean return:       {episode_totals.mean():.4f}")
        print(f"Std return:        {episode_totals.std():.4f}")
        print(f"Min return:        {episode_totals.min():.4f}")
        print(f"Max return:        {episode_totals.max():.4f}")
    print("=" * 50)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        exit(1)

    dataset_files = list(data_dir.glob("*.pt"))
    
    if not dataset_files:
        print(f"No .pt files found in {data_dir}")
        exit(0)

    print(f"Found {len(dataset_files)} datasets. Starting batch analysis...")
    
    for dataset_path in dataset_files:
        try:
            output_path = dataset_path.with_suffix(".png")
            plot_reward_distribution(str(dataset_path), str(output_path))
        except Exception as e:
            print(f"FAILED to process {dataset_path.name}: {e}")
            continue

    print("\nBatch analysis complete.")
