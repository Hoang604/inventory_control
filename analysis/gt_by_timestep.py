"""
Discounted Return (G_t) Analysis by Timestep

Computes and visualizes the average discounted return G_t for each timestep t
in the episode. This shows how expected future value decreases as we approach
the terminal state.

Output: analysis/gt_by_timestep.txt and analysis/gt_by_timestep.png
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_discounted_returns(rewards: list, gamma: float = 0.99) -> list:
    """
    Calculates G_t (discounted return) for every timestep t in an episode.
    
    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{T-t} * r_T
    
    Args:
        rewards: List of rewards for one episode.
        gamma: Discount factor.
    
    Returns:
        List of discounted returns, one for each timestep.
    """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def analyze_gt_by_timestep(
    dataset_path: str,
    output_txt: str,
    output_png: str,
    gamma: float = 0.99,
    steps_per_episode: int = 30,
    reward_scale: float = 0.1
) -> None:
    """
    Analyzes the distribution of G_t values for each timestep and writes
    diagnostics to file.
    
    Args:
        dataset_path: Path to the dataset .pt file.
        output_txt: Path to write text diagnostics.
        output_png: Path to save the plot.
        gamma: Discount factor.
        steps_per_episode: Number of steps per episode.
        reward_scale: Reward scaling factor used during training.
    """
    # Load dataset
    dataset = torch.load(dataset_path, weights_only=False)
    rewards_tensor = dataset['rewards'].flatten()
    
    total_samples = len(rewards_tensor)
    num_episodes = total_samples // steps_per_episode
    
    # Compute G_t for each episode and organize by timestep
    gt_by_timestep = {t: [] for t in range(steps_per_episode)}
    
    for ep in range(num_episodes):
        start_idx = ep * steps_per_episode
        end_idx = start_idx + steps_per_episode
        
        ep_rewards = rewards_tensor[start_idx:end_idx].tolist()
        ep_returns = compute_discounted_returns(ep_rewards, gamma)
        
        for t, g in enumerate(ep_returns):
            gt_by_timestep[t].append(g * reward_scale)
    
    # Compute statistics for each timestep
    timesteps = list(range(steps_per_episode))
    means = []
    stds = []
    mins = []
    maxs = []
    medians = []
    
    for t in timesteps:
        values = np.array(gt_by_timestep[t])
        means.append(values.mean())
        stds.append(values.std())
        mins.append(values.min())
        maxs.append(values.max())
        medians.append(np.median(values))
    
    means = np.array(means)
    stds = np.array(stds)
    
    # Write diagnostics to file
    with open(output_txt, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DISCOUNTED RETURN (G_t) ANALYSIS BY TIMESTEP\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Steps per episode: {steps_per_episode}\n")
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Reward scale: {reward_scale}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Global Mean G_t:     {np.mean(means):.4f}\n")
        f.write(f"G_0 (start state):   {means[0]:.4f} ± {stds[0]:.4f}\n")
        f.write(f"G_14 (mid state):    {means[14]:.4f} ± {stds[14]:.4f}\n")
        f.write(f"G_29 (end state):    {means[29]:.4f} ± {stds[29]:.4f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("DETAILED G_t BY TIMESTEP (scaled)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'t':<5} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}\n")
        f.write("-" * 70 + "\n")
        
        for t in timesteps:
            f.write(f"{t:<5} {means[t]:<10.4f} {stds[t]:<10.4f} "
                    f"{mins[t]:<10.4f} {maxs[t]:<10.4f} {medians[t]:<10.4f}\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 70 + "\n")
        f.write("- G_t represents the expected discounted return from timestep t.\n")
        f.write("- Q(s_t, a_t) should converge to approximately G_t for that timestep.\n")
        f.write("- V(s_t) should converge to the tau-expectile of Q(s_t, a) values.\n")
        f.write(f"- The decreasing trend shows correct discounting behavior.\n")
        f.write(f"- If your Q-values don't follow this curve, there may be a bug.\n")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Discounted Return G_t by Timestep', fontsize=14)
    
    # Left: Mean with std band
    ax1 = axes[0]
    ax1.plot(timesteps, means, 'b-', linewidth=2, label='Mean G_t')
    ax1.fill_between(timesteps, means - stds, means + stds, alpha=0.3, color='blue', label='±1 Std')
    ax1.axhline(np.mean(means), color='red', linestyle='--', label=f'Global Mean: {np.mean(means):.2f}')
    ax1.set_xlabel('Timestep t')
    ax1.set_ylabel('G_t (scaled)')
    ax1.set_title('Mean Discounted Return by Timestep')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Box plot for selected timesteps
    ax2 = axes[1]
    selected_t = [0, 5, 10, 15, 20, 25, 29]
    box_data = [gt_by_timestep[t] for t in selected_t]
    bp = ax2.boxplot(box_data, labels=[f't={t}' for t in selected_t], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('G_t (scaled)')
    ax2.set_title('G_t Distribution at Selected Timesteps')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostics written to: {output_txt}")
    print(f"Plot saved to: {output_png}")


if __name__ == "__main__":
    dataset_path = PROJECT_ROOT / "data" / "inv_management_base_stock.pt"
    output_txt = PROJECT_ROOT / "analysis" / "gt_by_timestep.txt"
    output_png = PROJECT_ROOT / "analysis" / "gt_by_timestep.png"
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        exit(1)
    
    analyze_gt_by_timestep(
        str(dataset_path),
        str(output_txt),
        str(output_png)
    )
