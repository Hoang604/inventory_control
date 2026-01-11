"""
Theoretical V-Value (Expectile) Analysis by Timestep and Tau

Computes the theoretical tau-expectile of G_t for each timestep, for multiple
tau values. This shows what V-network should converge to at different optimism levels.

Output: analysis/theoretical_v_by_tau.txt and analysis/theoretical_v_by_tau.png
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_discounted_returns(rewards: list, gamma: float = 0.99) -> list:
    """Calculates G_t for every timestep t in an episode."""
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def calculate_expectile(values: np.ndarray, tau: float, tolerance: float = 1e-6, max_iter: int = 1000) -> float:
    """
    Solves for the scalar 'e' that minimizes the asymmetric least squares loss:
    L(e) = sum{ |tau - 1(x < e)| * (x - e)^2 }
    """
    e = np.mean(values)
    
    for _ in range(max_iter):
        indices_high = values > e
        sum_high = np.sum(values[indices_high])
        sum_low = np.sum(values[~indices_high])
        count_high = np.sum(indices_high)
        count_low = np.sum(~indices_high)
        
        if count_high == 0 or count_low == 0:
            break
            
        numerator = tau * sum_high + (1 - tau) * sum_low
        denominator = tau * count_high + (1 - tau) * count_low
        
        new_e = numerator / denominator
        
        if abs(new_e - e) < tolerance:
            return new_e
        e = new_e
    
    return e


def analyze_theoretical_v_by_tau(
    dataset_path: str,
    output_txt: str,
    output_png: str,
    tau_values: list = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
    gamma: float = 0.99,
    steps_per_episode: int = 30,
    reward_scale: float = 0.1
) -> None:
    """
    Computes theoretical V (tau-expectile of G_t) for each timestep and tau value.
    """
    # Load dataset
    dataset = torch.load(dataset_path)
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
    
    # Compute expectile for each tau and timestep
    timesteps = list(range(steps_per_episode))
    gt_means = np.array([np.mean(gt_by_timestep[t]) for t in timesteps])
    
    expectiles_by_tau = {}
    for tau in tau_values:
        expectiles = []
        for t in timesteps:
            values = np.array(gt_by_timestep[t])
            exp_val = calculate_expectile(values, tau)
            expectiles.append(exp_val)
        expectiles_by_tau[tau] = np.array(expectiles)
    
    # Write diagnostics
    with open(output_txt, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("THEORETICAL V-VALUE (EXPECTILE) ANALYSIS BY TIMESTEP AND TAU\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Reward scale: {reward_scale}\n")
        f.write(f"Tau values analyzed: {tau_values}\n\n")
        
        f.write("-" * 100 + "\n")
        f.write("GLOBAL SUMMARY (Mean across all timesteps)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Tau':<10} {'Mean V':<15} {'Diff from G_t Mean':<20}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Mean G_t':<10} {gt_means.mean():<15.4f} {0.0:<20.4f}\n")
        for tau in tau_values:
            v_mean = expectiles_by_tau[tau].mean()
            diff = v_mean - gt_means.mean()
            f.write(f"{tau:<10} {v_mean:<15.4f} {diff:+<20.4f}\n")
        
        f.write("\n")
        f.write("-" * 100 + "\n")
        f.write("DETAILED VALUES BY TIMESTEP\n")
        f.write("-" * 100 + "\n")
        
        # Header
        header = f"{'t':<5} {'G_t Mean':<12}"
        for tau in tau_values:
            header += f"{'τ=' + str(tau):<12}"
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        
        # Values
        for t in timesteps:
            row = f"{t:<5} {gt_means[t]:<12.4f}"
            for tau in tau_values:
                row += f"{expectiles_by_tau[tau][t]:<12.4f}"
            f.write(row + "\n")
        
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 100 + "\n")
        f.write("- tau=0.5 gives the mean (no optimism)\n")
        f.write("- tau=0.8 gives the 80th percentile-like value (high optimism)\n")
        f.write("- Higher tau = V targets higher-return trajectories\n")
        f.write("- Your trained V-net should match the tau value in your config\n")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Theoretical V (tau-Expectile) by Timestep', fontsize=14)
    
    # Left: All curves
    ax1 = axes[0]
    ax1.plot(timesteps, gt_means, 'k-', linewidth=2, label='Mean G_t', alpha=0.8)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(tau_values)))
    for i, tau in enumerate(tau_values):
        ax1.plot(timesteps, expectiles_by_tau[tau], '-', linewidth=1.5, 
                 color=colors[i], label=f'τ={tau}')
    
    ax1.set_xlabel('Timestep t')
    ax1.set_ylabel('Value (scaled)')
    ax1.set_title('V-Expectile Curves by Tau')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Right: Difference from mean at each tau
    ax2 = axes[1]
    x_positions = np.arange(len(tau_values))
    
    # At t=0
    diffs_t0 = [expectiles_by_tau[tau][0] - gt_means[0] for tau in tau_values]
    # At t=14
    diffs_t14 = [expectiles_by_tau[tau][14] - gt_means[14] for tau in tau_values]
    # At t=29
    diffs_t29 = [expectiles_by_tau[tau][29] - gt_means[29] for tau in tau_values]
    
    width = 0.25
    ax2.bar(x_positions - width, diffs_t0, width, label='t=0 (start)', alpha=0.8)
    ax2.bar(x_positions, diffs_t14, width, label='t=14 (mid)', alpha=0.8)
    ax2.bar(x_positions + width, diffs_t29, width, label='t=29 (end)', alpha=0.8)
    
    ax2.set_xlabel('Tau')
    ax2.set_ylabel('V - Mean G_t')
    ax2.set_title('Optimism (V - G_t) by Tau at Different Timesteps')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(tau) for tau in tau_values])
    ax2.legend()
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostics written to: {output_txt}")
    print(f"Plot saved to: {output_png}")


if __name__ == "__main__":
    dataset_path = PROJECT_ROOT / "data" / "inv_management_base_stock.pt"
    output_txt = PROJECT_ROOT / "analysis" / "theoretical_v_by_tau.txt"
    output_png = PROJECT_ROOT / "analysis" / "theoretical_v_by_tau.png"
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        exit(1)
    
    analyze_theoretical_v_by_tau(
        str(dataset_path),
        str(output_txt),
        str(output_png)
    )
