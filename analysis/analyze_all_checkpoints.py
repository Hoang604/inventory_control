"""
Model Q/V Analysis vs Theoretical Values - Exact Backward DP

Computes EXACT theoretical Q* and V* using backward dynamic programming,
matching how IQL trains. Then compares model predictions to these exact targets.

Algorithm:
  1. Start at t=29 (terminal): Q*[29] = r[29], V*[29] = τ-expectile of Q*[29]
  2. Backward to t=0: Q*[t] = r[t] + γ × V*[t+1], V*[t] = τ-expectile of Q*[t]
  3. Compare model Q, V to these exact targets

Output: analysis/checkpoints/<experiment_name>/epoch_XX.png
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.iql.critics import QNet, VNet
from utils.config_loader import load_config


def calculate_expectile(values: np.ndarray, tau: float, tolerance: float = 1e-6, max_iter: int = 1000) -> float:
    """Solves for the tau-expectile of the given values."""
    if len(values) == 0:
        return 0.0
    
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


def compute_exact_q_v_backward(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    tau: float,
    steps_per_episode: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes exact Q* and V* using backward dynamic programming.
    
    This is mathematically identical to how IQL trains:
    - Q*(s, a) = r + γ(1-d) × V*(s')
    - V*(s) = τ-expectile of Q*(s, a) over actions at state s
    
    Since states at the same timestep are grouped, we compute:
    - V*[t] = τ-expectile of {Q*(s, a) for all (s,a) at timestep t}
    - Q*[t, ep] = r[t, ep] + γ × V*[t+1] (per-sample Q)
    
    Args:
        rewards: Shape (num_episodes, steps_per_episode) - rewards per sample
        dones: Shape (num_episodes, steps_per_episode) - done flags
        gamma: Discount factor
        tau: Expectile parameter
        steps_per_episode: Number of steps per episode
    
    Returns:
        q_star: Shape (num_episodes, steps_per_episode) - exact Q* per sample
        v_star: Shape (steps_per_episode,) - exact V* per timestep
        v_star_per_sample: Shape (num_episodes, steps_per_episode) - V* broadcasted
    """
    num_episodes = rewards.shape[0]
    
    q_star = np.zeros((num_episodes, steps_per_episode))
    v_star = np.zeros(steps_per_episode)
    
    # Backward pass: t = 29, 28, ..., 0
    for t in range(steps_per_episode - 1, -1, -1):
        if t == steps_per_episode - 1:
            # Terminal state: Q* = r (no future, done=True)
            q_star[:, t] = rewards[:, t]
        else:
            # Non-terminal: Q* = r + γ × V*[t+1]
            # Note: We use the shared V*[t+1] computed from all episodes
            q_star[:, t] = rewards[:, t] + gamma * v_star[t + 1]
        
        # V*[t] = τ-expectile of Q*[:, t] across all episodes
        v_star[t] = calculate_expectile(q_star[:, t], tau)
    
    # Broadcast V* to per-sample shape for easy error computation
    v_star_per_sample = np.broadcast_to(v_star, (num_episodes, steps_per_episode))
    
    return q_star, v_star, v_star_per_sample


def analyze_single_checkpoint(
    q_checkpoint_path: str,
    v_checkpoint_path: str,
    dataset: dict,
    config: dict,
    device: torch.device,
    gamma: float = 0.99,
    steps_per_episode: int = 30,
    reward_scale: float = 0.1,
    tau: float = 0.8
) -> dict:
    """
    Analyzes a single Q/V checkpoint using exact backward DP targets.
    """
    states = dataset['states']
    actions = dataset['actions']
    rewards_tensor = dataset['rewards'].flatten()
    dones_tensor = dataset['dones'].flatten()

    total_samples = len(states)
    num_episodes = total_samples // steps_per_episode

    # Reshape rewards and dones to (num_episodes, steps_per_episode)
    rewards = rewards_tensor.numpy().reshape(num_episodes, steps_per_episode) * reward_scale
    dones = dones_tensor.numpy().reshape(num_episodes, steps_per_episode)

    # Compute exact Q* and V* using backward DP
    q_star, v_star, v_star_per_sample = compute_exact_q_v_backward(
        rewards, dones, gamma, tau, steps_per_episode
    )

    # Load Q-network
    q_checkpoint = torch.load(q_checkpoint_path, map_location=device)
    q_config = q_checkpoint.get('config', config)
    q_net = QNet(q_config).to(device)
    q_net.load_state_dict(q_checkpoint['model_state_dict'])
    q_net.eval()

    # Load V-network
    v_checkpoint = torch.load(v_checkpoint_path, map_location=device)
    v_config = v_checkpoint.get('config', config)
    v_net = VNet(v_config).to(device)
    v_net.load_state_dict(v_checkpoint['model_state_dict'])
    v_net.eval()

    # Get model predictions for all samples
    q_model = np.zeros((num_episodes, steps_per_episode))
    v_model = np.zeros((num_episodes, steps_per_episode))

    with torch.no_grad():
        for ep in range(num_episodes):
            start_idx = ep * steps_per_episode
            end_idx = start_idx + steps_per_episode

            ep_states = states[start_idx:end_idx].to(device)
            ep_actions = actions[start_idx:end_idx].to(device)

            q_model[ep, :] = q_net(ep_states, ep_actions).cpu().numpy().flatten()
            v_model[ep, :] = v_net(ep_states).cpu().numpy().flatten()

    # Compute per-sample errors
    q_error = q_model - q_star
    v_error = v_model - v_star_per_sample

    # Aggregate statistics by timestep
    timesteps = list(range(steps_per_episode))
    
    q_model_means = q_model.mean(axis=0)
    v_model_means = v_model.mean(axis=0)
    q_star_means = q_star.mean(axis=0)
    
    q_error_by_t = np.abs(q_error).mean(axis=0)  # MAE per timestep
    v_error_by_t = np.abs(v_error).mean(axis=0)  # MAE per timestep

    return {
        'timesteps': timesteps,
        'q_model_means': q_model_means,
        'v_model_means': v_model_means,
        'q_star_means': q_star_means,
        'v_star': v_star,
        'q_error_by_t': q_error_by_t,
        'v_error_by_t': v_error_by_t,
        'mae_q': np.mean(np.abs(q_error)),
        'mae_v': np.mean(np.abs(v_error)),
    }


def create_checkpoint_plot(stats: dict, output_path: str, epoch: int, tau: float):
    """Creates a 2x2 plot for a single checkpoint."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Epoch {epoch} | MAE(Q)={stats["mae_q"]:.3f} | MAE(V)={stats["mae_v"]:.3f} | Exact Backward DP', 
                 fontsize=14)

    timesteps = stats['timesteps']
    
    # Top Left: Q* and V* (exact targets)
    ax1 = axes[0, 0]
    ax1.plot(timesteps, stats['q_star_means'], 'b-', linewidth=2, label='Q* (exact, mean over episodes)')
    ax1.plot(timesteps, stats['v_star'], 'r-', linewidth=2, label=f'V* (exact, τ={tau})')
    ax1.plot(timesteps, stats['q_model_means'], 'b--', linewidth=2, label='Q-Net (model)', alpha=0.7)
    ax1.plot(timesteps, stats['v_model_means'], 'r--', linewidth=2, label='V-Net (model)', alpha=0.7)
    ax1.set_xlabel('Timestep t')
    ax1.set_ylabel('Value (scaled)')
    ax1.set_title('Exact Q*/V* vs Model Predictions')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Top Right: Q comparison
    ax2 = axes[0, 1]
    ax2.plot(timesteps, stats['q_star_means'], 'b-', linewidth=2, label='Q* (exact)')
    ax2.plot(timesteps, stats['q_model_means'], 'b--', linewidth=2, label='Q-Net (model)')
    ax2.fill_between(timesteps, stats['q_star_means'], stats['q_model_means'], alpha=0.2, color='blue')
    ax2.set_xlabel('Timestep t')
    ax2.set_ylabel('Value (scaled)')
    ax2.set_title(f'Q-Network vs Exact Q* (MAE={stats["mae_q"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom Left: V comparison
    ax3 = axes[1, 0]
    ax3.plot(timesteps, stats['v_star'], 'r-', linewidth=2, label=f'V* (exact, τ={tau})')
    ax3.plot(timesteps, stats['v_model_means'], 'r--', linewidth=2, label='V-Net (model)')
    ax3.fill_between(timesteps, stats['v_star'], stats['v_model_means'], alpha=0.2, color='red')
    ax3.set_xlabel('Timestep t')
    ax3.set_ylabel('Value (scaled)')
    ax3.set_title(f'V-Network vs Exact V* (MAE={stats["mae_v"]:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom Right: Error by timestep
    ax4 = axes[1, 1]
    ax4.bar(np.array(timesteps) - 0.2, stats['q_error_by_t'], width=0.4, 
            label='Q Error', alpha=0.7, color='blue')
    ax4.bar(np.array(timesteps) + 0.2, stats['v_error_by_t'], width=0.4, 
            label='V Error', alpha=0.7, color='red')
    ax4.set_xlabel('Timestep t')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Error by Timestep')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_plot(all_stats: list, output_path: str, tau: float):
    """Creates a summary plot showing MAE evolution across epochs."""
    epochs = [s['epoch'] for s in all_stats]
    mae_q = [s['mae_q'] for s in all_stats]
    mae_v = [s['mae_v'] for s in all_stats]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Model Error vs Exact Targets (τ={tau}, Backward DP)', fontsize=14)

    # Left: MAE over epochs
    ax1 = axes[0]
    ax1.plot(epochs, mae_q, 'b-o', linewidth=2, markersize=4, label='MAE: Q vs Q*')
    ax1.plot(epochs, mae_v, 'r-o', linewidth=2, markersize=4, label='MAE: V vs V*')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Error Evolution (Exact Backward DP Targets)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Find best epoch
    combined = [q + v for q, v in zip(mae_q, mae_v)]
    best_idx = np.argmin(combined)
    best_epoch = epochs[best_idx]
    ax1.axvline(best_epoch, color='green', linestyle='--', alpha=0.5)
    ax1.legend()

    # Right: Table
    ax2 = axes[1]
    ax2.axis('off')
    
    sorted_stats = sorted(all_stats, key=lambda x: x['mae_q'] + x['mae_v'])
    
    table_data = []
    for s in sorted_stats[:10]:
        combined_err = s['mae_q'] + s['mae_v']
        table_data.append([s['epoch'], f"{s['mae_q']:.4f}", f"{s['mae_v']:.4f}", f"{combined_err:.4f}"])
    
    table = ax2.table(
        cellText=table_data,
        colLabels=['Epoch', 'MAE Q', 'MAE V', 'Combined'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax2.set_title('Top 10 Epochs (Exact Backward DP)', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return best_epoch


def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoints using exact backward DP")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment folder name in checkpoints/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    tau = config.get('iql', {}).get('tau', 0.8)
    gamma = config.get('iql', {}).get('gamma', 0.99)
    reward_scale = config.get('training', {}).get('reward_scale', 0.1)

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.experiment
    dataset_path = PROJECT_ROOT / "data" / "inv_management_base_stock.pt"
    
    output_dir = PROJECT_ROOT / "analysis" / "checkpoints_exact" / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset: {dataset_path}")
    dataset = torch.load(dataset_path)
    
    q_checkpoint_dir = checkpoint_dir / "q_net"
    v_checkpoint_dir = checkpoint_dir / "v_net"
    
    checkpoint_files = list(q_checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    
    epoch_pattern = re.compile(r'checkpoint_epoch_(\d+)\.pth')
    checkpoints = []
    for f in checkpoint_files:
        match = epoch_pattern.match(f.name)
        if match:
            epoch = int(match.group(1))
            q_path = q_checkpoint_dir / f.name
            v_path = v_checkpoint_dir / f.name
            if v_path.exists():
                checkpoints.append((epoch, str(q_path), str(v_path)))
    
    checkpoints.sort(key=lambda x: x[0])
    
    print(f"Found {len(checkpoints)} checkpoint pairs")
    print(f"Using tau={tau}, gamma={gamma}, reward_scale={reward_scale}")
    print(f"Output directory: {output_dir}")
    
    all_stats = []
    
    for epoch, q_path, v_path in checkpoints:
        print(f"Processing epoch {epoch}...")
        
        stats = analyze_single_checkpoint(
            q_path, v_path, dataset, config, device,
            gamma=gamma, reward_scale=reward_scale, tau=tau
        )
        stats['epoch'] = epoch
        all_stats.append(stats)
        
        output_path = output_dir / f"epoch_{epoch:03d}.png"
        create_checkpoint_plot(stats, str(output_path), epoch, tau)
    
    if all_stats:
        summary_path = output_dir / "summary.png"
        best_epoch = create_summary_plot(all_stats, str(summary_path), tau)
        print(f"\nSummary saved to: {summary_path}")
        print(f"Best epoch (lowest combined error): {best_epoch}")
        
        summary_txt = output_dir / "summary.txt"
        with open(summary_txt, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CHECKPOINT ANALYSIS - EXACT BACKWARD DP TARGETS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Experiment: {args.experiment}\n")
            f.write(f"Tau: {tau}\n")
            f.write(f"Gamma: {gamma}\n")
            f.write(f"Reward Scale: {reward_scale}\n")
            f.write(f"Total checkpoints: {len(all_stats)}\n\n")
            
            f.write("METHOD:\n")
            f.write("  Q*[t] = r[t] + γ × V*[t+1]  (backward from t=29 to t=0)\n")
            f.write("  V*[t] = τ-expectile of {Q*[t] over all episodes}\n")
            f.write("  This is mathematically identical to how IQL trains.\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("ALL EPOCHS (sorted by combined error)\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Epoch':<10} {'MAE Q':<15} {'MAE V':<15} {'Combined':<15}\n")
            f.write("-" * 70 + "\n")
            
            for s in sorted(all_stats, key=lambda x: x['mae_q'] + x['mae_v']):
                combined = s['mae_q'] + s['mae_v']
                f.write(f"{s['epoch']:<10} {s['mae_q']:<15.4f} {s['mae_v']:<15.4f} {combined:<15.4f}\n")
        
        print(f"Summary text saved to: {summary_txt}")


if __name__ == "__main__":
    main()
