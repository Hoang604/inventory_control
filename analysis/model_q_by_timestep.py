"""
Model Q/V Analysis vs Theoretical Values by Timestep

Loads a trained Q-network and V-network, computes predictions for each timestep,
and compares against both theoretical G_t (mean) and theoretical V (tau-expectile).

Output: analysis/model_q_by_timestep.txt and analysis/model_q_by_timestep.png
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.iql.critics import QNet, VNet
from utils.config_loader import load_config


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
    Solves for the scalar 'e' that minimizes the asymmetric least squares loss.
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


def analyze_model_q_by_timestep(
    q_checkpoint_path: str,
    v_checkpoint_path: str,
    dataset_path: str,
    output_txt: str,
    output_png: str,
    gamma: float = 0.99,
    steps_per_episode: int = 30,
    reward_scale: float = 0.1,
    tau: float = 0.8
) -> None:
    """
    Analyzes the Q and V network predictions by timestep and compares to 
    theoretical G_t (mean) and theoretical V (tau-expectile).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    # Load dataset
    dataset = torch.load(dataset_path)
    states = dataset['states']
    actions = dataset['actions']
    rewards_tensor = dataset['rewards'].flatten()

    total_samples = len(states)
    num_episodes = total_samples // steps_per_episode

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

    # Get tau from config if available
    tau = q_config.get('iql', {}).get('tau', tau)

    # Collect data by timestep
    q_by_timestep = {t: [] for t in range(steps_per_episode)}
    v_by_timestep = {t: [] for t in range(steps_per_episode)}
    gt_by_timestep = {t: [] for t in range(steps_per_episode)}

    with torch.no_grad():
        for ep in range(num_episodes):
            start_idx = ep * steps_per_episode
            end_idx = start_idx + steps_per_episode

            # Get episode data
            ep_states = states[start_idx:end_idx].to(device)
            ep_actions = actions[start_idx:end_idx].to(device)
            ep_rewards = rewards_tensor[start_idx:end_idx].tolist()

            # Compute model predictions
            q_vals = q_net(ep_states, ep_actions).cpu().numpy().flatten()
            v_vals = v_net(ep_states).cpu().numpy().flatten()

            # Compute theoretical G_t
            gt_vals = compute_discounted_returns(ep_rewards, gamma)
            gt_vals = [g * reward_scale for g in gt_vals]

            # Store by timestep
            for t in range(steps_per_episode):
                q_by_timestep[t].append(q_vals[t])
                v_by_timestep[t].append(v_vals[t])
                gt_by_timestep[t].append(gt_vals[t])

    # Compute statistics
    timesteps = list(range(steps_per_episode))
    
    # Model predictions
    q_means = np.array([np.mean(q_by_timestep[t]) for t in timesteps])
    q_stds = np.array([np.std(q_by_timestep[t]) for t in timesteps])
    v_means = np.array([np.mean(v_by_timestep[t]) for t in timesteps])
    v_stds = np.array([np.std(v_by_timestep[t]) for t in timesteps])
    
    # Theoretical values
    gt_means = np.array([np.mean(gt_by_timestep[t]) for t in timesteps])
    gt_stds = np.array([np.std(gt_by_timestep[t]) for t in timesteps])
    
    # Theoretical V (tau-expectile of G_t at each timestep)
    v_theoretical = np.array([
        calculate_expectile(np.array(gt_by_timestep[t]), tau) 
        for t in timesteps
    ])

    # Compute errors
    q_error_vs_gt = q_means - gt_means
    v_error_vs_gt = v_means - gt_means
    v_error_vs_theory = v_means - v_theoretical

    # Write diagnostics
    with open(output_txt, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("MODEL Q/V ANALYSIS VS THEORETICAL VALUES BY TIMESTEP\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Q-Net Checkpoint: {q_checkpoint_path}\n")
        f.write(f"V-Net Checkpoint: {v_checkpoint_path}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Tau (for theoretical V): {tau}\n\n")

        f.write("-" * 100 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Metric':<25} {'Model Q':<12} {'Model V':<12} {'Theory G_t':<12} {'Theory V':<12}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Global Mean':<25} {q_means.mean():<12.4f} {v_means.mean():<12.4f} "
                f"{gt_means.mean():<12.4f} {v_theoretical.mean():<12.4f}\n")
        f.write(f"{'At t=0 (start)':<25} {q_means[0]:<12.4f} {v_means[0]:<12.4f} "
                f"{gt_means[0]:<12.4f} {v_theoretical[0]:<12.4f}\n")
        f.write(f"{'At t=14 (mid)':<25} {q_means[14]:<12.4f} {v_means[14]:<12.4f} "
                f"{gt_means[14]:<12.4f} {v_theoretical[14]:<12.4f}\n")
        f.write(f"{'At t=29 (end)':<25} {q_means[29]:<12.4f} {v_means[29]:<12.4f} "
                f"{gt_means[29]:<12.4f} {v_theoretical[29]:<12.4f}\n\n")
        
        f.write(f"{'MAE: Model Q vs G_t':<25} {np.mean(np.abs(q_error_vs_gt)):<12.4f}\n")
        f.write(f"{'MAE: Model V vs G_t':<25} {np.mean(np.abs(v_error_vs_gt)):<12.4f}\n")
        f.write(f"{'MAE: Model V vs Theory V':<25} {np.mean(np.abs(v_error_vs_theory)):<12.4f}\n\n")

        f.write("-" * 100 + "\n")
        f.write("DETAILED VALUES BY TIMESTEP\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'t':<4} {'Model Q':<10} {'Model V':<10} {'Theory G_t':<10} "
                f"{'Theory V':<10} {'Q-G_t':<10} {'V-TheoryV':<10}\n")
        f.write("-" * 100 + "\n")

        for t in timesteps:
            f.write(f"{t:<4} {q_means[t]:<10.4f} {v_means[t]:<10.4f} {gt_means[t]:<10.4f} "
                    f"{v_theoretical[t]:<10.4f} {q_error_vs_gt[t]:<10.4f} {v_error_vs_theory[t]:<10.4f}\n")

        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("DIAGNOSIS\n")
        f.write("=" * 100 + "\n")

        # Q diagnosis
        if np.mean(np.abs(q_error_vs_gt)) < 2.0:
            f.write("[OK] Q-network closely tracks theoretical G_t values.\n")
        elif np.mean(q_error_vs_gt) > 0:
            f.write(f"[WARNING] Q-network is overestimating by avg {np.mean(q_error_vs_gt):.2f}.\n")
        else:
            f.write(f"[WARNING] Q-network is underestimating by avg {-np.mean(q_error_vs_gt):.2f}.\n")

        # V diagnosis
        if np.mean(np.abs(v_error_vs_theory)) < 2.0:
            f.write(f"[OK] V-network closely matches theoretical tau={tau} expectile.\n")
        elif np.mean(v_error_vs_theory) > 0:
            f.write(f"[INFO] V-network is more optimistic than tau={tau} expectile by avg {np.mean(v_error_vs_theory):.2f}.\n")
        else:
            f.write(f"[WARNING] V-network is less optimistic than expected (below tau={tau} expectile).\n")

    # Create visualization - 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Model vs Theoretical Values by Timestep (τ={tau})', fontsize=14)

    # Top Left: All 4 curves
    ax1 = axes[0, 0]
    ax1.plot(timesteps, gt_means, 'k-', linewidth=2, label='Theory: G_t (mean)', alpha=0.8)
    ax1.plot(timesteps, v_theoretical, 'k--', linewidth=2, label=f'Theory: V (τ={tau} expectile)', alpha=0.8)
    ax1.plot(timesteps, q_means, 'b-', linewidth=2, label='Model: Q-Net')
    ax1.plot(timesteps, v_means, 'r-', linewidth=2, label='Model: V-Net')
    ax1.fill_between(timesteps, gt_means - gt_stds, gt_means + gt_stds, alpha=0.1, color='black')
    ax1.set_xlabel('Timestep t')
    ax1.set_ylabel('Value (scaled)')
    ax1.set_title('All Values: Model vs Theoretical')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Top Right: Theory only (G_t and V expectile)
    ax2 = axes[0, 1]
    ax2.plot(timesteps, gt_means, 'k-', linewidth=2, label='G_t (mean)')
    ax2.fill_between(timesteps, gt_means - gt_stds, gt_means + gt_stds, alpha=0.2, color='black', label='±1 Std')
    ax2.plot(timesteps, v_theoretical, 'g-', linewidth=2, label=f'V (τ={tau} expectile)')
    ax2.set_xlabel('Timestep t')
    ax2.set_ylabel('Value (scaled)')
    ax2.set_title('Theoretical Values Only')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom Left: Model Q vs Theory G_t
    ax3 = axes[1, 0]
    ax3.plot(timesteps, gt_means, 'k-', linewidth=2, label='Theory: G_t', alpha=0.8)
    ax3.plot(timesteps, q_means, 'b-', linewidth=2, label='Model: Q-Net')
    ax3.fill_between(timesteps, q_means - q_stds, q_means + q_stds, alpha=0.2, color='blue')
    ax3.set_xlabel('Timestep t')
    ax3.set_ylabel('Value (scaled)')
    ax3.set_title('Q-Network vs Theoretical G_t')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom Right: Model V vs Theory V
    ax4 = axes[1, 1]
    ax4.plot(timesteps, v_theoretical, 'g-', linewidth=2, label=f'Theory: V (τ={tau})', alpha=0.8)
    ax4.plot(timesteps, v_means, 'r-', linewidth=2, label='Model: V-Net')
    ax4.fill_between(timesteps, v_means - v_stds, v_means + v_stds, alpha=0.2, color='red')
    ax4.set_xlabel('Timestep t')
    ax4.set_ylabel('Value (scaled)')
    ax4.set_title('V-Network vs Theoretical Expectile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Diagnostics written to: {output_txt}")
    print(f"Plot saved to: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model Q/V values by timestep")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment folder name in checkpoints/")
    args = parser.parse_args()

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.experiment
    q_path = checkpoint_dir / "q_net" / "best_loss.pth"
    v_path = checkpoint_dir / "v_net" / "best_loss.pth"
    dataset_path = PROJECT_ROOT / "data" / "inv_management_base_stock.pt"
    output_txt = PROJECT_ROOT / "analysis" / "model_q_by_timestep.txt"
    output_png = PROJECT_ROOT / "analysis" / "model_q_by_timestep.png"

    if not q_path.exists():
        print(f"Q-net checkpoint not found: {q_path}")
        exit(1)
    if not v_path.exists():
        print(f"V-net checkpoint not found: {v_path}")
        exit(1)

    analyze_model_q_by_timestep(
        str(q_path), str(v_path), str(dataset_path),
        str(output_txt), str(output_png)
    )
