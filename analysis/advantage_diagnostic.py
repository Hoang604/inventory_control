"""
Advantage Distribution Diagnostic

Loads trained Q-net and V-net checkpoints and analyzes the advantage distribution
across the dataset to diagnose training issues.

Output: analysis/advantage_diagnostic.txt
"""
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def diagnose_advantage(
    q_checkpoint_path: str,
    v_checkpoint_path: str,
    dataset_path: str,
    output_file: str
) -> None:
    """
    Loads Q and V networks, computes advantage over the dataset, and writes
    diagnostic information to a file.

    Args:
        q_checkpoint_path: Path to Q-network checkpoint.
        v_checkpoint_path: Path to V-network checkpoint.
        dataset_path: Path to the dataset .pt file.
        output_file: Path to write diagnostic output.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from src.models.iql.critics import QNet, VNet
    from utils.config_loader import load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    # Load dataset
    dataset = torch.load(dataset_path)
    states = dataset['states'].to(device)
    actions = dataset['actions'].to(device)
    rewards = dataset['rewards'].to(device)

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

    # Compute Q, V, and Advantage
    with torch.no_grad():
        batch_size = 1024
        all_q = []
        all_v = []
        all_adv = []

        for i in range(0, len(states), batch_size):
            s = states[i:i+batch_size]
            a = actions[i:i+batch_size]

            q_val = q_net(s, a)
            v_val = v_net(s)
            adv = q_val - v_val

            all_q.append(q_val.cpu())
            all_v.append(v_val.cpu())
            all_adv.append(adv.cpu())

        q_values = torch.cat(all_q).numpy().flatten()
        v_values = torch.cat(all_v).numpy().flatten()
        advantages = torch.cat(all_adv).numpy().flatten()
        rewards_np = rewards.cpu().numpy().flatten()

    # Compute statistics
    positive_adv_count = (advantages > 0).sum()
    positive_adv_pct = positive_adv_count / len(advantages) * 100

    # Write diagnostics to file
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ADVANTAGE DISTRIBUTION DIAGNOSTIC\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("CHECKPOINTS USED:\n")
        f.write(f"  Q-Net: {q_checkpoint_path}\n")
        f.write(f"  V-Net: {v_checkpoint_path}\n")
        f.write(f"  Dataset: {dataset_path}\n\n")

        f.write("-" * 70 + "\n")
        f.write("REWARD STATISTICS (raw, unscaled)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean:   {rewards_np.mean():.4f}\n")
        f.write(f"  Std:    {rewards_np.std():.4f}\n")
        f.write(f"  Min:    {rewards_np.min():.4f}\n")
        f.write(f"  Max:    {rewards_np.max():.4f}\n")
        f.write(f"  Median: {np.median(rewards_np):.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("Q-VALUE STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean:   {q_values.mean():.4f}\n")
        f.write(f"  Std:    {q_values.std():.4f}\n")
        f.write(f"  Min:    {q_values.min():.4f}\n")
        f.write(f"  Max:    {q_values.max():.4f}\n")
        f.write(f"  Median: {np.median(q_values):.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("V-VALUE STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean:   {v_values.mean():.4f}\n")
        f.write(f"  Std:    {v_values.std():.4f}\n")
        f.write(f"  Min:    {v_values.min():.4f}\n")
        f.write(f"  Max:    {v_values.max():.4f}\n")
        f.write(f"  Median: {np.median(v_values):.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("ADVANTAGE STATISTICS (Q - V)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean:         {advantages.mean():.4f}\n")
        f.write(f"  Std:          {advantages.std():.4f}\n")
        f.write(f"  Min:          {advantages.min():.4f}\n")
        f.write(f"  Max:          {advantages.max():.4f}\n")
        f.write(f"  Median:       {np.median(advantages):.4f}\n")
        f.write(f"  Positive:     {positive_adv_count} / {len(advantages)} ({positive_adv_pct:.2f}%)\n\n")

        f.write("-" * 70 + "\n")
        f.write("PERCENTILE ANALYSIS\n")
        f.write("-" * 70 + "\n")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        f.write(f"  {'Percentile':<12} {'Q-Value':<12} {'V-Value':<12} {'Advantage':<12}\n")
        for p in percentiles:
            q_p = np.percentile(q_values, p)
            v_p = np.percentile(v_values, p)
            a_p = np.percentile(advantages, p)
            f.write(f"  {p:<12} {q_p:<12.4f} {v_p:<12.4f} {a_p:<12.4f}\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("DIAGNOSIS\n")
        f.write("=" * 70 + "\n")

        if positive_adv_pct < 1:
            f.write("[PROBLEM] Advantage is almost never positive.\n")
            if v_values.mean() > q_values.mean():
                f.write("  -> V-network is systematically higher than Q-network.\n")
                f.write("  -> Possible causes:\n")
                f.write("     1. tau=0.8 expectile pushes V too high\n")
                f.write("     2. Q-network underfitting or not converging\n")
                f.write("     3. Target Q-network not being updated properly\n")
        elif positive_adv_pct < 10:
            f.write("[WARNING] Very few positive advantages (< 10%).\n")
            f.write("  -> This can be normal for tau=0.8, but monitor closely.\n")
        elif positive_adv_pct > 50:
            f.write("[WARNING] Too many positive advantages (> 50%).\n")
            f.write("  -> V-network may not be learning the upper expectile.\n")
        else:
            f.write("[OK] Advantage distribution looks reasonable.\n")
            f.write(f"  -> {positive_adv_pct:.1f}% positive advantages.\n")

        f.write("\n")

    print(f"Diagnostic written to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose advantage distribution")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment folder name in checkpoints/")
    args = parser.parse_args()

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.experiment
    q_path = checkpoint_dir / "q_net" / "best_loss.pth"
    v_path = checkpoint_dir / "v_net" / "best_loss.pth"
    dataset_path = PROJECT_ROOT / "data" / "inv_management_base_stock.pt"
    output_path = PROJECT_ROOT / "analysis" / "advantage_diagnostic.txt"

    if not q_path.exists():
        print(f"Q-net checkpoint not found: {q_path}")
        exit(1)
    if not v_path.exists():
        print(f"V-net checkpoint not found: {v_path}")
        exit(1)

    diagnose_advantage(str(q_path), str(v_path), str(dataset_path), str(output_path))
