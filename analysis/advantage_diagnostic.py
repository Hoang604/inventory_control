"""
Advantage Distribution Diagnostic - All Checkpoints

Loads trained Q-net and V-net checkpoints and analyzes the advantage distribution
across the dataset to diagnose training issues for ALL checkpoints in the experiment.

Output: analysis/advantage_diagnostic_<experiment>.txt
"""
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import re
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def diagnose_advantage(
    q_checkpoint_path: str,
    v_checkpoint_path: str,
    dataset_path: str,
    output_file: str,
    checkpoint_name: str
) -> None:
    """
    Loads Q and V networks, computes advantage over the dataset, and APPENDS
    diagnostic information to a file.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from src.models.iql.critics import QNet, VNet
    from utils.config_loader import load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # helper to load config safely
    def get_config_from_checkpoint(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        return ckpt.get('config')

    # Load dataset
    # We load it once in main usually, but here for simplicity we load inside or 
    # we could refactor to pass it in. To keep signature similar/easy:
    if isinstance(dataset_path, str):
        dataset = torch.load(dataset_path, weights_only=False)
    else:
        dataset = dataset_path # assume it's already loaded tensor dict

    states = dataset['states'].to(device)
    actions = dataset['actions'].to(device)
    rewards = dataset['rewards'].to(device)

    # Load config from checkpoint to ensure we use correct model params
    # Fallback to default if not in checkpoint
    ckpt_config = get_config_from_checkpoint(q_checkpoint_path)
    if not ckpt_config:
        ckpt_config = load_config()

    # Load Q-network
    q_checkpoint = torch.load(q_checkpoint_path, map_location=device, weights_only=False)
    q_net = QNet(ckpt_config).to(device)
    q_net.load_state_dict(q_checkpoint['model_state_dict'])
    q_net.eval()

    # Load V-network
    v_checkpoint = torch.load(v_checkpoint_path, map_location=device, weights_only=False)
    v_net = VNet(ckpt_config).to(device)
    v_net.load_state_dict(v_checkpoint['model_state_dict'])
    v_net.eval()

    # Compute Q, V, and Advantage
    with torch.no_grad():
        batch_size = 4096 # Larger batch for inference
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

    # Write diagnostics to file (APPEND mode)
    with open(output_file, 'a') as f:
        f.write("\n" + "#" * 70 + "\n")
        f.write(f"CHECKPOINT: {checkpoint_name}\n")
        f.write("#" * 70 + "\n")
        f.write(f"  Q-Net: {q_checkpoint_path}\n")
        f.write(f"  V-Net: {v_checkpoint_path}\n\n")

        f.write("-" * 30 + "\n")
        f.write("STATISTICS SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"  Reward Mean:  {rewards_np.mean():.4f}\n")
        f.write(f"  Q Mean:       {q_values.mean():.4f} (Std: {q_values.std():.4f})\n")
        f.write(f"  V Mean:       {v_values.mean():.4f} (Std: {v_values.std():.4f})\n")
        f.write(f"  Adv Mean:     {advantages.mean():.4f} (Std: {advantages.std():.4f})\n")
        f.write(f"  Adv Median:   {np.median(advantages):.4f}\n")
        f.write(f"  Pos Adv Pct:  {positive_adv_pct:.2f}% ({positive_adv_count}/{len(advantages)})\n")

        f.write("\n  Percentiles (Q | V | Adv):\n")
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        for p in percentiles:
            q_p = np.percentile(q_values, p)
            v_p = np.percentile(v_values, p)
            a_p = np.percentile(advantages, p)
            f.write(f"    {p:2d}%: {q_p:8.4f} | {v_p:8.4f} | {a_p:8.4f}\n")

        f.write("\n  Diagnosis:\n")
        if positive_adv_pct < 1:
             f.write("    [PROBLEM] Advantage almost never positive. V may be too high.\n")
        elif positive_adv_pct < 10:
             f.write("    [WARNING] Low positive advantage (<10%). Normal for high tau but check trends.\n")
        elif positive_adv_pct > 50:
             f.write("    [WARNING] High positive advantage (>50%). V may be underestimating.\n")
        else:
             f.write("    [OK] Reasonable positive advantage fraction.\n")
        
    print(f"Processed {checkpoint_name}")


def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Diagnose advantage distribution for all checkpoints")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment folder name in checkpoints/")
    args = parser.parse_args()

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.experiment
    output_path = PROJECT_ROOT / "analysis" / f"advantage_diagnostic_{args.experiment}.txt"
    
    # We load dataset once to pass to function (optimization)
    # Default to continuing dataset if exists, else base stock
    dataset_path_cont = PROJECT_ROOT / "data" / "inv_management_basestock.pt"
    dataset_path_episodic = PROJECT_ROOT / "data" / "inv_management_base_stock.pt"
    
    if dataset_path_cont.exists():
        dataset_path = dataset_path_cont
    elif dataset_path_episodic.exists():
        dataset_path = dataset_path_episodic
    else:
        print("No dataset found.")
        exit(1)
        
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=False)

    # Initialize output file
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"ADVANTAGE DISTRIBUTION DIAGNOSTIC - ALL CHECKPOINTS\n")
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

    # Find all Q checkpoints
    q_dir = checkpoint_dir / "q_net"
    v_dir = checkpoint_dir / "v_net"
    
    if not q_dir.exists():
        print(f"Directory not found: {q_dir}")
        exit(1)

    # Pattern for epoch checkpoints
    epoch_pattern = re.compile(r'checkpoint_epoch_(\d+)\.pth')
    
    checkpoints = []
    
    # 1. Find numbered epochs
    for f in q_dir.glob("checkpoint_epoch_*.pth"):
        match = epoch_pattern.match(f.name)
        if match:
            epoch = int(match.group(1))
            checkpoints.append({
                'epoch': epoch,
                'name': f"Epoch {epoch}",
                'q_path': f,
                'v_path': v_dir / f.name
            })
            
    # Sort by epoch
    checkpoints.sort(key=lambda x: x['epoch'])
    
    # 2. Add best_loss.pth at the end if it exists
    best_q_path = q_dir / "best_loss.pth"
    best_v_path = v_dir / "best_loss.pth"
    if best_q_path.exists():
         checkpoints.append({
                'epoch': 999999, # Sort last
                'name': "Best Loss",
                'q_path': best_q_path,
                'v_path': best_v_path
            })

    print(f"Found {len(checkpoints)} checkpoints.")

    for ckpt in checkpoints:
        if not ckpt['v_path'].exists():
            print(f"Skipping {ckpt['name']} - V-net missing")
            continue
            
        diagnose_advantage(
            str(ckpt['q_path']), 
            str(ckpt['v_path']), 
            dataset, 
            str(output_path),
            ckpt['name']
        )
        
    print(f"\nFull diagnostic written to: {output_path}")

if __name__ == "__main__":
    main()
