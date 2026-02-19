"""Advantage Distribution Diagnostic - All Checkpoints

Loads trained Q-net and V-net checkpoints and analyzes the advantage distribution
across the dataset to diagnose training issues for specific checkpoints.

Output: 
- analysis/advantage_diagnostic_<experiment>.txt
- analysis/advantage_evolution_<experiment>.png
"""
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def diagnose_advantage(
    q_checkpoint_path: str,
    v_checkpoint_path: str,
    dataset_path: str,
    output_file: str,
    checkpoint_name: str
) -> dict:
    """
    Loads Q and V networks, computes advantage over the dataset, and APPENDS
    diagnostic information to a file. Returns metrics for plotting.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from src.models.iql.critics import QNet, VNet
    from utils.config_loader import load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_config_from_checkpoint(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        return ckpt.get('config')

    # Load dataset
    if isinstance(dataset_path, str):
        dataset = torch.load(dataset_path, weights_only=False)
    else:
        dataset = dataset_path

    states = dataset['states'].to(device)
    actions = dataset['actions'].to(device)
    rewards = dataset['rewards'].to(device)

    ckpt_config = get_config_from_checkpoint(q_checkpoint_path)
    if not ckpt_config:
        ckpt_config = load_config()

    # Load networks
    q_checkpoint = torch.load(q_checkpoint_path, map_location=device, weights_only=False)
    q_net = QNet(ckpt_config).to(device)
    q_net.load_state_dict(q_checkpoint['model_state_dict'])
    q_net.eval()

    v_checkpoint = torch.load(v_checkpoint_path, map_location=device, weights_only=False)
    v_net = VNet(ckpt_config).to(device)
    v_net.load_state_dict(v_checkpoint['model_state_dict'])
    v_net.eval()

    # Compute Q, V, and Advantage
    with torch.no_grad():
        batch_size = 4096
        all_q, all_v, all_adv = [], [], []

        for i in range(0, len(states), batch_size):
            s, a = states[i:i+batch_size], actions[i:i+batch_size]
            q_val, v_val = q_net(s, a), v_net(s)
            adv = q_val - v_val
            all_q.append(q_val.cpu()); all_v.append(v_val.cpu()); all_adv.append(adv.cpu())

        advantages = torch.cat(all_adv).numpy().flatten()
        rewards_np = rewards.cpu().numpy().flatten()

    # Normality Analysis
    adv_mean, adv_std = advantages.mean(), advantages.std()
    adv_skew, adv_kurt = stats.skew(advantages), stats.kurtosis(advantages)
    ks_stat, _ = stats.kstest(advantages, 'norm', args=(adv_mean, adv_std))

    # Weight Analysis
    beta = ckpt_config['iql'].get('beta', 1.0)
    adv_clip = ckpt_config['training'].get('adv_weight_clip', 100.0)
    weights = np.exp(beta * advantages)
    weights = np.clip(weights, a_min=None, a_max=adv_clip)
    
    # ESS
    ess = (np.sum(weights)**2) / np.sum(weights**2)
    ess_pct = (ess / len(weights)) * 100
    
    positive_adv_pct = (advantages > 0).sum() / len(advantages) * 100

    # Write to file
    with open(output_file, 'a') as f:
        f.write(f"\n{'#'*70}\nCHECKPOINT: {checkpoint_name}\n{'#'*70}\n")
        f.write(f"  Adv Mean: {adv_mean:.4f} (Std: {adv_std:.4f})\n")
        f.write(f"  Normality: KS-Stat={ks_stat:.4f}, Skew={adv_skew:.4f}\n")
        f.write(f"  ESS: {ess:.2f} ({ess_pct:.2f}%)\n")

    print(f"Processed {checkpoint_name}")
    return {
        'name': checkpoint_name, 'advantages': advantages, 'weights': weights,
        'beta': beta, 'adv_clip': adv_clip, 'ess_pct': ess_pct,
        'pos_adv_pct': positive_adv_pct, 'adv_mean': adv_mean, 'adv_std': adv_std,
        'adv_skew': adv_skew, 'adv_kurt': adv_kurt, 'ks_stat': ks_stat
    }

def plot_evolution(results, output_path):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(4, 2)
    colors = sns.color_palette("rocket", n_colors=len(results))
    percentiles = [50, 75, 90, 95]

    # 1. Advantage Distribution
    ax1 = fig.add_subplot(gs[0, :])
    for i, res in enumerate(results):
        sns.kdeplot(res['advantages'], ax=ax1, label=res['name'], color=colors[i], fill=True, alpha=0.1)
        p_vals = np.percentile(res['advantages'], percentiles)
        for p, val in zip(percentiles, p_vals):
            ax1.axvline(val, color=colors[i], linestyle=':', alpha=0.5)
            if i == len(results) - 1:
                ax1.text(val, ax1.get_ylim()[1]*0.95, f'{p}%: {val:.2f}', color=colors[i], 
                         rotation=90, ha='right', va='top', fontsize=9, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("Advantage Distribution Evolution", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')

    # 2. Weight Distribution (Log Scale)
    ax2 = fig.add_subplot(gs[1, :])
    for i, res in enumerate(results):
        w_plot = np.clip(res['weights'], 1e-5, None)
        sns.kdeplot(w_plot, ax=ax2, color=colors[i], label=res['name'], fill=True, alpha=0.05, log_scale=True)
        p_vals_w = np.percentile(res['weights'], percentiles)
        for p, val in zip(percentiles, p_vals_w):
            ax2.axvline(val, color=colors[i], linestyle=':', alpha=0.5)
            if i == len(results) - 1:
                ax2.text(val, ax2.get_ylim()[1]*0.95, f'{p}%: {val:.2f}', color=colors[i], 
                         rotation=90, ha='right', va='top', fontsize=9, fontweight='bold')
    ax2.axvline(x=1.0, color='black', linestyle=':', alpha=0.5, label='Weight=1 (BC)')
    clip_limit = results[-1]['adv_clip']
    ax2.axvline(x=clip_limit, color='red', linestyle='--', alpha=0.5, label=f'Clip ({clip_limit})')
    ax2.set_xlim(1e-4, clip_limit * 2)
    ax2.set_title("Training Weight Distribution (Log Scale)", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')

    # 3. Mean & Std Dev
    ax3 = fig.add_subplot(gs[2, 0])
    epochs = range(len(results))
    ax3.plot(epochs, [r['adv_mean'] for r in results], label='Mean', marker='o')
    ax3.plot(epochs, [r['adv_std'] for r in results], label='Std Dev', marker='x')
    ax3.set_title("Mean & Std Dev Evolution"); ax3.legend()

    # 4. Normality (KS-Stat)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(epochs, [r['ks_stat'] for r in results], color='purple', marker='d')
    ax4.set_title("Normality Score (Lower = More Normal)")

    # 5. ESS %
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.plot(epochs, [r['ess_pct'] for r in results], color='crimson', marker='o')
    ax5.set_ylim(0, 105); ax5.set_title("Selectivity Trend (ESS %)")

    # 6. Skewness & Kurtosis
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.plot(epochs, [r['adv_skew'] for r in results], label='Skew', marker='^')
    ax6.plot(epochs, [r['adv_kurt'] for r in results], label='Kurt', marker='v')
    ax6.set_title("Shape Evolution"); ax6.legend()

    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Evolution plot saved to: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.experiment
    output_text = PROJECT_ROOT / "analysis" / f"advantage_diagnostic_{args.experiment}.txt"
    output_plot = PROJECT_ROOT / "analysis" / f"advantage_evolution_{args.experiment}.png"
    
    dataset_path = PROJECT_ROOT / "data" / "inv_management_basestock.pt"
    if not dataset_path.exists(): dataset_path = PROJECT_ROOT / "data" / "inv_management_base_stock.pt"
    dataset = torch.load(dataset_path, weights_only=False)

    with open(output_text, 'w') as f:
        f.write(f"ADVANTAGE DIAGNOSTIC: {args.experiment}\n{'='*70}\n")

    q_dir, v_dir = checkpoint_dir / "q_net", checkpoint_dir / "v_net"
    target_epochs = []
    checkpoints = []
    for f in q_dir.glob("checkpoint_epoch_*.pth"):
        match = re.match(r'checkpoint_epoch_(\d+)\.pth', f.name)
        if match and int(match.group(1)) in target_epochs:
            checkpoints.append({'epoch': int(match.group(1)), 'name': f"Epoch {match.group(1)}", 'q_path': f, 'v_path': v_dir / f.name})
    
    checkpoints.sort(key=lambda x: x['epoch'])
    if (q_dir / "best_loss.pth").exists():
        checkpoints.append({'epoch': 999999, 'name': "Best Loss", 'q_path': q_dir / "best_loss.pth", 'v_path': v_dir / "best_loss.pth"})

    all_results = [diagnose_advantage(str(c['q_path']), str(c['v_path']), dataset, str(output_text), c['name']) for c in checkpoints if c['v_path'].exists()]
    if all_results: plot_evolution(all_results, str(output_plot))

if __name__ == "__main__":
    main()
