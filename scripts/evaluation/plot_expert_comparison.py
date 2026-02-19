import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import BaseStockPolicy

# Constants
ARTIFACTS_DIR = "paper/artifacts"
RESULTS_FILE = os.path.join(ARTIFACTS_DIR, "results.json")
OUTPUT_PDF = os.path.join(ARTIFACTS_DIR, "expert_comparison.pdf")
OUTPUT_PNG = os.path.join(ARTIFACTS_DIR, "expert_comparison.png")

# The 5 experts used in generate_basestock_dataset.py
TRAINING_EXPERTS = [
    [70, 170, 15],
    [70, 170, 10],
    [80, 190, 10],
    [70, 160, 15],
    [80, 170, 10],
]

# The global optimum baseline used in the paper
BASELINE_EXPERT = [70, 170, 15]

def evaluate_policy(env, z_params, n_episodes=100):
    """Evaluates a Base-Stock policy for a given number of episodes."""
    rewards = []
    for i in range(n_episodes):
        policy = BaseStockPolicy(env, z=z_params)
        obs, _ = env.reset(seed=i + 10000) # Use a different seed range for testing
        done = False
        total_reward = 0
        while not done:
            action = policy.get_action()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

def main():
    print("Starting Expert Comparison Evaluation...")
    env = InvManagementEnv(render_mode=None)
    
    expert_results = []
    
    # 1. Evaluate Training Experts
    for i, z in enumerate(TRAINING_EXPERTS):
        print(f"Evaluating Training Expert {i+1}: z={z}...")
        mean, std = evaluate_policy(env, z)
        expert_results.append({
            'name': f'Expert {i+1}',
            'mean': mean,
            'std': std,
            'color': 'gray',
            'alpha': 0.6
        })
        
    # 2. Evaluate Baseline Expert
    print(f"Evaluating Baseline Expert: z={BASELINE_EXPERT}...")
    b_mean, b_std = evaluate_policy(env, BASELINE_EXPERT)
    expert_results.append({
        'name': 'Best Expert\n(Baseline)',
        'mean': b_mean,
        'std': b_std,
        'color': 'steelblue',
        'alpha': 0.9
    })
    
    # 3. Load IQL Result
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            res = json.load(f)
            iql_mean = res['method']['mean_reward']
            iql_std = res['method']['std_dev']
            expert_results.append({
                'name': 'IQL Agent\n(Ours)',
                'mean': iql_mean,
                'std': iql_std,
                'color': 'crimson',
                'alpha': 1.0
            })
    else:
        print("Warning: results.json not found. Skipping IQL Agent in plot.")

    # 4. Plotting
    print("Generating plot...")
    names = [r['name'] for r in expert_results]
    means = [r['mean'] for r in expert_results]
    stds = [r['std'] for r in expert_results]
    colors = [r['color'] for r in expert_results]
    alphas = [r['alpha'] for r in expert_results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, means, yerr=stds, color=colors, capsize=5, edgecolor='black', linewidth=1)
    
    # Apply alphas
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
        
    plt.ylabel('Mean Episode Reward (Profit)', fontsize=12)
    plt.title('Breaking the Heuristic Ceiling: IQL vs. Expert Mixture', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF)
    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"Comparison plot saved to {OUTPUT_PDF} and {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
