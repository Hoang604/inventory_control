"""
Generate Inventory Trajectory Comparison Plot

This script generates a plot comparing inventory levels over time for:
1. IQL Agent (trained on mixture of experts)
2. Best Expert (Base Stock Policy z=[80, 180, 40])

Outputs:
- paper/artifacts/inventory_trajectory.csv (raw data)
- paper/artifacts/inventory_comparison.png (publication figure)
- paper/artifacts/inventory_comparison.pdf (high-res version)
"""

import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.config_loader import load_config
from src.base.inv_management_env import InvManagementEnv
from src.models.iql.actor import Actor
from src.base.policies import BaseStockPolicy

# Artifact paths
ARTIFACTS_DIR = "paper/artifacts"
TRAJECTORY_CSV = os.path.join(ARTIFACTS_DIR, "inventory_trajectory.csv")
PLOT_PNG = os.path.join(ARTIFACTS_DIR, "inventory_comparison.png")
PLOT_PDF = os.path.join(ARTIFACTS_DIR, "inventory_comparison.pdf")

def run_episode_with_tracking(env, agent=None, baseline_policy_callable=None, seed=42):
    """
    Run a single episode and track inventory/demand history.
    
    Args:
        env: The environment instance
        agent: The IQL agent (Actor network), or None for baseline
        baseline_policy_callable: Callable for baseline policy
        seed: Random seed for reproducibility
        
    Returns:
        total_reward: Cumulative reward
        inventory_history: List of inventory levels at each timestep (shape: [T, num_echelons])
        demand_history: List of demand values at each timestep
    """
    env.reset(seed=seed)
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0
    step = 0
    
    inventory_history = []
    demand_history = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    while not done:
        if agent is None:
            if baseline_policy_callable:
                action = baseline_policy_callable(obs)
            else:
                action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean, _ = agent(state_tensor)
                action = action_mean.cpu().numpy()[0]
        
        next_obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        # Track inventory and demand
        inventory_history.append(info['inventory'].copy())
        demand_history.append(info['demand'])
        
        obs = next_obs
        step += 1
    
    return total_reward, np.array(inventory_history), np.array(demand_history)

def main():
    """Main execution function."""
    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Create environment
    env = InvManagementEnv(render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load IQL Agent
    checkpoint_path = "/home/hoang/python/inventory_control/checkpoints/inv_management_iql_minmax_run_14122025_225614/actor/checkpoint_epoch_99.pth"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"IQL checkpoint not found at {checkpoint_path}")
    
    print(f"Loading IQL agent from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("Warning: Config not found in checkpoint. Using default config.")
        config = load_config()
    
    actor = Actor(config).to(device)
    actor.load_state_dict(checkpoint['model_state_dict'])
    actor.eval()
    
    # Setup Best Expert Policy
    target_levels = np.array([80, 180, 40])
    base_stock_policy = BaseStockPolicy(env, z=target_levels)
    
    def base_stock_callable(obs_val):
        return base_stock_policy.get_action()
    
    # Run deterministic episode for Best Expert
    print(f"\nRunning Best Expert (z={target_levels}) with seed=42...")
    expert_reward, expert_inventory, expert_demand = run_episode_with_tracking(
        env, agent=None, baseline_policy_callable=base_stock_callable, seed=42
    )
    
    # Run deterministic episode for IQL Agent
    print(f"Running IQL Agent with seed=42...")
    iql_reward, iql_inventory, iql_demand = run_episode_with_tracking(
        env, agent=actor, baseline_policy_callable=None, seed=42
    )
    
    print(f"\n--- Single Episode Results ---")
    print(f"Best Expert Reward: {expert_reward:.2f}")
    print(f"IQL Agent Reward:   {iql_reward:.2f}")
    print(f"Difference:         {iql_reward - expert_reward:.2f}")
    
    # Prepare data for CSV export
    num_timesteps = len(expert_inventory)
    num_echelons = expert_inventory.shape[1]
    
    data_rows = []
    
    for t in range(num_timesteps):
        for echelon_idx in range(num_echelons):
            # Expert row
            data_rows.append({
                'timestep': t + 1,
                'echelon': echelon_idx,
                'policy': 'Best Expert',
                'inventory_level': expert_inventory[t, echelon_idx],
                'demand': expert_demand[t]
            })
            # IQL row
            data_rows.append({
                'timestep': t + 1,
                'echelon': echelon_idx,
                'policy': 'IQL Agent',
                'inventory_level': iql_inventory[t, echelon_idx],
                'demand': iql_demand[t]
            })
    
    df = pd.DataFrame(data_rows)
    df.to_csv(TRAJECTORY_CSV, index=False)
    print(f"\nTrajectory data saved to: {TRAJECTORY_CSV}")
    
    # Generate plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Inventory Trajectory Comparison: IQL Agent vs Best Expert', 
                 fontsize=16, fontweight='bold')
    
    timesteps = np.arange(1, num_timesteps + 1)
    echelon_names = ['Retailer (Echelon 0)', 'Warehouse (Echelon 1)', 'Factory (Echelon 2)']
    
    for echelon_idx in range(num_echelons):
        ax = axes[echelon_idx]
        
        # Plot inventory levels
        ax.plot(timesteps, expert_inventory[:, echelon_idx], 
                label='Best Expert', color='#FF6B35', linewidth=2, marker='o', markersize=4)
        ax.plot(timesteps, iql_inventory[:, echelon_idx], 
                label='IQL Agent', color='#004E89', linewidth=2, marker='s', markersize=4)
        
        # Add demand as background reference (only for Echelon 0)
        if echelon_idx == 0:
            ax2 = ax.twinx()
            ax2.bar(timesteps, expert_demand, alpha=0.2, color='gray', 
                   label='Customer Demand', width=0.6)
            ax2.set_ylabel('Demand', fontsize=10, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.legend(loc='upper right', fontsize=9)
        
        ax.set_ylabel('Inventory Level', fontsize=11, fontweight='bold')
        ax.set_title(echelon_names[echelon_idx], fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, num_timesteps + 1)
    
    axes[-1].set_xlabel('Timestep', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figures
    plt.savefig(PLOT_PNG, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {PLOT_PNG}")
    
    plt.savefig(PLOT_PDF, bbox_inches='tight')
    print(f"High-res plot saved to: {PLOT_PDF}")
    
    print("\n" + "="*60)
    print("TRAJECTORY COMPARISON COMPLETE")
    print("="*60)
    print(f"Artifacts exported to: {ARTIFACTS_DIR}/")
    print("  - inventory_trajectory.csv")
    print("  - inventory_comparison.png")
    print("  - inventory_comparison.pdf")
    print("="*60)

if __name__ == "__main__":
    main()
