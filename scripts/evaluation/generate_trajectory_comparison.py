import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from src.base.inv_management_env import InvManagementEnv
from src.models.iql.actor import Actor
from src.base.policies import BaseStockPolicy
from utils.config_loader import load_config

# Constants
ARTIFACTS_DIR = "paper/artifacts"
OUTPUT_PDF = os.path.join(ARTIFACTS_DIR, "inventory_comparison.pdf")
OUTPUT_PNG = os.path.join(ARTIFACTS_DIR, "inventory_comparison.png")
CHECKPOINT_PATH = "checkpoints/inv_management_iql_minmax_run_18022026_190530/actor/best_loss.pth"

# The true global optimum baseline
BEST_EXPERT_PARAMS = [70, 170, 15]

def run_trajectory(env, policy_type='iql', actor=None, seed=42):
    """Runs a single episode and returns inventory trajectories."""
    obs, _ = env.reset(seed=seed)
    done = False
    
    # Stages: Retailer, Distributor, Manufacturer
    inventory_history = []
    demand_history = []
    
    # Initial inventory
    inventory_history.append(env.init_inv.copy())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if policy_type == 'expert':
        policy = BaseStockPolicy(env, z=BEST_EXPERT_PARAMS)
    
    while not done:
        if policy_type == 'iql':
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean, _ = actor(state_tensor)
                action = action_mean.cpu().numpy()[0]
        else:
            action = policy.get_action()
            
        obs, reward, done, _, info = env.step(action)
        inventory_history.append(info['inventory'].copy())
        demand_history.append(info['demand'])
        
    return np.array(inventory_history), np.array(demand_history)

def main():
    print("Generating Inventory Trajectory Comparison...")
    env = InvManagementEnv(render_mode=None)
    
    # 1. Load IQL Actor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = checkpoint.get('config', load_config())
    actor = Actor(config).to(device)
    actor.load_state_dict(checkpoint['model_state_dict'])
    actor.eval()
    
    # 2. Run Trajectories (using same seed for identical demand)
    TEST_SEED = 12345
    print(f"Running trajectories with seed {TEST_SEED}...")
    iql_inv, demand = run_trajectory(env, policy_type='iql', actor=actor, seed=TEST_SEED)
    expert_inv, _ = run_trajectory(env, policy_type='expert', seed=TEST_SEED)
    
    # 3. Plotting
    print("Generating plots...")
    stages = ['Retailer', 'Distributor', 'Manufacturer']
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    time_steps = np.arange(len(iql_inv))
    
    for i in range(3):
        ax = axes[i]
        ax.plot(time_steps, expert_inv[:, i], label='Best Expert (Baseline)', 
                color='steelblue', linestyle='--', linewidth=2)
        ax.plot(time_steps, iql_inv[:, i], label='IQL Agent (Ours)', 
                color='crimson', linewidth=2.5)
        
        ax.set_title(f'Echelon {i}: {stages[i]} Inventory Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Units', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper right')
            
    axes[2].set_xlabel('Time Period (Days)', fontsize=12)
    
    plt.tight_layout()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PDF)
    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"Trajectory comparison saved to {OUTPUT_PDF} and {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
