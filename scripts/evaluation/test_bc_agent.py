import torch
import numpy as np
import os
import json
import scipy.stats as stats
from utils.config_loader import load_config
from src.base.inv_management_env import InvManagementEnv
from src.models.iql.actor import Actor
from src.base.policies import BaseStockPolicy

ARTIFACTS_DIR = "paper/artifacts"
RESULTS_FILE = os.path.join(ARTIFACTS_DIR, "bc_results.json")

def run_episode(env, actor, device):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_mean, _ = actor(state_tensor)
            action = action_mean.cpu().numpy()[0]
            
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
    return total_reward

def main():
    env = InvManagementEnv(render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    
    # Find the latest BC checkpoint
    checkpoint_root = Path("checkpoints")
    bc_checkpoints = list(checkpoint_root.glob("bc_inv_management_bc_baseline_*"))
    if not bc_checkpoints:
        print("Error: No BC checkpoints found.")
        return
    
    latest_bc = sorted(bc_checkpoints)[-1]
    checkpoint_path = latest_bc / "actor" / "best_loss.pth"
    
    print(f"Loading BC agent from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    actor = Actor(config).to(device)
    actor.load_state_dict(checkpoint['model_state_dict'])
    actor.eval()
    
    NUM_TEST_EPISODES = 100
    print(f"Running BC Agent Evaluation ({NUM_TEST_EPISODES} episodes)...")
    
    bc_rewards = []
    for _ in range(NUM_TEST_EPISODES):
        bc_rewards.append(run_episode(env, actor, device))
        
    bc_mean = np.mean(bc_rewards)
    bc_std = np.std(bc_rewards)
    
    # Load IQL results for comparison
    iql_results_path = os.path.join(ARTIFACTS_DIR, "results.json")
    if os.path.exists(iql_results_path):
        with open(iql_results_path, 'r') as f:
            iql_res = json.load(f)
            iql_mean = iql_res['method']['mean_reward']
            baseline_mean = iql_res['baseline']['mean_reward']
    else:
        iql_mean = 0
        baseline_mean = 0
        
    results_data = {
        "method": {
            "name": "BC Agent (Baseline)",
            "mean_reward": float(bc_mean),
            "std_dev": float(bc_std),
            "n_episodes": NUM_TEST_EPISODES
        },
        "comparison": {
            "vs_iql_diff": float(bc_mean - iql_mean),
            "vs_expert_diff": float(bc_mean - baseline_mean)
        }
    }
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=4)
        
    print("\n" + "="*40)
    print("BC EVALUATION RESULTS")
    print("="*40)
    print(f"BC Agent: {bc_mean:.2f} +/- {bc_std:.2f}")
    print(f"IQL Agent: {iql_mean:.2f}")
    print(f"Best Expert: {baseline_mean:.2f}")
    print("-" * 40)
    print(f"IQL vs BC Improvement: {iql_mean - bc_mean:.2f}")
    print("="*40)

if __name__ == "__main__":
    from pathlib import Path
    main()
