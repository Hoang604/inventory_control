"""
Generate Min-Max (s,S) Policy Dataset

Uses optimal parameters from grid search: s=[65,160,0], S=[70,165,60]
"""
import os
from pathlib import Path
import torch
import numpy as np
import tqdm
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.base.inv_management_env import InvManagementEnv
from src.base.policies import MinMaxPolicy
from utils.config_loader import load_config

# Optimal configs from grid search (profit: 349.85)
CONFIGS = [
    [[65, 70], [160, 165], [0, 60]],
    [[70, 75], [160, 165], [0, 60]],
    [[70, 75], [170, 175], [0, 55]],
    [[65, 70], [165, 170], [0, 45]],
    [[65, 70], [165, 175], [0, 40]],
]


def generate_dataset(num_warehouses: int, days_per_warehouse: int, save_path: str) -> None:
    print(f"Generating Min-Max Dataset")
    print(f"  Warehouses: {num_warehouses}, Days: {days_per_warehouse}")
    print(f"  Total samples: {num_warehouses * days_per_warehouse}")

    all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []

    for warehouse_id in tqdm.tqdm(range(num_warehouses), desc="Warehouses"):
        env = InvManagementEnv(max_steps_per_episode=days_per_warehouse + 1, render_mode=None)
        
        config = CONFIGS[warehouse_id % len(CONFIGS)]
        policy = MinMaxPolicy(env, min_max_params=config)

        obs, _ = env.reset(seed=warehouse_id * 1000)
        current_state = obs

        for day in range(days_per_warehouse):
            action = policy.get_action()
            noise = np.random.normal(0, 5, size=action.shape)
            action = np.clip(action + noise, 0, env.supply_capacity)

            next_obs, reward, done, _, _ = env.step(action)

            all_states.append(current_state)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_obs)
            all_dones.append(0.0)

            current_state = next_obs
            if done:
                obs, _ = env.reset(seed=warehouse_id * 1000 + day)
                current_state = obs

    dataset = {
        'states': torch.tensor(np.array(all_states), dtype=torch.float32),
        'actions': torch.tensor(np.array(all_actions), dtype=torch.float32),
        'rewards': torch.tensor(np.array(all_rewards), dtype=torch.float32).unsqueeze(1),
        'next_states': torch.tensor(np.array(all_next_states), dtype=torch.float32),
        'dones': torch.tensor(np.array(all_dones), dtype=torch.float32).unsqueeze(1)
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, save_path)
    print(f"\nSaved to {save_path}")
    print(f"  Samples: {len(dataset['states'])}, Avg reward: {dataset['rewards'].mean():.4f}")


if __name__ == "__main__":
    config = load_config()
    num_warehouses = config.get('environment', {}).get('num_warehouses', 10)
    days_per_warehouse = config.get('environment', {}).get('days_per_warehouse', 3650)
    save_path = str(PROJECT_ROOT / "data" / "inv_management_minmax.pt")
    generate_dataset(num_warehouses, days_per_warehouse, save_path)
