"""
Generate Continuing Task Dataset for Inventory Control

Generates data from multiple warehouses over extended time periods,
treating it as a continuing task (no artificial episode resets).

This is more realistic: 10 warehouses × 365 days = 3,650 samples.
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
from src.base.policies import BaseStockPolicy
from utils.config_loader import load_config


def generate_continuing_dataset(
    num_warehouses: int,
    days_per_warehouse: int,
    save_path: str
) -> None:
    """
    Generates a continuing task dataset from multiple warehouses.
    
    Each warehouse runs for days_per_warehouse steps without reset.
    The 'done' flag is 0 for all samples (continuing task).
    
    Args:
        num_warehouses: Number of independent warehouse trajectories.
        days_per_warehouse: Number of days per warehouse.
        save_path: Path to save the dataset.
    """
    # Top base-stock configurations from grid search
    top_configs = [
        [80, 180, 40],
        [80, 200, 40],
        [80, 200, 300],
        [80, 160, 40],
        [80, 180, 300],
        [80, 180, 80],
        [80, 180, 60],
        [60, 140, 40],
        [80, 200, 60],
        [80, 200, 280]
    ]

    print(f"Generating Continuing Task Dataset")
    print(f"  Warehouses: {num_warehouses}")
    print(f"  Days per warehouse: {days_per_warehouse}")
    print(f"  Total samples: {num_warehouses * days_per_warehouse}")

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []

    for warehouse_id in tqdm.tqdm(range(num_warehouses), desc="Warehouses"):
        # Each warehouse uses a different demand seed for diversity
        env = InvManagementEnv(
            max_steps_per_episode=days_per_warehouse + 1,  # +1 to avoid done at boundary
            render_mode=None
        )
        
        # Assign a random top configuration to this warehouse
        config_idx = warehouse_id % len(top_configs)
        current_z = np.array(top_configs[config_idx])
        policy = BaseStockPolicy(env, z=current_z)

        obs, info = env.reset(seed=warehouse_id * 1000)  # Different seed per warehouse
        current_state = obs

        for day in range(days_per_warehouse):
            action = policy.get_action()

            # Add small noise for exploration
            noise = np.random.normal(0, 5, size=action.shape)
            action = action + noise
            action = np.clip(action, 0, env.supply_capacity)

            next_obs, reward, done, _, info = env.step(action)

            all_states.append(current_state)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_obs)
            # Key difference: done is always 0 for continuing task
            all_dones.append(0.0)

            current_state = next_obs

            # If environment happens to terminate, reset and continue
            if done:
                obs, info = env.reset(seed=warehouse_id * 1000 + day)
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
    
    print(f"\nDataset saved to {save_path}")
    print(f"  Total samples: {len(dataset['states'])}")
    print(f"  Average step reward: {dataset['rewards'].mean().item():.4f}")
    print(f"  Done flags (should be all 0): unique={torch.unique(dataset['dones']).tolist()}")


if __name__ == "__main__":
    config = load_config()
    
    num_warehouses = config.get('environment', {}).get('num_warehouses', 10)
    days_per_warehouse = config.get('environment', {}).get('days_per_warehouse', 365)
    
    save_path = str(PROJECT_ROOT / "data" / "inv_management_continuing.pt")
    
    generate_continuing_dataset(num_warehouses, days_per_warehouse, save_path)
