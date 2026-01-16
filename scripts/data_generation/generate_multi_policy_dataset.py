"""
Generate Multi-Policy Dataset for Inventory Control

Generates data using multiple policies with their OPTIMAL parameters:
- Base-Stock (refined)
- Min-Max (s,S) (refined)
- Lot-for-Lot (refined)
- Periodic Review (T,S)
- (R,Q) Fixed Quantity
- Noisy Base-Stock

This creates diverse training data for offline RL.
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
from src.base.policies import (
    BaseStockPolicy, MinMaxPolicy, RQPolicy,
    PeriodicReviewPolicy, LotForLotPolicy, NoisyBaseStockPolicy
)
from utils.config_loader import load_config


# ==================== OPTIMAL PARAMETERS (from grid search) ====================

# Base-Stock: z0=70, z1=170, z2=15 (profit: 346.11)
BASESTOCK_CONFIGS = [
    [70, 170, 15],
    [70, 170, 10],
    [80, 190, 10],
    [70, 160, 15],
    [80, 170, 10],
]

# Min-Max: s0=65, S0=70, s1=160, S1=165, s2=0, S2=60 (profit: 349.85)
MINMAX_CONFIGS = [
    [[65, 70], [160, 165], [0, 60]],
    [[70, 75], [160, 165], [0, 60]],
    [[70, 75], [170, 175], [0, 55]],
    [[65, 70], [165, 170], [0, 45]],
    [[65, 70], [165, 175], [0, 40]],
]

# Lot-for-Lot: d0=44, d1=20, d2=1 (profit: 350.83)
LOTFORLOT_CONFIGS = [
    [44, 20, 1],
    [46, 20, 1],
    [47, 21, 1],
    [43, 19, 1],
    [45, 20, 1],
]

# Periodic Review: T=2, S=[100, 200, 50] (profit: 363.51)
PERIODIC_CONFIGS = [
    (2, [100, 200, 50]),
    (3, [150, 250, 50]),
    (6, [200, 250, 50]),
    (6, [200, 300, 50]),
    (3, [100, 200, 50]),
]

# RQ: R0=80, Q0=20, R1=200, Q1=20, R2=0, Q2=5 (profit: 431.44)
RQ_CONFIGS = [
    [[80, 20], [200, 20], [0, 5]],
    [[80, 20], [200, 20], [0, 15]],
    [[80, 20], [200, 20], [0, 10]],
    [[90, 20], [180, 20], [15, 5]],
    [[100, 20], [180, 25], [0, 5]],
]

# Noisy Base-Stock: z=[80,180,40], noise_std=15 (profit: 363.66)
NOISY_BASESTOCK_CONFIGS = [
    ([80, 180, 40], 15),
    ([80, 180, 40], 10),
    ([80, 180, 40], 5),
    ([80, 200, 40], 5),
    ([100, 200, 100], 20),
]


class LotForLotPolicyPerStage:
    """Lot-for-Lot with per-stage demand."""
    def __init__(self, env, demand_per_stage):
        self.env = env
        self.demand_per_stage = np.array(demand_per_stage)

    def get_action(self):
        period = self.env.period
        num_stages = len(self.env.I[0]) if len(self.env.I) > 0 else 3
        action = self.demand_per_stage.copy().astype(np.float32)

        if period >= len(self.env.I):
            current_I = self.env.I[-1]
        else:
            current_I = self.env.I[period]

        for i in range(num_stages):
            if current_I[i] > self.demand_per_stage[i]:
                action[i] = max(0, self.demand_per_stage[i] - 
                               (current_I[i] - self.demand_per_stage[i]))
        return action.astype(np.float32)


def generate_multi_policy_dataset(
    num_warehouses: int,
    days_per_warehouse: int,
    save_path: str
) -> None:
    """
    Generates a dataset using multiple policies with optimal parameters.
    
    Each warehouse uses a different policy configuration for diversity.
    """
    print(f"Generating Multi-Policy Dataset")
    print(f"  Warehouses: {num_warehouses}")
    print(f"  Days per warehouse: {days_per_warehouse}")
    print(f"  Total samples: {num_warehouses * days_per_warehouse}")
    print(f"  Policies: Base-Stock, Min-Max, Lot-for-Lot, Periodic, RQ, Noisy")

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []

    # Policy types to cycle through
    policy_types = ['basestock', 'minmax', 'lotforlot', 'periodic', 'rq', 'noisy']

    for warehouse_id in tqdm.tqdm(range(num_warehouses), desc="Warehouses"):
        env = InvManagementEnv(
            max_steps_per_episode=days_per_warehouse + 1,
            render_mode=None
        )
        
        # Select policy type for this warehouse
        policy_type = policy_types[warehouse_id % len(policy_types)]
        config_idx = (warehouse_id // len(policy_types)) % 5
        
        # Create policy based on type
        if policy_type == 'basestock':
            z = BASESTOCK_CONFIGS[config_idx]
            policy = BaseStockPolicy(env, z=z)
        elif policy_type == 'minmax':
            params = MINMAX_CONFIGS[config_idx]
            policy = MinMaxPolicy(env, min_max_params=params)
        elif policy_type == 'lotforlot':
            demand = LOTFORLOT_CONFIGS[config_idx]
            policy = LotForLotPolicyPerStage(env, demand)
        elif policy_type == 'periodic':
            T, S = PERIODIC_CONFIGS[config_idx]
            policy = PeriodicReviewPolicy(env, review_period=T, S_levels=S)
        elif policy_type == 'rq':
            params = RQ_CONFIGS[config_idx]
            policy = RQPolicy(env, rq_params=params)
        else:  # noisy
            z, noise_std = NOISY_BASESTOCK_CONFIGS[config_idx]
            policy = NoisyBaseStockPolicy(env, z=z, noise_std=noise_std)

        obs, info = env.reset(seed=warehouse_id * 1000)
        current_state = obs

        for day in range(days_per_warehouse):
            action = policy.get_action()

            # Add small exploration noise
            noise = np.random.normal(0, 3, size=action.shape)
            action = action + noise
            action = np.clip(action, 0, env.supply_capacity)

            next_obs, reward, done, _, info = env.step(action)

            all_states.append(current_state)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_obs)
            all_dones.append(0.0)  # Continuing task

            current_state = next_obs

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


if __name__ == "__main__":
    config = load_config()
    
    num_warehouses = config.get('environment', {}).get('num_warehouses', 10)
    days_per_warehouse = config.get('environment', {}).get('days_per_warehouse', 365)
    
    save_path = str(PROJECT_ROOT / "data" / "inv_management_multi_policy.pt")
    
    generate_multi_policy_dataset(num_warehouses, days_per_warehouse, save_path)
