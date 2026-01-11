import os
from pathlib import Path

import torch
import numpy as np
import tqdm
import os
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import BaseStockPolicy

def generate_base_stock_dataset(num_episodes: int, steps_per_episode: int, save_path: str):
    """
    Generates a dataset using a MIXTURE of High-Performance Base-Stock Policies.

    Instead of a single fixed strategy, we randomly select one of the top 10 
    configurations found via Grid Search. This creates a "Mixture of Experts" 
    dataset that is both high-quality (mean score ~330) and diverse.
    """
    env = InvManagementEnv(
        max_steps_per_episode=steps_per_episode, render_mode=None)

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

    policy = BaseStockPolicy(env, z=top_configs[0])

    print(
        f"Generating dataset with Mixture of Top {len(top_configs)} Base Stock Policies")
    print(f"Episodes: {num_episodes}")

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []

    for episode in tqdm.tqdm(range(num_episodes)):
        obs, info = env.reset()
        current_state = obs

        config_idx = np.random.randint(0, len(top_configs))
        current_z = np.array(top_configs[config_idx])

        policy.z = current_z

        for step in range(steps_per_episode):
            action = policy.get_action()

            noise = np.random.normal(0, 5, size=action.shape)
            action = action + noise
            action = np.clip(action, 0, env.supply_capacity)

            next_obs, reward, done, _, info = env.step(action)

            all_states.append(current_state)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_obs)
            all_dones.append(float(done))

            current_state = next_obs

            if done:
                break

    dataset = {
        'states': torch.tensor(np.array(all_states), dtype=torch.float32),
        'actions': torch.tensor(np.array(all_actions), dtype=torch.float32),
        'rewards': torch.tensor(np.array(all_rewards), dtype=torch.float32).unsqueeze(1),
        'next_states': torch.tensor(np.array(all_next_states), dtype=torch.float32),
        'dones': torch.tensor(np.array(all_dones), dtype=torch.float32).unsqueeze(1)
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, save_path)
    print(f"Mixture of Experts Dataset saved to {save_path}")
    print(
        f"Average Step Reward in Dataset: {dataset['rewards'].mean().item():.4f}")

if __name__ == "__main__":
    NUM_EPISODES = 2000
    STEPS_PER_EPISODE = 30
    DATASET_DIR = "data"
    DATASET_FILENAME = "inv_management_base_stock.pt"
    SAVE_PATH = os.path.join(DATASET_DIR, DATASET_FILENAME)

    generate_base_stock_dataset(NUM_EPISODES, STEPS_PER_EPISODE, SAVE_PATH)
