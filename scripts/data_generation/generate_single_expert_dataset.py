"""
Generate Single-Expert Dataset for Ablation Study

This script generates a dataset using ONLY the single best expert configuration
(Base-Stock Policy with z=[80, 180, 40]), without any mixture or diversity.

Purpose: To test the "Implicit Synthesis" hypothesis by comparing IQL trained on:
  - Mixture of diverse experts (original dataset)
  - Single best expert (this dataset)

Output: data/inv_management_single_expert.pt
"""
import os
import torch
import numpy as np
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import BaseStockPolicy


def generate_single_expert_dataset(num_episodes: int, steps_per_episode: int, save_path: str):
    """
    Generates a dataset using a SINGLE High-Performance Base-Stock Policy.

    This is the control condition for the ablation study. We use the global
    optimum from the grid search: z=[80, 180, 40].

    Args:
        num_episodes: Number of episodes to generate
        steps_per_episode: Maximum steps per episode
        save_path: Path to save the dataset
    """
    env = InvManagementEnv(
        max_steps_per_episode=steps_per_episode, render_mode=None)

    # The single best configuration from grid search
    best_config = [80, 180, 40]

    policy = BaseStockPolicy(env, z=best_config)

    print("="*60)
    print("GENERATING SINGLE-EXPERT DATASET (Ablation Study)")
    print("="*60)
    print(f"Configuration: Base-Stock Policy with z={best_config}")
    print(f"Episodes: {num_episodes}")
    print(f"Steps per episode: {steps_per_episode}")
    print("="*60)

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []

    for episode in tqdm.tqdm(range(num_episodes), desc="Generating episodes"):
        obs, info = env.reset()
        current_state = obs

        for step in range(steps_per_episode):
            action = policy.get_action()

            # Add same noise as in mixture dataset for consistency
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

    print("="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Dataset saved to: {save_path}")
    print(f"Total transitions: {len(all_states)}")
    print(f"Average step reward: {dataset['rewards'].mean().item():.4f}")
    print(f"Reward std: {dataset['rewards'].std().item():.4f}")
    print("="*60)


if __name__ == "__main__":
    NUM_EPISODES = 2000
    STEPS_PER_EPISODE = 30
    DATASET_DIR = "data"
    DATASET_FILENAME = "inv_management_single_expert.pt"
    SAVE_PATH = os.path.join(DATASET_DIR, DATASET_FILENAME)

    generate_single_expert_dataset(NUM_EPISODES, STEPS_PER_EPISODE, SAVE_PATH)
