import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd

def compute_discounted_returns(rewards, gamma=0.99):
    """
    Calculates G_t (discounted return) for every timestep t in an episode.
    """
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def main():
    dataset_path = "data/inv_management_base_stock.pt"
    gamma = 0.99
    steps_per_episode = 30
    reward_scale = 0.1

    print(f"Loading {dataset_path}...")
    try:
        dataset = torch.load(dataset_path, weights_only=False)
        rewards_tensor = dataset['rewards'].flatten()
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    total_samples = len(rewards_tensor)
    num_episodes = total_samples // steps_per_episode
    print(f"Analyzing {num_episodes} episodes...")

    data_records = []

    for i in range(num_episodes):
        start_idx = i * steps_per_episode
        end_idx = start_idx + steps_per_episode

        ep_rewards = rewards_tensor[start_idx:end_idx].tolist()
        ep_returns = compute_discounted_returns(ep_rewards, gamma)

        for t, g_t in enumerate(ep_returns):
            data_records.append({
                'timestep': t,
                'G_t_scaled': g_t * reward_scale,
                'reward_scaled': ep_rewards[t] * reward_scale
            })

    df = pd.DataFrame(data_records)

    stats = df.groupby('timestep')['G_t_scaled'].agg(['mean', 'count', 'min'])

    print("\n=== THE ANATOMY OF DECAY (Proof of 0.49) ===")
    print(f"{'Step (t)':<10} | {'Mean G_t (Value)':<20} | {'Explanation'}")
    print("-" * 60)

    display_steps = [0, 5, 10, 15, 20, 25, 29]

    for t in display_steps:
        mean_val = stats.loc[t, 'mean']
        if t == 0:
            note = "Start of Episode (Total Score)"
        elif t == 15:
            note = "Middle of Episode"
        elif t == 29:
            note = "Last Step (No future)"
        else:
            note = ""

        print(f"{t:<10} | {mean_val:>8.4f}             | {note}")

    print("-" * 60)

    global_mean = df['G_t_scaled'].mean()
    print(f"\nGlobal Average (All Steps Combined): {global_mean:.4f}")

    neg_count = df[df['G_t_scaled'] < 0].shape[0]
    total_count = df.shape[0]
    neg_percent = (neg_count / total_count) * 100

    print(f"\n[Why is it so low?]")
    print(
        f"Percentage of states with NEGATIVE Future Value: {neg_percent:.1f}%")
    print(
        f"Average Value of these 'Doomed' states:          {df[df['G_t_scaled'] < 0]['G_t_scaled'].mean():.4f}")

if __name__ == "__main__":
    main()
