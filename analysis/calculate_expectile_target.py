import os
from pathlib import Path

import torch
import numpy as np
import os

def compute_discounted_returns(rewards, gamma=0.99):
    """
    Calculates G_t (discounted return) for every timestep t in an episode.
    G_t = r_t + gamma * r_{t+1} + ... + gamma^{T-t} * r_T
    """
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def calculate_expectile(values, tau=0.7, tolerance=1e-5, max_iter=1000):
    """
    Solves for the scalar 'e' that minimizes the asymmetric least squares loss.
    """
    values = np.array(values)
    e = np.mean(values)

    for i in range(max_iter):
        indices_high = values > e
        sum_high = np.sum(values[indices_high])
        sum_low = np.sum(values[~indices_high])
        count_high = np.sum(indices_high)
        count_low = np.sum(~indices_high)

        numerator = tau * sum_high + (1 - tau) * sum_low
        denominator = tau * count_high + (1 - tau) * count_low

        new_e = numerator / denominator

        if abs(new_e - e) < tolerance:
            return new_e

        e = new_e

    print("Warning: Expectile calculation did not strictly converge.")
    return e

def main():
    dataset_path = "data/inv_management_base_stock.pt"
    gamma = 0.99
    steps_per_episode = 30
    reward_scale = 0.1
    target_tau = 0.8

    print(f"Loading {dataset_path}...")
    try:
        dataset = torch.load(dataset_path, weights_only=False)
        rewards_tensor = dataset['rewards'].flatten()
    except Exception as e:
        print(f"Error: {e}")
        return

    total_samples = len(rewards_tensor)
    num_episodes = total_samples // steps_per_episode
    print(
        f"Processing {num_episodes} episodes (Assuming {steps_per_episode} steps/ep)...")

    all_timesteps_returns = []
    episode_total_rewards = []
    episode_discounted_totals = []

    for i in range(num_episodes):
        start_idx = i * steps_per_episode
        end_idx = start_idx + steps_per_episode

        ep_rewards = rewards_tensor[start_idx:end_idx].tolist()

        ep_returns = compute_discounted_returns(ep_rewards, gamma)
        all_timesteps_returns.extend(ep_returns)

        episode_total_rewards.append(sum(ep_rewards))
        episode_discounted_totals.append(
            ep_returns[0])

    scaled_timestep_returns = np.array(all_timesteps_returns) * reward_scale
    scaled_episode_totals = np.array(episode_total_rewards) * reward_scale
    scaled_discounted_totals = np.array(
        episode_discounted_totals) * reward_scale

    print(f"\n--- EPISODE-LEVEL STATISTICS (Average Score per Game) ---")
    print(f"Total Episodes Analyzed: {num_episodes}")

    print(f"\n[Undiscounted Total Reward (Sum r)]")
    print(
        f"  Avg Raw Score:     {np.mean(episode_total_rewards):.4f}  <-- Real-world performance avg")
    print(
        f"  Avg Scaled Score:  {np.mean(scaled_episode_totals):.4f}  <-- Compare with Agent logs")
    print(f"  Best Episode:      {np.max(episode_total_rewards):.4f}")
    print(f"  Worst Episode:     {np.min(episode_total_rewards):.4f}")

    print(f"\n[Discounted Total Return (G_0)]")
    print(f"  Avg Scaled G_0:    {np.mean(scaled_discounted_totals):.4f}")

    print("\n--- TIMESTEP-LEVEL STATISTICS (Q-Network Targets) ---")
    print(
        f"Global Mean Return (Mean of all G_t): {np.mean(scaled_timestep_returns):.4f}")
    print(
        f"Global Min G_t:                       {np.min(scaled_timestep_returns):.4f}")
    print(
        f"Global Max G_t:                       {np.max(scaled_timestep_returns):.4f}")

    expectile_val = calculate_expectile(
        scaled_timestep_returns, tau=target_tau)

    print(f"\n--- TARGET ANALYSIS (tau={target_tau}) ---")
    print(f"Calculated {target_tau}-Expectile: {expectile_val:.4f}")
    print(
        f"Interpretation: Your V-Network should converge to approx {expectile_val:.4f}")

    if expectile_val > np.mean(scaled_timestep_returns):
        print("Status: OPTIMISTIC (Target > Mean). This biases the agent toward high-return states.")
    else:
        print("Status: PESSIMISTIC (Target < Mean). This biases the agent toward safety.")

if __name__ == "__main__":
    main()
