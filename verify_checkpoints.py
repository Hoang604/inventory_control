import torch
import numpy as np
import os
import glob
import re
import pandas as pd
from utils.config_loader import load_config
from src.base.inv_management_env import InvManagementEnv
from src.models.iql.actor import Actor
from src.base.policies import MinMaxPolicy


def run_evaluation_loop(env, agent, num_episodes=30):
    """Runs the agent in the environment for num_episodes and returns mean reward."""
    rewards = []
    device = next(agent.parameters()).device

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean, _ = agent(state_tensor)
                action = action_mean.cpu().numpy()[0]

            next_obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            obs = next_obs

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


def get_baseline_performance(env, num_episodes=30):
    """Calculates the baseline MinMax policy performance."""
    min_max_policy = MinMaxPolicy(env)
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()

        # Randomize parameters per episode (matching training/test logic)
        S_levels = sorted(np.random.randint(10, 200, size=3))
        s_levels = [np.random.randint(0, S) for S in S_levels]
        policy_params = np.column_stack((s_levels, S_levels))

        done = False
        total_reward = 0

        while not done:
            action = min_max_policy.get_action(params=policy_params)
            next_obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            obs = next_obs

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


def main():
    # Configuration
    config = load_config()
    experiment_dir = "checkpoints/inv_management_iql_minmax_run_06122025_232855"
    actor_dir = os.path.join(experiment_dir, "actor")

    if not os.path.exists(actor_dir):
        print(f"Error: Directory not found: {actor_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = InvManagementEnv(render_mode=None)

    # 1. Calculate Baseline
    print("Calculating Baseline (MinMax) Performance...")
    baseline_mean, baseline_std = get_baseline_performance(
        env, num_episodes=50)
    print(f"Baseline: {baseline_mean:.2f} +/- {baseline_std:.2f}\n")

    # 2. Find and Sort Checkpoints
    checkpoint_files = glob.glob(os.path.join(
        actor_dir, "checkpoint_epoch_*.pth"))

    # Extract epoch numbers
    checkpoints = []
    for f in checkpoint_files:
        match = re.search(r"checkpoint_epoch_(\d+).pth", f)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, f))

    # Sort by epoch
    checkpoints.sort(key=lambda x: x[0])

    # Filter: Epoch 4, 9, 14... (Every 5 epochs starting from 4)
    # Adjust filter logic as needed. The user asked for 4, 9...
    selected_checkpoints = [cp for cp in checkpoints if (cp[0] + 1) % 5 == 0]

    results = []

    print(f"{ 'Epoch':<10} | { 'Mean Reward':<15} | { 'Std Dev':<15} | { 'Vs Baseline':<15}")
    print("-----------------------------------------------------------------")

    for epoch, file_path in selected_checkpoints:
        # Load Agent
        actor = Actor(config).to(device)
        checkpoint_data = torch.load(file_path, map_location=device)
        actor.load_state_dict(checkpoint_data['model_state_dict'])
        actor.eval()

        # Evaluate
        mean_reward, std_reward = run_evaluation_loop(
            env, actor, num_episodes=50)

        # Compare
        diff = mean_reward - baseline_mean

        print(
            f"{epoch:<10} | {mean_reward:<15.2f} | {std_reward:<15.2f} | {diff:<+15.2f}")

        results.append({
            "epoch": epoch,
            "mean_reward": mean_reward,
            "std_dev": std_reward,
            "diff": diff
        })

    # Save to CSV for easier analysis
    df = pd.DataFrame(results)
    df.to_csv("checkpoint_analysis.csv", index=False)
    print(f"\nAnalysis saved to checkpoint_analysis.csv")


if __name__ == "__main__":
    main()
