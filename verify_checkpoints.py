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
    default_config = load_config()
    
    # Auto-discover experiments starting with EXP_
    base_dir = "checkpoints"
    experiment_dirs = sorted(glob.glob(os.path.join(base_dir, "EXP_*")))
    
    if not experiment_dirs:
        print("No experiments found matching 'checkpoints/EXP_*'")
        return

    print(f"Found {len(experiment_dirs)} experiments to evaluate.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = InvManagementEnv(render_mode=None)

    # 1. Calculate Baseline (Once for all)
    print("\nCalculating Baseline (MinMax) Performance...")
    baseline_mean, baseline_std = get_baseline_performance(
        env, num_episodes=250)
    print(f"Baseline: {baseline_mean:.2f} +/- {baseline_std:.2f}\n")

    all_results = []
    
    print(f"{ 'Experiment':<40} | { 'Epoch':<5} | { 'Mean Reward':<12} | { 'Std Dev':<10} | { 'Diff':<10}")
    print("-" * 95)

    for experiment_path in experiment_dirs:
        experiment_name = os.path.basename(experiment_path)
        actor_dir = os.path.join(experiment_path, "actor")

        if not os.path.exists(actor_dir):
            continue

        # Find checkpoints
        checkpoint_files = glob.glob(os.path.join(actor_dir, "checkpoint_epoch_*.pth"))
        
        # Extract and sort
        checkpoints = []
        for f in checkpoint_files:
            match = re.search(r"checkpoint_epoch_(\d+).pth", f)
            if match:
                epoch = int(match.group(1))
                checkpoints.append((epoch, f))
        
        checkpoints.sort(key=lambda x: x[0])
        
        # Filter: Every 5 epochs (or just 4, 9, 14...)
        selected_checkpoints = [cp for cp in checkpoints if (cp[0] + 1) % 5 == 0]

        for epoch, file_path in selected_checkpoints:
            try:
                # Load Checkpoint
                checkpoint_data = torch.load(file_path, map_location=device)
                
                # Config Strategy
                if 'config' in checkpoint_data:
                    current_config = checkpoint_data['config']
                else:
                    current_config = default_config

                # Init Agent
                actor = Actor(current_config).to(device)
                actor.load_state_dict(checkpoint_data['model_state_dict'])
                actor.eval()

                # Evaluate
                mean_reward, std_reward = run_evaluation_loop(env, actor, num_episodes=30) # Reduced episodes for speed
                diff = mean_reward - baseline_mean

                print(
                    f"{experiment_name:<40} | {epoch:<5} | {mean_reward:<12.2f} | {std_reward:<10.2f} | {diff:<+10.2f}")

                all_results.append({
                    "experiment": experiment_name,
                    "epoch": epoch,
                    "mean_reward": mean_reward,
                    "std_dev": std_reward,
                    "diff": diff,
                    "baseline_mean": baseline_mean
                })
            except Exception as e:
                print(f"Error evaluating {experiment_name} Epoch {epoch}: {e}")

    # Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = "all_experiments_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"\nAnalysis saved to {output_file}")
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    main()
