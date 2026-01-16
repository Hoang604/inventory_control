import torch
import numpy as np
import os
import yaml
import json
import scipy.stats as stats
from utils.config_loader import load_config
from src.base.inv_management_env import InvManagementEnv
from src.models.iql.actor import Actor
from src.base.policies import BaseStockPolicy

ARTIFACTS_DIR = "paper/artifacts"
RESULTS_FILE = os.path.join(ARTIFACTS_DIR, "results.json")
HYPERPARAMS_FILE = os.path.join(ARTIFACTS_DIR, "hyperparameters.yaml")


def save_results(results_dict):
    """Saves the results dictionary to a JSON file in the artifacts directory."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {RESULTS_FILE}")


def save_hyperparameters(config):
    """Saves the configuration to a YAML file in the artifacts directory."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(HYPERPARAMS_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Hyperparameters saved to {HYPERPARAMS_FILE}")


def run_episode(env, agent=None, render=False, baseline_policy_callable=None):
    """
    Runs a single episode.
    If agent is None, uses a specified baseline policy callable.
    If agent is provided, uses the agent's policy.
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if render:
        policy_name = ""
        if agent is None:
            policy_name = "Base Stock Policy" if baseline_policy_callable else "Random Policy"
        else:
            policy_name = "IQL Agent"
        print(f"\n--- Starting Episode ({policy_name}) ---")

    while not done:
        if agent is None:
            if baseline_policy_callable:
                action = baseline_policy_callable(obs)
            else:
                action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean, _ = agent(state_tensor)
                action = action_mean.cpu().numpy()[0]

        next_obs, reward, done, _, info = env.step(action)
        total_reward += reward

        if render:
            print(f"Step {step+1}:")
            print(f"  Demand: {info['demand']}")
            print(f"  Inventory: {info['inventory']}")
            print(f"  Action (Orders): {action}")
            print(f"  Sales: {info['sales']}")
            print(f"  Lost Sales: {info['lost_sales']}")
            print(f"  Step Reward: {reward:.2f}")
            print("-" * 20)

        obs = next_obs
        step += 1

    return total_reward


def main():
    env = InvManagementEnv(render_mode=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "/home/hoang/python/inventory_control/checkpoints/inv_management_iql_minmax_run_12012026_143429/actor/checkpoint_epoch_99.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading trained agent from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if 'config' in checkpoint:
            print("Configuration loaded successfully from checkpoint.")
            config = checkpoint['config']
        else:
            print(
                "Warning: Config not found in checkpoint. Falling back to default config.")
            config = load_config()

        actor = Actor(config).to(device)
        actor.load_state_dict(checkpoint['model_state_dict'])
        actor.eval()

        save_hyperparameters(config)

    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}.")
        return

    NUM_TEST_EPISODES = 100

    target_levels = np.array([80, 180, 40])
    base_stock_policy = BaseStockPolicy(env, z=target_levels)

    print(
        f"\nRunning Base Stock Policy Experiment (z={target_levels}, {NUM_TEST_EPISODES} episodes)...")
    baseline_rewards = []

    def base_stock_callable(obs_val):
        return base_stock_policy.get_action()

    for _ in range(NUM_TEST_EPISODES):
        baseline_rewards.append(run_episode(
            env, agent=None, render=False, baseline_policy_callable=base_stock_callable))

    print(f"\nRunning IQL Agent Experiment ({NUM_TEST_EPISODES} episodes)...")
    agent_rewards = []
    for _ in range(NUM_TEST_EPISODES):
        agent_rewards.append(run_episode(env, agent=actor, render=False))

    base_mean = np.mean(baseline_rewards)
    base_std = np.std(baseline_rewards)
    agent_mean = np.mean(agent_rewards)
    agent_std = np.std(agent_rewards)

    t_stat, p_value = stats.ttest_ind(
        agent_rewards, baseline_rewards, equal_var=False)

    diff = agent_mean - base_mean
    pct_improvement = (diff / abs(base_mean)) * \
        100 if base_mean != 0 else 0

    results_data = {
        "baseline": {
            "name": f"Base Stock Policy (z={target_levels.tolist()})",
            "mean_reward": float(base_mean),
            "std_dev": float(base_std),
            "n_episodes": NUM_TEST_EPISODES
        },
        "method": {
            "name": "IQL Agent",
            "mean_reward": float(agent_mean),
            "std_dev": float(agent_std),
            "n_episodes": NUM_TEST_EPISODES
        },
        "comparison": {
            "absolute_improvement": float(diff),
            "percent_improvement": float(pct_improvement),
            "p_value": float(p_value),
            "statistically_significant": bool(p_value < 0.05)
        }
    }
    save_results(results_data)

    print("\n" + "="*40)
    print("FINAL RESULTS (Averaged)")
    print("="*40)
    print(f"Base Stock Policy: {base_mean:.2f} +/- {base_std:.2f}")
    print(f"IQL Agent:         {agent_mean:.2f} +/- {agent_std:.2f}")
    print("-" * 40)
    if diff > 0:
        print(f"Improvement: +{diff:.2f} (+{pct_improvement:.1f}%)")
    else:
        print(f"Difference: {diff:.2f}")

    print(f"P-value: {p_value:.4e}")
    if p_value < 0.05:
        print("Result: Statistically Significant")
    else:
        print("Result: Not Significant")
    print("="*40)


if __name__ == "__main__":
    main()
