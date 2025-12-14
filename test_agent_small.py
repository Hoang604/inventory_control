import torch
import numpy as np
import os
import yaml
import json
import scipy.stats as stats
from utils.config_loader import load_config
from src.base.inv_management_env import InvManagementEnv
from src.models.iql.actor import SActor
from src.base.policies import MinMaxPolicy

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
    If agent is None, uses random actions or a specified baseline policy.
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
            policy_name = "Min-Max Policy" if baseline_policy_callable else "Random Policy"
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

    checkpoint_path = "checkpoints/EXP_03_TAU_EXTREME_07122025_024648/actor/best_loss.pth"

    if os.path.exists(checkpoint_path):
        print(f"Loading trained agent from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'config' in checkpoint:
            print("Configuration loaded successfully from checkpoint.")
            config = checkpoint['config']
        else:
            print(
                "Warning: Config not found in checkpoint. Falling back to default config.")
            config = load_config()

        actor = SActor(config).to(device)
        actor.load_state_dict(checkpoint['model_state_dict'])
        actor.eval()

        save_hyperparameters(config)

    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}.")
        return

    NUM_TEST_EPISODES = 100

    print(
        f"\nRunning Min-Max Policy Experiment ({NUM_TEST_EPISODES} episodes)...")
    min_max_rewards = []
    min_max_policy = MinMaxPolicy(env)

    for _ in range(NUM_TEST_EPISODES):
        S_levels = sorted(np.random.randint(10, 200, size=3))
        s_levels = [np.random.randint(0, S) for S in S_levels]
        episode_policy_params = np.column_stack((s_levels, S_levels))

        def min_max_callable(obs_val): return min_max_policy.get_action(
            params=episode_policy_params)
        min_max_rewards.append(run_episode(
            env, agent=None, render=False, baseline_policy_callable=min_max_callable))

    print(f"\nRunning IQL Agent Experiment ({NUM_TEST_EPISODES} episodes)...")
    agent_rewards = []
    for _ in range(NUM_TEST_EPISODES):
        agent_rewards.append(run_episode(env, agent=actor, render=False))

    min_max_mean = np.mean(min_max_rewards)
    min_max_std = np.std(min_max_rewards)
    agent_mean = np.mean(agent_rewards)
    agent_std = np.std(agent_rewards)

    t_stat, p_value = stats.ttest_ind(
        agent_rewards, min_max_rewards, equal_var=False)

    diff = agent_mean - min_max_mean
    pct_improvement = (diff / abs(min_max_mean)) * \
        100 if min_max_mean != 0 else 0

    results_data = {
        "baseline": {
            "name": "Randomized Min-Max Policy",
            "mean_reward": float(min_max_mean),
            "std_dev": float(min_max_std),
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

    print("\n" + "="*30)
    print("FINAL RESULTS (Averaged)")
    print("="*30)
    print(f"Min-Max Policy: {min_max_mean:.2f} +/- {min_max_std:.2f}")
    print(f"IQL Agent:     {agent_mean:.2f} +/- {agent_std:.2f}")
    print("-" * 30)
    if diff > 0:
        print(f"Improvement: +{diff:.2f} (+{pct_improvement:.1f}%)")
    else:
        print(f"Difference: {diff:.2f}")
    print(f"P-value: {p_value:.4e}")
    print("="*30)


if __name__ == "__main__":
    main()
