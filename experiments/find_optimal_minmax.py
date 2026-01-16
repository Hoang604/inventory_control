"""
Grid Search for Optimal Min-Max (s, S) Policy Parameters.

Searches for the best reorder point (s) and order-up-to level (S)
for each stage in the supply chain (per-stage parameters).
"""
import itertools
import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import MinMaxPolicy


def evaluate_configuration(env, params, num_episodes=30):
    """
    Runs the environment multiple times with a specific (s, S) configuration
    per stage and returns the average profit.
    """
    policy = MinMaxPolicy(env, min_max_params=params)
    total_rewards = []

    for _ in range(num_episodes):
        _, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = policy.get_action()
            _, reward, done, _, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def main():
    env = InvManagementEnv(periods=30, render_mode=None)

    # Reduced search ranges for manageable search space
    # s_range: 5 values, S_range: 8 values
    # Valid pairs about 35, so 35^3 ≈ 42,875... still too large
    # Let's use coarser grid: s_range: 4 values, S_range: 6 values
    # Valid pairs about 20, so 20^3 = 8,000 (good!)
    
    s_range = [0, 40, 80, 120]  # 4 values
    S_range = [60, 100, 140, 180, 220, 260]  # 6 values

    print("Starting Grid Search for Optimal Min-Max (s, S) Per-Stage Policy...")
    print(f"s Range (reorder point): {s_range}")
    print(f"S Range (order-up-to): {S_range}")

    # Generate valid (s, S) pairs where s < S
    valid_pairs = [(s, S) for s in s_range for S in S_range if s < S]
    print(f"Valid (s, S) pairs: {len(valid_pairs)}")

    # Generate all combinations for 3 stages
    all_configs = list(itertools.product(valid_pairs, valid_pairs, valid_pairs))

    print(f"Total Configurations to Test: {len(all_configs)}")
    print(f"Episodes per Configuration: 30")
    print("-" * 50)

    results = []

    for config in tqdm.tqdm(all_configs):
        params = np.array(config)
        avg_score = evaluate_configuration(env, params, num_episodes=30)

        results.append({
            's0': config[0][0], 'S0': config[0][1],
            's1': config[1][0], 'S1': config[1][1],
            's2': config[2][0], 'S2': config[2][1],
            'mean_profit': avg_score
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by='mean_profit', ascending=False)

    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS (Top 20 Configurations)")
    print("=" * 50)
    print(df.head(20).to_string(index=False))

    best_config = df.iloc[0]

    print("\n" + "=" * 50)
    print("OPTIMAL MIN-MAX PARAMETERS (Per-Stage):")
    print(f"Stage 0: s={int(best_config['s0'])}, S={int(best_config['S0'])}")
    print(f"Stage 1: s={int(best_config['s1'])}, S={int(best_config['S1'])}")
    print(f"Stage 2: s={int(best_config['s2'])}, S={int(best_config['S2'])}")
    print(f"Expected Profit: {best_config['mean_profit']:.2f}")
    print("=" * 50)

    df.to_csv("optimal_minmax_params.csv", index=False)
    print("Full results saved to 'optimal_minmax_params.csv'")


if __name__ == "__main__":
    main()
