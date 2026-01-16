"""
Refined Grid Search for Base-Stock Policy - Testing lower z2 values.

Best from initial: z0=80, z1=180, z2=40
Issue: z2=40 was at lower edge. Testing z2 < 40.
"""
import itertools
import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import BaseStockPolicy


def evaluate_configuration(env, z_params, num_episodes=50):
    policy = BaseStockPolicy(env, z=z_params)
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

    # Refined search with lower z2 values
    z0_range = list(range(60, 110, 10))   # 5 values (around 80)
    z1_range = list(range(160, 220, 10))  # 6 values (around 180-200)
    z2_range = list(range(10, 60, 5))     # 10 values (extend lower to 10!)

    print("Refined Grid Search for Base-Stock Policy...")
    print(f"z0: {z0_range}")
    print(f"z1: {z1_range}")
    print(f"z2: {z2_range}")

    all_configs = list(itertools.product(z0_range, z1_range, z2_range))

    print(f"Total Configurations: {len(all_configs)}")
    print(f"Episodes per Config: 50")
    print("-" * 50)

    results = []
    for config in tqdm.tqdm(all_configs):
        z_params = np.array(config)
        avg_score = evaluate_configuration(env, z_params, num_episodes=50)
        results.append({
            'z0': config[0],
            'z1': config[1],
            'z2': config[2],
            'mean_profit': avg_score
        })

    df = pd.DataFrame(results).sort_values(by='mean_profit', ascending=False)

    print("\n" + "=" * 50)
    print("TOP 30 CONFIGURATIONS")
    print("=" * 50)
    print(df.head(30).to_string(index=False))

    best = df.iloc[0]
    print(f"\nBEST: z0={int(best['z0'])}, z1={int(best['z1'])}, z2={int(best['z2'])}, profit={best['mean_profit']:.2f}")

    df.to_csv("optimal_basestock_params_refined.csv", index=False)


if __name__ == "__main__":
    main()
