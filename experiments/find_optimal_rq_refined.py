"""
Final Refined Grid Search for (R, Q) Policy.

Best from first refinement: R0=80, Q0=20, R1=140, Q1=30, R2=10, Q2=10 (profit: 383.28)
Exploring edges: R1 higher, R2 lower, Q2 lower.
"""
import itertools
import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import RQPolicy


def evaluate_configuration(env, rq_params, num_episodes=50):
    policy = RQPolicy(env, rq_params=rq_params)
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

    # Final refinement - extending edges
    R0_range = [70, 80, 90, 100]          # 4 values (centered)
    Q0_range = [15, 20, 25]               # 3 values (centered at 20)
    R1_range = [140, 150, 160, 170, 180]  # 5 values (extend higher!)
    Q1_range = [20, 25, 30, 35]           # 4 values (around 30)
    R2_range = [0, 5, 10, 15, 20]         # 5 values (extend lower!)
    Q2_range = [5, 10, 15]                # 3 values (extend lower!)

    print("Final Refined Grid Search for (R, Q) Policy...")
    print(f"R0: {R0_range}, Q0: {Q0_range}")
    print(f"R1: {R1_range}, Q1: {Q1_range}")
    print(f"R2: {R2_range}, Q2: {Q2_range}")

    all_configs = list(itertools.product(
        itertools.product(R0_range, Q0_range),
        itertools.product(R1_range, Q1_range),
        itertools.product(R2_range, Q2_range)
    ))

    print(f"Total Configurations: {len(all_configs)}")
    print(f"Episodes per Config: 50")
    print("-" * 50)

    results = []
    for config in tqdm.tqdm(all_configs):
        rq_params = np.array(config)
        avg_score = evaluate_configuration(env, rq_params, num_episodes=50)
        results.append({
            'R0': config[0][0], 'Q0': config[0][1],
            'R1': config[1][0], 'Q1': config[1][1],
            'R2': config[2][0], 'Q2': config[2][1],
            'mean_profit': avg_score
        })

    df = pd.DataFrame(results).sort_values(by='mean_profit', ascending=False)

    print("\n" + "=" * 50)
    print("TOP 30 CONFIGURATIONS")
    print("=" * 50)
    print(df.head(30).to_string(index=False))

    best = df.iloc[0]
    print(f"\nBEST: R0={int(best['R0'])}, Q0={int(best['Q0'])}, "
          f"R1={int(best['R1'])}, Q1={int(best['Q1'])}, "
          f"R2={int(best['R2'])}, Q2={int(best['Q2'])}, profit={best['mean_profit']:.2f}")

    df.to_csv("optimal_rq_params_refined2.csv", index=False)


if __name__ == "__main__":
    main()
