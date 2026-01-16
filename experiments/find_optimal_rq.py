"""
Grid Search for Optimal (R, Q) Policy Parameters.

Searches for the best reorder point (R) and fixed order quantity (Q)
for each stage in the supply chain.
"""
import itertools
import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import RQPolicy


def evaluate_configuration(env, rq_params, num_episodes=20):
    """
    Runs the environment multiple times with a specific (R, Q) configuration
    and returns the average profit.

    Args:
        env: The environment instance.
        rq_params: Array of [(R, Q), ...] for each stage.
        num_episodes: Number of episodes to run for averaging.
    """
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

    # Search ranges
    R_range = list(range(20, 121, 20))  # Reorder points
    Q_range = list(range(20, 101, 20))  # Fixed order quantities

    print("Starting Grid Search for Optimal (R, Q) Policy...")
    print(f"R Range (reorder point): {R_range}")
    print(f"Q Range (order quantity): {Q_range}")

    # Generate all valid (R, Q) pairs for 3 stages
    rq_pairs = list(itertools.product(R_range, Q_range))
    all_configs = list(itertools.product(rq_pairs, rq_pairs, rq_pairs))

    print(f"Total Configurations to Test: {len(all_configs)}")
    print(f"Episodes per Configuration: 20")
    print("-" * 50)

    results = []

    for config in tqdm.tqdm(all_configs):
        rq_params = np.array(config)
        avg_score = evaluate_configuration(env, rq_params, num_episodes=20)

        results.append({
            'R0': config[0][0], 'Q0': config[0][1],
            'R1': config[1][0], 'Q1': config[1][1],
            'R2': config[2][0], 'Q2': config[2][1],
            'mean_profit': avg_score
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by='mean_profit', ascending=False)

    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS (Top 10 Configurations)")
    print("=" * 50)
    print(df.head(10).to_string(index=False))

    best_config = df.iloc[0]

    print("\n" + "=" * 50)
    print("OPTIMAL (R, Q) PARAMETERS:")
    print(f"Stage 0: R={int(best_config['R0'])}, Q={int(best_config['Q0'])}")
    print(f"Stage 1: R={int(best_config['R1'])}, Q={int(best_config['Q1'])}")
    print(f"Stage 2: R={int(best_config['R2'])}, Q={int(best_config['Q2'])}")
    print(f"Expected Profit: {best_config['mean_profit']:.2f}")
    print("=" * 50)

    df.to_csv("optimal_rq_params.csv", index=False)
    print("Full results saved to 'optimal_rq_params.csv'")


if __name__ == "__main__":
    main()
