"""
Grid Search for Optimal Periodic Review (T, S) Policy Parameters.

Searches for the best review period (T) and order-up-to level (S)
for the supply chain.
"""
import itertools
import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import PeriodicReviewPolicy


def evaluate_configuration(env, review_period, S_levels, num_episodes=20):
    """
    Runs the environment multiple times with a specific (T, S) configuration
    and returns the average profit.

    Args:
        env: The environment instance.
        review_period: Number of periods between reviews (T).
        S_levels: Order-up-to levels for each stage.
        num_episodes: Number of episodes to run for averaging.
    """
    policy = PeriodicReviewPolicy(env, review_period=review_period, S_levels=S_levels)
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
    T_range = list(range(1, 11))  # Review periods (1 to 10)
    S_range = list(range(50, 301, 50))  # Order-up-to levels

    print("Starting Grid Search for Optimal Periodic Review (T, S) Policy...")
    print(f"T Range (review period): {T_range}")
    print(f"S Range (order-up-to): {S_range}")

    # Generate all combinations of T and S for 3 stages
    S_configs = list(itertools.product(S_range, S_range, S_range))
    all_configs = list(itertools.product(T_range, S_configs))

    print(f"Total Configurations to Test: {len(all_configs)}")
    print(f"Episodes per Configuration: 20")
    print("-" * 50)

    results = []

    for T, S_tuple in tqdm.tqdm(all_configs):
        S_levels = np.array(S_tuple)
        avg_score = evaluate_configuration(env, T, S_levels, num_episodes=20)

        results.append({
            'T': T,
            'S0': S_tuple[0],
            'S1': S_tuple[1],
            'S2': S_tuple[2],
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
    print("OPTIMAL PERIODIC REVIEW PARAMETERS:")
    print(f"Review Period T: {int(best_config['T'])}")
    print(f"S Levels: [{int(best_config['S0'])}, {int(best_config['S1'])}, {int(best_config['S2'])}]")
    print(f"Expected Profit: {best_config['mean_profit']:.2f}")
    print("=" * 50)

    df.to_csv("optimal_periodic_review_params.csv", index=False)
    print("Full results saved to 'optimal_periodic_review_params.csv'")


if __name__ == "__main__":
    main()
