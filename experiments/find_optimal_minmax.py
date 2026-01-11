import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import MinMaxPolicy


def evaluate_configuration(env, s_val, S_val, num_episodes=20):
    """
    Runs the environment multiple times with a specific (s, S) configuration
    applied to ALL stages.

    Args:
        env: The environment instance.
        s_val (int): Reorder point.
        S_val (int): Order-up-to level.
        num_episodes (int): Number of episodes to run for averaging.
    """
    params = np.array([
        [s_val, S_val],
        [s_val, S_val],
        [s_val, S_val]
    ])

    policy = MinMaxPolicy(env, min_max_params=params)
    total_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
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

    S_range = list(range(20, 201, 10))

    s_range = list(range(0, 101, 10))

    print(f"Starting Grid Search for Optimal Min-Max (s, S) Standard Policy...")
    print(f"S Range: {S_range}")
    print(f"s Range: {s_range}")

    valid_pairs = [(s, S) for s in s_range for S in S_range if s < S]

    print(f"Total Valid Configurations: {len(valid_pairs)}")
    print(f"Episodes per Configuration: 20")
    print("-" * 50)

    results = []

    for (s, S) in tqdm.tqdm(valid_pairs):
        avg_score = evaluate_configuration(env, s, S, num_episodes=20)

        results.append({
            's': s,
            'S': S,
            'mean_profit': avg_score
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by='mean_profit', ascending=False)

    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS (Top 10 Configurations)")
    print("="*50)
    print(df.head(10).to_string(index=False))

    best_config = df.iloc[0]

    print("\n" + "="*50)
    print(f"THE 'REAL MANAGER' (OPTIMAL) PARAMETERS:")
    print(
        f"Standard Policy: (s={int(best_config['s'])}, S={int(best_config['S'])})")
    print(f"Expected Profit: {best_config['mean_profit']:.2f}")
    print("="*50)

    df.to_csv("optimal_minmax_params.csv", index=False)
    print("Full results saved to 'optimal_minmax_params.csv'")


if __name__ == "__main__":
    main()
