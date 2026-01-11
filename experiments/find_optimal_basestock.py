import itertools
import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import BaseStockPolicy


def evaluate_configuration(env, z_params, num_episodes=20):
    """
    Runs the environment multiple times with a specific 'z' configuration
    and returns the average profit.

    Args:
        env: The environment instance.
        z_params: Array of base stock levels.
        num_episodes (int): Number of episodes to run for averaging (default 20).
    """
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

    search_values = list(range(40, 320, 20))

    print(f"Starting Grid Search for Optimal Base Stock Policy...")
    print(f"Search Grid per Stage: {search_values}")
    total_combinations = len(search_values) ** 3
    print(f"Total Configurations to Test: {total_combinations}")
    print(f"Episodes per Configuration: 20")
    print("-" * 50)

    results = []

    parameter_grid = list(itertools.product(
        search_values, search_values, search_values))

    for z_tuple in tqdm.tqdm(parameter_grid):
        z_params = np.array(z_tuple)

        avg_score = evaluate_configuration(env, z_params, num_episodes=20)

        results.append({
            'z_retailer': z_params[0],
            'z_distributor': z_params[1],
            'z_manufacturer': z_params[2],
            'mean_profit': avg_score
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by='mean_profit', ascending=False)

    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS (Top 10 Configurations)")
    print("="*50)
    print(df.head(10).to_string(index=False))

    best_config = df.iloc[0]
    best_z = best_config[['z_retailer', 'z_distributor',
                          'z_manufacturer']].values.astype(int)

    print("\n" + "="*50)
    print(f"THE 'REAL MANAGER' (OPTIMAL) PARAMETERS:")
    print(f"Retailer Target:     {best_z[0]}")
    print(f"Distributor Target:  {best_z[1]}")
    print(f"Manufacturer Target: {best_z[2]}")
    print(f"Expected Profit:     {best_config['mean_profit']:.2f}")
    print("="*50)

    df.to_csv("optimal_basestock_params.csv", index=False)
    print("Full results saved to 'optimal_basestock_params.csv'")


if __name__ == "__main__":
    main()
