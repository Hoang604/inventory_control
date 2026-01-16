"""
Grid Search for Optimal Noisy Base-Stock Policy Parameters.

Searches for the best base-stock levels (z) and noise standard deviation
for the noisy base-stock policy used for exploration.
"""
import itertools
import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import NoisyBaseStockPolicy


def evaluate_configuration(env, z_params, noise_std, num_episodes=20):
    """
    Runs the environment multiple times with a specific noisy base-stock configuration
    and returns the average profit.

    Args:
        env: The environment instance.
        z_params: Base-stock levels for each stage.
        noise_std: Standard deviation of Gaussian noise.
        num_episodes: Number of episodes to run for averaging.
    """
    policy = NoisyBaseStockPolicy(env, z=z_params, noise_std=noise_std)
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

    # Use known good base-stock levels from previous search
    good_z_configs = [
        [80, 180, 40],
        [80, 200, 40],
        [100, 200, 100],
        [60, 140, 40],
    ]

    # Noise levels to search
    noise_std_range = [5, 10, 15, 20, 25, 30]

    print("Starting Grid Search for Optimal Noisy Base-Stock Policy...")
    print(f"Base-Stock Configs: {good_z_configs}")
    print(f"Noise Std Range: {noise_std_range}")

    all_configs = list(itertools.product(good_z_configs, noise_std_range))
    print(f"Total Configurations to Test: {len(all_configs)}")
    print(f"Episodes per Configuration: 20")
    print("-" * 50)

    results = []

    for z_config, noise_std in tqdm.tqdm(all_configs):
        z_params = np.array(z_config)
        avg_score = evaluate_configuration(env, z_params, noise_std, num_episodes=20)

        results.append({
            'z0': z_config[0],
            'z1': z_config[1],
            'z2': z_config[2],
            'noise_std': noise_std,
            'mean_profit': avg_score
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by='mean_profit', ascending=False)

    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS (All Configurations)")
    print("=" * 50)
    print(df.to_string(index=False))

    best_config = df.iloc[0]

    print("\n" + "=" * 50)
    print("OPTIMAL NOISY BASE-STOCK PARAMETERS:")
    print(f"Base-Stock Levels: [{int(best_config['z0'])}, {int(best_config['z1'])}, {int(best_config['z2'])}]")
    print(f"Noise Std: {best_config['noise_std']}")
    print(f"Expected Profit: {best_config['mean_profit']:.2f}")
    print("=" * 50)

    df.to_csv("optimal_noisy_basestock_params.csv", index=False)
    print("Full results saved to 'optimal_noisy_basestock_params.csv'")


if __name__ == "__main__":
    main()
