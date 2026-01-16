"""
Third Refined Grid Search for Min-Max (s, S) Policy.

Best from second refinement: s0=65, S0=70, s1=150, S1=155, s2=0, S2=40-60
Extending s1/S1 range higher, and S0 range lower.
"""
import itertools
import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import MinMaxPolicy


def evaluate_configuration(env, params, num_episodes=50):
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

    # Third refinement: extend s1/S1 higher, S0 lower
    s0_range = list(range(60, 75, 5))     # 3 values (around 65)
    S0_range = list(range(65, 80, 5))     # 3 values (extend lower)
    s1_range = list(range(145, 175, 5))   # 6 values (extend higher)
    S1_range = list(range(150, 180, 5))   # 6 values (extend higher)
    s2_range = [0]                         # Fixed at 0
    S2_range = list(range(35, 65, 5))     # 6 values

    print("Third Refined Grid Search for Min-Max (s, S) Policy...")
    print(f"Stage 0: s in {s0_range}, S in {S0_range}")
    print(f"Stage 1: s in {s1_range}, S in {S1_range}")
    print(f"Stage 2: s in {s2_range}, S in {S2_range}")

    # Generate valid configs
    configs = []
    for s0 in s0_range:
        for S0 in S0_range:
            if s0 >= S0:
                continue
            for s1 in s1_range:
                for S1 in S1_range:
                    if s1 >= S1:
                        continue
                    for s2 in s2_range:
                        for S2 in S2_range:
                            if s2 >= S2:
                                continue
                            configs.append(((s0, S0), (s1, S1), (s2, S2)))

    print(f"Total Configurations: {len(configs)}")
    print(f"Episodes per Config: 50")
    print("-" * 50)

    results = []
    for config in tqdm.tqdm(configs):
        params = np.array(config)
        avg_score = evaluate_configuration(env, params, num_episodes=50)
        results.append({
            's0': config[0][0], 'S0': config[0][1],
            's1': config[1][0], 'S1': config[1][1],
            's2': config[2][0], 'S2': config[2][1],
            'mean_profit': avg_score
        })

    df = pd.DataFrame(results).sort_values(by='mean_profit', ascending=False)

    print("\n" + "=" * 50)
    print("TOP 30 CONFIGURATIONS")
    print("=" * 50)
    print(df.head(30).to_string(index=False))

    best = df.iloc[0]
    print(f"\nBEST: s0={int(best['s0'])}, S0={int(best['S0'])}, "
          f"s1={int(best['s1'])}, S1={int(best['S1'])}, "
          f"s2={int(best['s2'])}, S2={int(best['S2'])}, profit={best['mean_profit']:.2f}")

    df.to_csv("optimal_minmax_params_refined3.csv", index=False)


if __name__ == "__main__":
    main()
