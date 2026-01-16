"""
Second Refined Grid Search for Lot-for-Lot Policy.

Best from first refinement: d0=45, d1=19, d2=5
Now searching with even finer grid and extended d0 range.
"""
import itertools
import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv


class LotForLotPolicyPerStage:
    def __init__(self, env, demand_per_stage):
        self.env = env
        self.demand_per_stage = np.array(demand_per_stage)

    def get_action(self):
        period = self.env.period
        num_stages = len(self.env.I[0]) if len(self.env.I) > 0 else 3
        action = self.demand_per_stage.copy().astype(np.float32)

        if period >= len(self.env.I):
            current_I = self.env.I[-1]
        else:
            current_I = self.env.I[period]

        for i in range(num_stages):
            if current_I[i] > self.demand_per_stage[i]:
                action[i] = max(0, self.demand_per_stage[i] - 
                               (current_I[i] - self.demand_per_stage[i]))
        return action.astype(np.float32)


def evaluate_configuration(env, demand_per_stage, num_episodes=50):
    policy = LotForLotPolicyPerStage(env, demand_per_stage)
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

    # Second refinement around: d0=45, d1=19, d2=5
    # Extend d0 higher, fine-tune d1 and d2
    d0_range = list(range(42, 56, 1))  # 14 values (extend higher)
    d1_range = list(range(16, 24, 1))  # 8 values (around 19)
    d2_range = list(range(3, 10, 1))   # 7 values (around 5)
    # 14 * 8 * 7 = 784 configurations

    print("Second Refined Grid Search for Lot-for-Lot Policy...")
    print(f"d0 (Retailer): {d0_range[0]} to {d0_range[-1]}")
    print(f"d1 (Distributor): {d1_range[0]} to {d1_range[-1]}")
    print(f"d2 (Manufacturer): {d2_range[0]} to {d2_range[-1]}")
    
    all_configs = list(itertools.product(d0_range, d1_range, d2_range))
    print(f"Total Configurations: {len(all_configs)}")
    print(f"Episodes per Config: 50")
    print("-" * 50)

    results = []
    for config in tqdm.tqdm(all_configs):
        avg_score = evaluate_configuration(env, np.array(config), num_episodes=50)
        results.append({'d0': config[0], 'd1': config[1], 'd2': config[2], 'mean_profit': avg_score})

    df = pd.DataFrame(results).sort_values(by='mean_profit', ascending=False)

    print("\n" + "=" * 50)
    print("TOP 30 CONFIGURATIONS")
    print("=" * 50)
    print(df.head(30).to_string(index=False))

    best = df.iloc[0]
    print(f"\nBEST: d0={int(best['d0'])}, d1={int(best['d1'])}, d2={int(best['d2'])}, profit={best['mean_profit']:.2f}")

    df.to_csv("optimal_lot_for_lot_params_refined2.csv", index=False)


if __name__ == "__main__":
    main()
