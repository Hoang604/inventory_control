"""
Grid Search for Optimal Lot-for-Lot (L4L) Policy Parameters.

Searches for the best expected demand per stage for the L4L policy.
Each stage can have different demand expectations.
"""
import itertools
import numpy as np
import pandas as pd
import tqdm
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import LotForLotPolicy


class LotForLotPolicyPerStage:
    """
    Lot-for-Lot policy with per-stage expected demand.
    """
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


def evaluate_configuration(env, demand_per_stage, num_episodes=30):
    """
    Runs the environment multiple times with per-stage demand expectations.
    """
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

    # Per-stage demand expectations (around true mean of 20)
    # Range: 10 to 35 with step 1 = 26 values per stage
    # 26^3 = 17,576 configurations (within target 3000-20000)
    demand_range = list(range(10, 36, 1))

    print("Starting Grid Search for Optimal Lot-for-Lot Policy (Per-Stage)...")
    print(f"Demand Range per Stage: {demand_range[0]} to {demand_range[-1]} (step=1)")
    
    all_configs = list(itertools.product(demand_range, demand_range, demand_range))
    
    print(f"Total Configurations to Test: {len(all_configs)}")
    print(f"Episodes per Configuration: 30")
    print("-" * 50)

    results = []

    for config in tqdm.tqdm(all_configs):
        demand_per_stage = np.array(config)
        avg_score = evaluate_configuration(env, demand_per_stage, num_episodes=30)

        results.append({
            'd0': config[0],
            'd1': config[1],
            'd2': config[2],
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
    print("OPTIMAL LOT-FOR-LOT PARAMETERS (Per-Stage Demand):")
    print(f"Stage 0 Demand: {int(best_config['d0'])}")
    print(f"Stage 1 Demand: {int(best_config['d1'])}")
    print(f"Stage 2 Demand: {int(best_config['d2'])}")
    print(f"Expected Profit: {best_config['mean_profit']:.2f}")
    print("=" * 50)

    df.to_csv("optimal_lot_for_lot_params.csv", index=False)
    print("Full results saved to 'optimal_lot_for_lot_params.csv'")


if __name__ == "__main__":
    main()
