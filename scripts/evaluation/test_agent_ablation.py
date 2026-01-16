"""
Test Agent Performance with Ablation Study

This script evaluates and compares three policies:
1. Best Expert: Base-Stock Policy with z=[80, 180, 40]
2. IQL (Mixture): Trained on diverse expert mixture
3. IQL (Single): Trained on single best expert (ablation)

Outputs:
- paper/artifacts/ablation_study.json (detailed comparison)
- Updates paper/artifacts/results.json with ablation data
"""
import torch
import numpy as np
import os
import json
import scipy.stats as stats
from utils.config_loader import load_config
from src.base.inv_management_env import InvManagementEnv
from src.models.iql.actor import Actor
from src.base.policies import BaseStockPolicy

ARTIFACTS_DIR = "paper/artifacts"
RESULTS_FILE = os.path.join(ARTIFACTS_DIR, "results.json")
ABLATION_FILE = os.path.join(ARTIFACTS_DIR, "ablation_study.json")


def run_episode(env, agent=None, baseline_policy_callable=None):
    """Run a single episode and return total reward."""
    obs, _ = env.reset()
    done = False
    total_reward = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while not done:
        if agent is None:
            if baseline_policy_callable:
                action = baseline_policy_callable(obs)
            else:
                action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean, _ = agent(state_tensor)
                action = action_mean.cpu().numpy()[0]

        next_obs, reward, done, _, info = env.step(action)
        total_reward += reward
        obs = next_obs

    return total_reward


def load_agent(checkpoint_path, device):
    """Load an IQL agent from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading agent from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("Warning: Config not found in checkpoint. Using default config.")
        config = load_config()

    actor = Actor(config).to(device)
    actor.load_state_dict(checkpoint['model_state_dict'])
    actor.eval()

    return actor


def main():
    """Main evaluation function."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    env = InvManagementEnv(render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_TEST_EPISODES = 100

    print("="*60)
    print("ABLATION STUDY: Three-Way Policy Comparison")
    print("="*60)

    # 1. Best Expert (Base Stock Policy)
    target_levels = np.array([80, 180, 40])
    base_stock_policy = BaseStockPolicy(env, z=target_levels)

    def base_stock_callable(obs_val):
        return base_stock_policy.get_action()

    print(f"\n[1/3] Testing Best Expert (z={target_levels})...")
    baseline_rewards = []
    for _ in range(NUM_TEST_EPISODES):
        baseline_rewards.append(run_episode(
            env, agent=None, baseline_policy_callable=base_stock_callable))

    base_mean = np.mean(baseline_rewards)
    base_std = np.std(baseline_rewards)
    print(f"  Mean Reward: {base_mean:.2f} ± {base_std:.2f}")

    # 2. IQL (Mixture) - Original agent
    mixture_checkpoint = "/home/hoang/python/inventory_control/checkpoints/inv_management_iql_minmax_run_14122025_225614/actor/checkpoint_epoch_99.pth"

    print(f"\n[2/3] Testing IQL (Mixture)...")
    iql_mixture = load_agent(mixture_checkpoint, device)

    mixture_rewards = []
    for _ in range(NUM_TEST_EPISODES):
        mixture_rewards.append(run_episode(env, agent=iql_mixture))

    mixture_mean = np.mean(mixture_rewards)
    mixture_std = np.std(mixture_rewards)
    print(f"  Mean Reward: {mixture_mean:.2f} ± {mixture_std:.2f}")

    # 3. IQL (Single Expert) - Ablation
    # Find the latest ablation checkpoint
    ablation_base = "/home/hoang/python/inventory_control/checkpoints"
    ablation_dirs = [d for d in os.listdir(ablation_base)
                     if d.startswith("inv_management_iql_single_expert_ablation")]

    if not ablation_dirs:
        print("\n[3/3] IQL (Single Expert) checkpoint not found!")
        print("  Status: TRAINING REQUIRED")
        print("  Action: Run train_single_expert_ablation.py first")
        single_mean = None
        single_std = None
        single_rewards = None
    else:
        # Use the most recent ablation experiment
        latest_ablation = sorted(ablation_dirs)[-1]
        single_checkpoint = os.path.join(
            ablation_base, latest_ablation, "actor", "checkpoint_epoch_99.pth"
        )

        print(f"\n[3/3] Testing IQL (Single Expert)...")
        iql_single = load_agent(single_checkpoint, device)

        single_rewards = []
        for _ in range(NUM_TEST_EPISODES):
            single_rewards.append(run_episode(env, agent=iql_single))

        single_mean = np.mean(single_rewards)
        single_std = np.std(single_rewards)
        print(f"  Mean Reward: {single_mean:.2f} ± {single_std:.2f}")

    # Statistical comparisons
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)

    # Mixture vs Baseline
    t_stat_mb, p_value_mb = stats.ttest_ind(
        mixture_rewards, baseline_rewards, equal_var=False)
    improvement_mb = ((mixture_mean - base_mean) / abs(base_mean)) * 100

    print(f"\nIQL (Mixture) vs Best Expert:")
    print(
        f"  Improvement: +{mixture_mean - base_mean:.2f} (+{improvement_mb:.1f}%)")
    print(f"  P-value: {p_value_mb:.4e}")
    print(f"  Significant: {'YES' if p_value_mb < 0.05 else 'NO'}")

    if single_mean is not None:
        # Single vs Baseline
        t_stat_sb, p_value_sb = stats.ttest_ind(
            single_rewards, baseline_rewards, equal_var=False)
        improvement_sb = ((single_mean - base_mean) / abs(base_mean)) * 100

        print(f"\nIQL (Single) vs Best Expert:")
        print(
            f"  Improvement: +{single_mean - base_mean:.2f} (+{improvement_sb:.1f}%)")
        print(f"  P-value: {p_value_sb:.4e}")
        print(f"  Significant: {'YES' if p_value_sb < 0.05 else 'NO'}")

        # Mixture vs Single (The Key Test!)
        t_stat_ms, p_value_ms = stats.ttest_ind(
            mixture_rewards, single_rewards, equal_var=False)
        improvement_ms = ((mixture_mean - single_mean) /
                          abs(single_mean)) * 100 if single_mean != 0 else 0

        print(f"\n*** IQL (Mixture) vs IQL (Single) - ABLATION TEST ***")
        print(
            f"  Difference: {mixture_mean - single_mean:+.2f} ({improvement_ms:+.1f}%)")
        print(f"  P-value: {p_value_ms:.4e}")
        print(f"  Significant: {'YES' if p_value_ms < 0.05 else 'NO'}")

        if mixture_mean > single_mean and p_value_ms < 0.05:
            print(
                "\n  ✓ HYPOTHESIS CONFIRMED: Mixture of experts provides significant gain!")
            print("    The 'Implicit Synthesis' effect is real.")
        elif abs(mixture_mean - single_mean) < 5:
            print("\n  ✗ HYPOTHESIS REJECTED: Performance is similar.")
            print("    The gain likely comes from IQL itself, not diversity.")
        else:
            print("\n  ? INCONCLUSIVE: Results need further investigation.")

    # Export results
    ablation_data = {
        "experiment_date": "2026-01-11",
        "num_test_episodes": NUM_TEST_EPISODES,
        "policies": {
            "best_expert": {
                "name": f"Base Stock Policy (z={target_levels.tolist()})",
                "mean_reward": float(base_mean),
                "std_dev": float(base_std)
            },
            "iql_mixture": {
                "name": "IQL trained on Mixture of Experts",
                "mean_reward": float(mixture_mean),
                "std_dev": float(mixture_std),
                "checkpoint": mixture_checkpoint
            }
        },
        "comparisons": {
            "mixture_vs_baseline": {
                "improvement": float(mixture_mean - base_mean),
                "percent_improvement": float(improvement_mb),
                "p_value": float(p_value_mb),
                "statistically_significant": bool(p_value_mb < 0.05)
            }
        }
    }

    if single_mean is not None:
        ablation_data["policies"]["iql_single_expert"] = {
            "name": "IQL trained on Single Best Expert",
            "mean_reward": float(single_mean),
            "std_dev": float(single_std),
            "checkpoint": single_checkpoint
        }
        ablation_data["comparisons"]["single_vs_baseline"] = {
            "improvement": float(single_mean - base_mean),
            "percent_improvement": float(improvement_sb),
            "p_value": float(p_value_sb),
            "statistically_significant": bool(p_value_sb < 0.05)
        }
        ablation_data["comparisons"]["mixture_vs_single"] = {
            "improvement": float(mixture_mean - single_mean),
            "percent_improvement": float(improvement_ms),
            "p_value": float(p_value_ms),
            "statistically_significant": bool(p_value_ms < 0.05),
            "hypothesis_confirmed": bool(mixture_mean > single_mean and p_value_ms < 0.05)
        }

    # Save ablation study
    with open(ABLATION_FILE, "w") as f:
        json.dump(ablation_data, f, indent=4)
    print(f"\n{'-'*60}")
    print(f"Ablation study saved to: {ABLATION_FILE}")

    # Update main results.json
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    else:
        results = {}

    results["ablation"] = {
        "iql_single_expert_mean_reward": float(single_mean) if single_mean else None,
        "iql_single_expert_std_dev": float(single_std) if single_std else None,
        "mixture_advantage": float(mixture_mean - single_mean) if single_mean else None,
        "hypothesis_confirmed": bool(mixture_mean > single_mean and p_value_ms < 0.05) if single_mean else None
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Updated results file: {RESULTS_FILE}")

    print("="*60)


if __name__ == "__main__":
    main()
