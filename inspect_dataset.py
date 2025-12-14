import torch
import os


def calculate_ground_truth(dataset_path, gamma=0.99, steps_per_episode=30):
    print(f"Loading {dataset_path}...")
    try:
        dataset = torch.load(dataset_path)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    rewards = dataset['rewards'].flatten()
    total_samples = len(rewards)

    num_episodes = total_samples // steps_per_episode

    print(f"Detected {total_samples} samples.")
    print(
        f"Assuming {steps_per_episode} steps/episode -> {num_episodes} episodes.")

    episode_returns = []

    for i in range(num_episodes):
        start_idx = i * steps_per_episode
        end_idx = start_idx + steps_per_episode

        ep_rewards = rewards[start_idx:end_idx]

        g_0 = 0
        current_gamma = 1.0
        for r in ep_rewards:
            g_0 += r.item() * current_gamma
            current_gamma *= gamma

        episode_returns.append(g_0)

    episode_returns = torch.tensor(episode_returns)

    print("\n=== GROUND TRUTH STATISTICS ===")
    print(
        f"True Mean Return (Target for Q-Mean): {episode_returns.mean().item():.4f}")
    print(
        f"True Return Std (Target for Q-Std):   {episode_returns.std().item():.4f}")
    print(
        f"Min Episode Return:                   {episode_returns.min().item():.4f}")
    print(
        f"Max Episode Return:                   {episode_returns.max().item():.4f}")
    print("===================================")

    return episode_returns.mean().item()


if __name__ == "__main__":
    calculate_ground_truth(
        "data/inv_management_base_stock.pt",
        gamma=0.99,
        steps_per_episode=30
    )
