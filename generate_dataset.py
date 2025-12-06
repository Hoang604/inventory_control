import torch
import numpy as np
import tqdm
import os
from src.base.inv_management_env import InvManagementEnv
from src.base.policies import MinMaxPolicy

def generate_dataset(num_episodes: int, steps_per_episode: int, save_path: str):
    """
    Generates a dataset using a Randomized Min-Max (s, S) Policy.
    
    For each episode, we sample random (s, S) parameters for each stage.
    This creates a dataset of diverse inventory management behaviors,
    ranging from lean (low s, low S) to hoarder (high s, high S) strategies.
    """
    env = InvManagementEnv(max_steps_per_episode=steps_per_episode, render_mode=None)
    policy = MinMaxPolicy(env)

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []

    print(f"Generating dataset with Randomized Min-Max (s, S) Policy for {num_episodes} episodes...")

    for episode in tqdm.tqdm(range(num_episodes)):
        obs, info = env.reset()
        current_state = obs
        
        # Sample random (s, S) parameters for this episode
        # S: Order-Up-To level (Target Echelon Inventory)
        # s: Reorder Point (Trigger level)
        
        # Heuristic logic:
        # S should cover demand during lead time + safety stock.
        # Randomize S between 10 and 200.
        S_levels = sorted(np.random.randint(10, 200, size=3))
        
        # s must be less than S. Randomize s between 0 and S (exclusive).
        s_levels = [np.random.randint(0, S) for S in S_levels]
        
        # Combine into params array: shape (3, 2) -> [[s0, S0], [s1, S1], [s2, S2]]
        policy_params = np.column_stack((s_levels, S_levels))

        for step in range(steps_per_episode):
            # Get action from heuristic
            action = policy.get_action(params=policy_params)
            
            # Add exploration noise to create a richer dataset for IQL
            # IQL needs to see "nearby" actions to estimate values correctly.
            noise = np.random.normal(0, 5, size=action.shape)
            action = action + noise
            action = np.clip(action, 0, env.supply_capacity)
            
            # Step the environment
            next_obs, reward, done, _, info = env.step(action)

            # Store the transition
            all_states.append(current_state)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_obs)

            current_state = next_obs

            if done:
                break
    
    # Convert lists to PyTorch Tensors
    states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32)
    actions_tensor = torch.tensor(np.array(all_actions), dtype=torch.float32)
    rewards_tensor = torch.tensor(np.array(all_rewards), dtype=torch.float32).unsqueeze(1)
    next_states_tensor = torch.tensor(np.array(all_next_states), dtype=torch.float32)

    dataset = {
        'states': states_tensor,
        'actions': actions_tensor,
        'rewards': rewards_tensor,
        'next_states': next_states_tensor
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, save_path)
    print(f"Min-Max Dataset generated and saved to {save_path}")
    print(f"Dataset size: {len(all_states)} transitions")

if __name__ == "__main__":
    # Example usage:
    NUM_EPISODES = 2000 
    STEPS_PER_EPISODE = 30
    DATASET_DIR = "data"
    DATASET_FILENAME = "inv_management_dataset.pt"
    SAVE_PATH = os.path.join(DATASET_DIR, DATASET_FILENAME)

    generate_dataset(NUM_EPISODES, STEPS_PER_EPISODE, SAVE_PATH)
