import torch
import os

def inspect_dataset(dataset_path: str):
    """
    Loads the inventory management dataset and prints statistics and samples of rewards.

    Args:
        dataset_path (str): The path to the inv_management_dataset.pt file.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = torch.load(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the file is a valid PyTorch tensor file.")
        return

    if 'rewards' in dataset:
        rewards = dataset['rewards']
        print(f"\n--- Rewards Data ---")
        print(f"Type: {type(rewards)}")
        print(f"Shape: {rewards.shape}")
        print(f"Mean: {rewards.mean().item():.4f}")
        print(f"Std Dev: {rewards.std().item():.4f}")
        print(f"Min: {rewards.min().item():.4f}")
        print(f"Max: {rewards.max().item():.4f}")

        # Print first 10 sample rewards
        print(f"\nFirst 10 sample rewards:")
        for i in range(min(10, rewards.shape[0])):
            print(f"  {rewards[i].item():.4f}")

        # Print last 10 sample rewards
        if rewards.shape[0] > 10:
            print(f"\nLast 10 sample rewards:")
            for i in range(rewards.shape[0] - 10, rewards.shape[0]):
                print(f"  {rewards[i].item():.4f}")

    else:
        print("Key 'rewards' not found in the dataset.")
        print(f"Available keys: {dataset.keys()}")

if __name__ == "__main__":
    DATA_DIR = "data"
    DATASET_FILENAME = "inv_management_dataset.pt"
    DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

    inspect_dataset(DATASET_PATH)
