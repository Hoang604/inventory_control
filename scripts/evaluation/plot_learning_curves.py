import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_learning_curves(csv_path="paper/artifacts/training_history.csv", output_dir="paper/artifacts"):
    """
    Plots Q-loss, V-loss, and Actor-loss from the training history.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot Q-Loss
    axes[0].plot(df['epoch'], df['val_q_loss'], label='Val Q-Loss', color='blue')
    axes[0].set_title('Q-Network Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # Plot V-Loss
    axes[1].plot(df['epoch'], df['val_v_loss'], label='Val V-Loss', color='green')
    axes[1].set_title('V-Network Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    # Plot Actor-Loss
    axes[2].plot(df['epoch'], df['val_actor_loss'], label='Val Actor Loss', color='red')
    axes[2].set_title('Actor Validation Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].legend()

    plt.tight_layout()
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "learning_curves.png")
    pdf_path = os.path.join(output_dir, "learning_curves.pdf")
    
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    
    print(f"Plots saved to {png_path} and {pdf_path}")
    plt.close()

if __name__ == "__main__":
    plot_learning_curves()
