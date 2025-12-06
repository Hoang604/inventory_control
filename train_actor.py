import os
import torch
import argparse
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from utils.config_loader import load_config
from logger_config import setup_logging
from src.models.iql.actor import Actor
from src.models.iql.critics import QNet, VNet
from src.models.iql.agent import IQLAgent
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser(description="Train Actor network for IQL Agent")
    parser.add_argument("--experiment_id", type=str, required=True, 
                        help="The folder name in checkpoints/ logs/ (e.g. inv_management_iql_minmax_run_06122025_180002)")
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # --- Dataset Generation and Loading ---
    # (Must match train_q_v_net.py exactly)
    DATA_DIR = "data"
    DATASET_FILENAME = "inv_management_dataset.pt"
    DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run train_q_v_net.py or generate_dataset.py first.")

    dataset = torch.load(DATASET_PATH)
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_states = dataset['next_states']

    # Apply reward scaling
    rewards = rewards / 10.0

    # Create Full TensorDataset
    full_dataset = TensorDataset(states, actions, rewards, next_states)

    # --- Data Split (Fixed Seed for Reproducibility) ---
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # IMPORTANT: Use SAME seed as train_q_v_net.py
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    logger.info(f"Data split: {train_size} Training samples, {val_size} Validation samples")

    # Create DataLoaders
    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- IQL Agent Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    actor_net = Actor(config).to(device)
    q_net = QNet(config).to(device)
    target_q_net = QNet(config).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    v_net = VNet(config).to(device)

    learning_rate = 1e-5
    v_optimizer = Adam(v_net.parameters(), lr=learning_rate)
    q_optimizer = Adam(q_net.parameters(), lr=learning_rate)
    actor_optimizer = Adam(actor_net.parameters(), lr=learning_rate)

    agent = IQLAgent(
        device=device, actor=actor_net, q_net=q_net, target_net=target_q_net, v_net=v_net,
        tau=0.7, gamma=0.99, alpha=0.005, beta=1.0,
        v_optimizer=v_optimizer, q_optimizer=q_optimizer, actor_optimizer=actor_optimizer,
        config=config
    )

    # --- Setup Existing Experiment Paths ---
    base_path = os.getcwd()
    agent.log_path = Path(base_path) / "logs" / args.experiment_id
    agent.checkpoint_path = Path(base_path) / "checkpoints" / args.experiment_id
    
    if not agent.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {agent.checkpoint_path}")

    agent.writer = SummaryWriter(log_dir=agent.log_path)
    logger.info(f"Resuming experiment: {args.experiment_id}")

    # --- RELOAD BEST CHECKPOINTS ---
    logger.info("Reloading 'best_loss.pth' for Q and V networks...")
    
    best_q_path = agent.checkpoint_path / "q_net" / "best_loss.pth"
    best_v_path = agent.checkpoint_path / "v_net" / "best_loss.pth"

    if best_q_path.exists():
        agent._load_model('q_net', str(best_q_path))
        # Sync target net with the best Q-net
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        logger.info("Target Network synced with Best Q-Network.")
    else:
        logger.warning(f"Best Q-Net checkpoint not found at {best_q_path}. STARTING WITH RANDOM Q-NET (Are you sure?)")

    if best_v_path.exists():
        agent._load_model('v_net', str(best_v_path))
    else:
        logger.warning(f"Best V-Net checkpoint not found at {best_v_path}. STARTING WITH RANDOM V-NET")
        
    # --- Train Actor ---
    epochs = 100
    logger.info(f"Starting Actor training for {epochs} epochs...")
    
    actor_metrics = agent.train_actor(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_training_path=None
    )
    
    # Export Diagnostics (Saving specifically for actor)
    agent.export_diagnostics([], actor_metrics, file_path="training_diagnostics_actor.csv")
    logger.info("Actor training completed.")

if __name__ == "__main__":
    main()
