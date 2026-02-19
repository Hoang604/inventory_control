"""
Train Behavior Cloning (BC) Baseline

This script trains a BC agent on the same expert mixture dataset used for IQL.
"""
import logging
import torch
import os
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from src.models.bc.agent import BCAgent
from src.models.iql.actor import Actor
from utils.logger_config import setup_logging
from utils.config_loader import load_config

setup_logging()
logger = logging.getLogger(__name__)

def train_bc():
    config = load_config()
    
    # Training parameters
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = float(config['iql']['actor_learning_rate']) # Use same LR as IQL Actor
    validation_split = config['training'].get('validation_split', 0.8)
    
    project_root = Path(__file__).resolve().parents[2]
    
    # Dataset path (Mixture of Experts)
    dataset_path = project_root / "data" / "inv_management_basestock.pt"
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = torch.load(dataset_path, weights_only=False)
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_states = dataset['next_states']
    dones = dataset['dones']
    
    # Episode-level split
    steps_per_episode = config.get('environment', {}).get('days_per_warehouse', 30)
    total_samples = len(states)
    num_episodes = total_samples // steps_per_episode
    
    truncated_samples = num_episodes * steps_per_episode
    states = states[:truncated_samples]
    actions = actions[:truncated_samples]
    rewards = rewards[:truncated_samples]
    next_states = next_states[:truncated_samples]
    dones = dones[:truncated_samples]
    
    train_episodes = int(validation_split * num_episodes)
    train_end_idx = train_episodes * steps_per_episode
    
    train_states = states[:train_end_idx]
    train_actions = actions[:train_end_idx]
    train_rewards = rewards[:train_end_idx]
    train_next_states = next_states[:train_end_idx]
    train_dones = dones[:train_end_idx]

    val_states = states[train_end_idx:]
    val_actions = actions[train_end_idx:]
    val_rewards = rewards[train_end_idx:]
    val_next_states = next_states[train_end_idx:]
    val_dones = dones[train_end_idx:]
    
    train_dataset = TensorDataset(train_states, train_actions, train_rewards, train_next_states, train_dones)
    val_dataset = TensorDataset(val_states, val_actions, val_rewards, val_next_states, val_dones)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize Actor and Agent
    actor_net = Actor(config).to(device)
    optimizer = Adam(actor_net.parameters(), lr=learning_rate)
    
    agent = BCAgent(
        device=device,
        actor=actor_net,
        actor_optimizer=optimizer,
        config=config
    )
    
    logger.info("Starting Behavior Cloning training...")
    agent.train(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        experimental_name="inv_management_bc_baseline"
    )
    logger.info("BC training completed.")

if __name__ == "__main__":
    train_bc()
