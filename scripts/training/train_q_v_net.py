"""
Train Q-Network and V-Network

This module trains the Q and V critic networks for IQL.
Can be used standalone or imported by main.py for pipeline orchestration.
"""
from scripts.data_generation.generate_dataset import generate_base_stock_dataset
from src.models.iql.agent import IQLAgent
from src.models.iql.critics import QNet, VNet
from src.models.iql.actor import Actor
from utils.logger_config import setup_logging
from utils.config_loader import load_config
import logging
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import Adam
import torch
import os
from pathlib import Path

setup_logging()
logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


def train_qv_networks(dataset_path=None, experiment_name=None):
    """
    Train Q and V networks and return the experiment ID.

    Args:
        dataset_path: Path to dataset file (default: data/inv_management_base_stock.pt)
        experiment_name: Name for the experiment (default: inv_management_iql_minmax_run)

    Returns:
        str: Experiment ID (directory name in checkpoints/) or None if training failed
    """
    config = load_config()

    # Training parameters
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = float(config['iql']['learning_rate'])
    tau = config['iql']['tau']
    gamma = config['iql']['gamma']
    alpha = config['iql']['alpha']
    beta = config['iql']['beta']

    reward_scale = config['training'].get('reward_scale', 0.1)
    validation_split = config['training'].get('validation_split', 0.8)
    seed = config['training'].get('seed', 42)

    project_root = Path(__file__).resolve().parents[2]
    # Dataset configuration
    if dataset_path is None:
        dataset_path = str(project_root / "data" / "inv_management_base_stock.pt")

    if not os.path.exists(dataset_path):
        logger.info(
            f"Dataset not found at {dataset_path}. Generating dataset...")
        generate_base_stock_dataset(
            num_episodes=2000,
            steps_per_episode=30,
            save_path=dataset_path
        )

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = torch.load(dataset_path)
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards'] * reward_scale
    next_states = dataset['next_states']
    dones = dataset['dones']

    # Data preparation
    full_dataset = TensorDataset(states, actions, rewards, next_states, dones)
    total_size = len(full_dataset)
    train_size = int(validation_split * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator)

    logger.info(f"Data split: {train_size} train, {val_size} validation")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    actor_net = Actor(config).to(device)
    q_net = QNet(config).to(device)
    target_q_net = QNet(config).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    v_net = VNet(config).to(device)

    v_optimizer = Adam(v_net.parameters(), lr=learning_rate)
    q_optimizer = Adam(q_net.parameters(), lr=learning_rate)
    actor_optimizer = Adam(actor_net.parameters(), lr=learning_rate)

    agent = IQLAgent(
        device=device, actor=actor_net, q_net=q_net,
        target_net=target_q_net, v_net=v_net,
        tau=tau, gamma=gamma, alpha=alpha, beta=beta,
        v_optimizer=v_optimizer, q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        config=config
    )

    # Setup experiment directories
    if experiment_name is None:
        experiment_name = "inv_management_iql_minmax_run"

    base_path = project_root
    agent._create_new_experimental(
        experimental_name=experiment_name,
        base_logging_path=base_path / "logs",
        base_checkpoint_path=base_path / "checkpoints"
    )

    # Training
    logger.info(f"Starting Q/V training for {epochs} epochs...")
    q_v_metrics = agent.train_q_and_v(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_v_path=None,
        resume_q_path=None
    )

    agent.export_diagnostics(
        q_v_metrics, [], file_path="training_diagnostics_qv.csv")
    logger.info("Q/V training completed")

    return agent.log_path.name  # Return experiment ID


def main():
    """CLI entry point for standalone execution."""
    logger.info("="*60)
    logger.info("Training Q-Network and V-Network")
    logger.info("="*60)

    experiment_id = train_qv_networks()

    logger.info("\n" + "="*60)
    logger.info(f"Q/V TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"\nNext step: Train actor with:")
    logger.info(
        f"  python3 scripts/training/train_actor.py --experiment_id {experiment_id}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
