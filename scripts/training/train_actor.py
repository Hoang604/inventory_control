"""
Train Actor Network

This module trains the Actor (policy) network for IQL using pre-trained Q/V networks.
Can be used standalone with CLI args or imported for pipeline orchestration.
"""
from src.models.iql.agent import IQLAgent
from src.models.iql.critics import QNet, VNet
from src.models.iql.actor import Actor
from utils.logger_config import setup_logging
from utils.config_loader import load_config
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import argparse
import torch
import os
from pathlib import Path

setup_logging()
logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


def train_actor_network(experiment_id, dataset_path=None):
    """
    Train the Actor network using pre-trained Q and V networks.

    Args:
        experiment_id: Experiment directory name (in checkpoints/ and logs/)
        dataset_path: Path to dataset file (default: data/inv_management_base_stock.pt)

    Returns:
        bool: True if training succeeded, False otherwise
    """
    config = load_config()

    # Training parameters
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = float(config['iql']['actor_learning_rate'])
    tau = config['iql']['tau']
    gamma = config['iql']['gamma']
    alpha = config['iql']['alpha']
    beta = config['iql']['beta']

    reward_scale = config['training'].get('reward_scale', 0.1)
    validation_split = config['training'].get('validation_split', 0.8)
    seed = config['training'].get('seed', 42)

    project_root = Path(__file__).resolve().parents[2]

    # Dataset configuration from config.yaml
    if dataset_path is None:
        dataset_config = config.get('dataset', {})
        explicit_path = dataset_config.get('path')
        
        if explicit_path:
            dataset_path = str(project_root / explicit_path)
        else:
            dataset_name = dataset_config.get('name', 'continuing')
            dataset_mapping = {
                'continuing': 'inv_management_continuing.pt',
                'base_stock': 'inv_management_base_stock.pt',
                'single_expert': 'inv_management_single_expert.pt',
                'multi_policy': 'inv_management_multi_policy.pt',
            }
            dataset_file = dataset_mapping.get(dataset_name, f'inv_management_{dataset_name}.pt')
            dataset_path = str(project_root / "data" / dataset_file)

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return False

    dataset = torch.load(dataset_path, weights_only=False)
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards'] * reward_scale
    next_states = dataset['next_states']
    dones = dataset['dones']

    # Episode-level data splitting (preserves episode structure for Spearman correlation)
    steps_per_episode = config.get('environment', {}).get('days_per_warehouse', 30)
    total_samples = len(states)
    num_episodes = total_samples // steps_per_episode

    # Truncate to exact multiple of steps_per_episode
    truncated_samples = num_episodes * steps_per_episode
    states = states[:truncated_samples]
    actions = actions[:truncated_samples]
    rewards = rewards[:truncated_samples]
    next_states = next_states[:truncated_samples]
    dones = dones[:truncated_samples]

    # Split at EPISODE level (not sample level)
    train_episodes = int(validation_split * num_episodes)
    val_episodes = num_episodes - train_episodes
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

    logger.info(f"Episode-level split: {train_episodes} train episodes ({len(train_dataset)} samples), "
                f"{val_episodes} val episodes ({len(val_dataset)} samples)")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # Setup paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    base_path = project_root
    experiment_checkpoint_path = base_path / "checkpoints" / experiment_id
    experiment_log_path = base_path / "logs" / experiment_id

    if not experiment_checkpoint_path.exists():
        logger.error(
            f"Checkpoint path not found: {experiment_checkpoint_path}")
        return False

    # Load pre-trained Q and V networks
    best_q_path = experiment_checkpoint_path / "q_net" / "best_loss.pth"
    best_v_path = experiment_checkpoint_path / "v_net" / "best_loss.pth"

    if not best_q_path.exists():
        logger.error(f"Q-Net checkpoint not found at {best_q_path}")
        return False

    if not best_v_path.exists():
        logger.error(f"V-Net checkpoint not found at {best_v_path}")
        return False

    q_checkpoint_data = torch.load(best_q_path, map_location=device, weights_only=False)
    v_checkpoint_data = torch.load(best_v_path, map_location=device, weights_only=False)
    logger.info("Loaded Q and V network checkpoints")

    # Use config from checkpoints if available
    q_v_net_config = (
        q_checkpoint_data.get('config') or
        v_checkpoint_data.get('config') or
        config
    )

    # Initialize networks
    q_net = QNet(q_v_net_config).to(device)
    target_q_net = QNet(q_v_net_config).to(device)
    v_net = VNet(q_v_net_config).to(device)

    q_net.load_state_dict(q_checkpoint_data['model_state_dict'])
    target_q_net.load_state_dict(q_net.state_dict())
    v_net.load_state_dict(v_checkpoint_data['model_state_dict'])
    logger.info("Q/V networks initialized with pre-trained weights")

    actor_net = Actor(config).to(device)

    v_optimizer = Adam(v_net.parameters(), lr=learning_rate)
    q_optimizer = Adam(q_net.parameters(), lr=learning_rate)
    actor_optimizer = Adam(actor_net.parameters(), lr=learning_rate)

    # Create agent
    agent = IQLAgent(
        device=device, actor=actor_net, q_net=q_net,
        target_net=target_q_net, v_net=v_net,
        tau=tau, gamma=gamma, alpha=alpha, beta=beta,
        v_optimizer=v_optimizer, q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        config=config
    )

    agent.log_path = experiment_log_path
    agent.checkpoint_path = experiment_checkpoint_path
    agent.writer = SummaryWriter(log_dir=agent.log_path)

    logger.info(f"Resuming experiment: {experiment_id}")

    # Train actor
    logger.info(f"Starting Actor training for {epochs} epochs...")
    actor_metrics = agent.train_actor(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_training_path=None
    )

    agent.export_diagnostics(
        [], actor_metrics, file_path="training_diagnostics_actor.csv")
    logger.info("Actor training completed")

    return True


def main():
    """CLI entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Train Actor network for IQL Agent")
    parser.add_argument(
        "--experiment", type=str, required=True,
        help="Experiment folder name in checkpoints/ and logs/")
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Training Actor Network")
    logger.info("="*60)
    logger.info(f"Experiment ID: {args.experiment}")

    success = train_actor_network(args.experiment)

    if success:
        logger.info("\n" + "="*60)
        logger.info("ACTOR TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Checkpoints: checkpoints/{args.experiment}/actor/")
        logger.info("="*60)
    else:
        logger.error("Actor training failed")


if __name__ == "__main__":
    main()
