"""
Train Q-Network and V-Network

This module trains the Q and V critic networks for IQL.
Can be used standalone or imported by main.py for pipeline orchestration.
"""
from src.models.iql.agent import IQLAgent
from src.models.iql.critics import QNet, VNet
from src.models.iql.actor import Actor
from utils.logger_config import setup_logging
from utils.config_loader import load_config
import logging
from torch.utils.data import TensorDataset, DataLoader
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
            dataset_file = dataset_mapping.get(
                dataset_name, f'inv_management_{dataset_name}.pt')
            dataset_path = str(project_root / "data" / dataset_file)

    if not os.path.exists(dataset_path):
        logger.error(
            f"Dataset not found at {dataset_path}. "
            f"Please generate the dataset first."
        )
        return None

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = torch.load(dataset_path, weights_only=False)
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards'] * reward_scale
    next_states = dataset['next_states']
    dones = dataset['dones']

    # Episode-level data splitting (preserves episode structure for Spearman correlation)
    steps_per_episode = config.get(
        'environment', {}).get('days_per_warehouse', 30)
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

    train_dataset = TensorDataset(
        train_states, train_actions, train_rewards, train_next_states, train_dones)
    val_dataset = TensorDataset(
        val_states, val_actions, val_rewards, val_next_states, val_dones)

    logger.info(f"Episode-level split: {train_episodes} train episodes ({len(train_dataset)} samples), "
                f"{val_episodes} val episodes ({len(val_dataset)} samples)")

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
