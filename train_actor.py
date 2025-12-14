import os
import torch
import argparse
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from utils.config_loader import load_config
from logger_config import setup_logging
from src.models.iql.actor import Actor, SActor
from src.models.iql.critics import QNet, VNet
from src.models.iql.agent import IQLAgent
import logging

setup_logging()
logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


def main():
    parser = argparse.ArgumentParser(
        description="Train Actor network for IQL Agent")
    parser.add_argument("--experiment_id", type=str, required=True,
                        help="The folder name in checkpoints/ logs/ (e.g. inv_management_iql_minmax_run_06122025_180002)")
    args = parser.parse_args()

    config = load_config()

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

    DATA_DIR = "data"
    DATASET_FILENAME = "inv_management_dataset.pt"
    DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run train_q_v_net.py or generate_dataset.py first.")

    dataset = torch.load(DATASET_PATH)
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_states = dataset['next_states']

    rewards = rewards * reward_scale

    full_dataset = TensorDataset(states, actions, rewards, next_states)

    total_size = len(full_dataset)
    train_size = int(validation_split * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator)

    logger.info(
        f"Data split: {train_size} Training samples, {val_size} Validation samples")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    base_path = os.getcwd()
    experiment_checkpoint_path = Path(
        base_path) / "checkpoints" / args.experiment_id
    experiment_log_path = Path(base_path) / "logs" / args.experiment_id

    if not experiment_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path not found: {experiment_checkpoint_path}")

    best_q_path = experiment_checkpoint_path / "q_net" / "best_loss.pth"
    best_v_path = experiment_checkpoint_path / "v_net" / "best_loss.pth"

    q_checkpoint_data = None
    v_checkpoint_data = None

    if best_q_path.exists():
        q_checkpoint_data = torch.load(best_q_path, map_location=device)
        logger.info(f"Loaded Q-Net checkpoint from {best_q_path}")
    else:
        logger.error(
            f"Best Q-Net checkpoint not found at {best_q_path}. Cannot proceed without a trained Q-Net.")
        return

    if best_v_path.exists():
        v_checkpoint_data = torch.load(best_v_path, map_location=device)
        logger.info(f"Loaded V-Net checkpoint from {best_v_path}")
    else:
        logger.error(
            f"Best V-Net checkpoint not found at {best_v_path}. Cannot proceed without a trained V-Net.")
        return

    q_v_net_config = None
    if 'config' in q_checkpoint_data:
        q_v_net_config = q_checkpoint_data['config']
        logger.info(
            "Using configuration from Q-Net checkpoint for Q/V network initialization.")
    elif 'config' in v_checkpoint_data:
        q_v_net_config = v_checkpoint_data['config']
        logger.info(
            "Using configuration from V-Net checkpoint for Q/V network initialization.")
    else:
        q_v_net_config = config
        logger.warning(
            "Config not found in Q/V checkpoints. Falling back to initially loaded config for Q/V network initialization.")

    q_net = QNet(q_v_net_config).to(device)
    target_q_net = QNet(q_v_net_config).to(device)
    v_net = VNet(q_v_net_config).to(device)

    q_net.load_state_dict(q_checkpoint_data['model_state_dict'])
    target_q_net.load_state_dict(q_net.state_dict())
    v_net.load_state_dict(v_checkpoint_data['model_state_dict'])
    logger.info(
        "Q and V networks successfully initialized and state dictionaries loaded.")

    actor_net = SActor(config).to(device)

    v_optimizer = Adam(v_net.parameters(), lr=learning_rate)
    q_optimizer = Adam(q_net.parameters(), lr=learning_rate)
    actor_optimizer = Adam(actor_net.parameters(), lr=learning_rate)

    agent = IQLAgent(
        device=device, actor=actor_net, q_net=q_net, target_net=target_q_net, v_net=v_net,
        tau=tau, gamma=gamma, alpha=alpha, beta=beta,
        v_optimizer=v_optimizer, q_optimizer=q_optimizer, actor_optimizer=actor_optimizer,
        config=config
    )

    agent.log_path = experiment_log_path
    agent.checkpoint_path = experiment_checkpoint_path

    if not agent.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path not found: {agent.checkpoint_path}")

    agent.writer = SummaryWriter(log_dir=agent.log_path)
    logger.info(
        f"Resuming experiment: {args.experiment_id}; Logging to: {agent.log_path}; Checkpoints in: {agent.checkpoint_path}")

    logger.info(f"Starting Actor training for {epochs} epochs...")

    actor_metrics = agent.train_actor(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_training_path=None
    )

    agent.export_diagnostics(
        [], actor_metrics, file_path="training_diagnostics_actor.csv")
    logger.info("Actor training completed.")


if __name__ == "__main__":
    main()