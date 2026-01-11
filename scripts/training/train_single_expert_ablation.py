"""
Train IQL Agent on Single-Expert Dataset (Ablation Study)

This script trains an IQL agent on the single-expert dataset to test the
"Implicit Synthesis" hypothesis. Results will show whether IQL's performance
gains come from:
  A) Learning from diverse experts (Mixture of Experts dataset)
  B) The IQL algorithm itself (Single Expert dataset)

Expected Result:
  - If IQL(mixture) >> IQL(single): Proves "Implicit Synthesis" hypothesis
  - If IQL(mixture) ≈ IQL(single): Suggests gain comes from IQL, not diversity

Output: checkpoints/inv_management_iql_single_expert_ablation/
"""
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

setup_logging()
logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


def main():
    """Main training function for single-expert ablation study."""
    config = load_config()

    # Training hyperparameters
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

    # Dataset paths
    DATA_DIR = "data"
    DATASET_FILENAME = "inv_management_single_expert.pt"
    DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Single-expert dataset not found at {DATASET_PATH}. "
            f"Please run generate_single_expert_dataset.py first."
        )

    logger.info("="*60)
    logger.info("ABLATION STUDY: Training IQL on Single-Expert Dataset")
    logger.info("="*60)
    logger.info(f"Loading dataset from: {DATASET_PATH}")

    # Load dataset
    dataset = torch.load(DATASET_PATH)
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_states = dataset['next_states']
    dones = dataset['dones']

    logger.info(f"Dataset size: {len(states)} transitions")
    logger.info(f"Mean reward: {rewards.mean().item():.4f}")

    # Scale rewards
    rewards = rewards * reward_scale

    # Create train/val split
    full_dataset = TensorDataset(states, actions, rewards, next_states, dones)

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

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize networks
    actor_net = Actor(config).to(device)
    q_net = QNet(config).to(device)
    target_q_net = QNet(config).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    v_net = VNet(config).to(device)

    # Initialize optimizers
    v_optimizer = Adam(v_net.parameters(), lr=learning_rate)
    q_optimizer = Adam(q_net.parameters(), lr=learning_rate)
    actor_optimizer = Adam(actor_net.parameters(), lr=learning_rate)

    # Create IQL agent
    agent = IQLAgent(
        device=device,
        actor=actor_net,
        q_net=q_net,
        target_net=target_q_net,
        v_net=v_net,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        v_optimizer=v_optimizer,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        config=config
    )

    # Setup experiment paths
    experimental_name = "inv_management_iql_single_expert_ablation"
    base_path = os.getcwd()
    base_logging_path = os.path.join(base_path, "logs")
    base_checkpoint_path = os.path.join(base_path, "checkpoints")

    agent._create_new_experimental(
        experimental_name=experimental_name,
        base_logging_path=base_logging_path,
        base_checkpoint_path=base_checkpoint_path
    )

    logger.info("="*60)
    logger.info("PHASE 1: Training Q-Network and V-Network")
    logger.info("="*60)

    # Train Q and V networks
    logger.info(f"Starting Q/V training for {epochs} epochs...")
    q_v_metrics = agent.train_q_and_v(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_v_path=None,
        resume_q_path=None
    )
    agent.export_diagnostics(
        q_v_metrics, [], file_path="training_diagnostics_qv_ablation.csv")
    logger.info(
        "Q/V training completed. Best Critics selected by Stability Score.")

    # Reload best Q and V networks
    logger.info(
        "Reloading 'best_loss.pth' for Q and V networks before training Actor...")

    best_q_path = agent.checkpoint_path / "q_net" / "best_loss.pth"
    best_v_path = agent.checkpoint_path / "v_net" / "best_loss.pth"

    if best_q_path.exists():
        agent._load_model('q_net', str(best_q_path))
        agent.target_q_net.load_state_dict(agent.q_net.state_dict())
        logger.info("Target Network synced with Best Q-Network.")
    else:
        logger.warning(
            f"Best Q-Net checkpoint not found at {best_q_path}. Actor training may be suboptimal.")

    if best_v_path.exists():
        agent._load_model('v_net', str(best_v_path))
    else:
        logger.warning(
            f"Best V-Net checkpoint not found at {best_v_path}. Actor training may be suboptimal.")

    logger.info("="*60)
    logger.info("PHASE 2: Training Actor Network")
    logger.info("="*60)

    # Train actor network
    logger.info(f"Starting Actor training for {epochs} epochs...")
    actor_metrics = agent.train_actor(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_training_path=None
    )
    agent.export_diagnostics(
        [], actor_metrics, file_path="training_diagnostics_actor_ablation.csv")
    logger.info(
        "Actor training completed. Best Actor selected by Estimated Policy Value.")

    logger.info("="*60)
    logger.info("ABLATION TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Experiment ID: {agent.log_path.name}")
    logger.info(f"Checkpoints saved to: {agent.checkpoint_path}")
    logger.info(f"Logs saved to: {agent.log_path}")
    logger.info("="*60)
    logger.info("\nNext Steps:")
    logger.info("1. Run test_agent_ablation.py to evaluate performance")
    logger.info("2. Compare results with mixture-trained IQL agent")
    logger.info("="*60)


if __name__ == "__main__":
    main()
