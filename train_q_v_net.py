import os
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from utils.config_loader import load_config
from logger_config import setup_logging
from src.models.iql.actor import Actor
from src.models.iql.critics import QNet, VNet
from src.models.iql.agent import IQLAgent
from generate_dataset import generate_dataset
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def main():
    # Load configuration
    config = load_config()

    # Extract hyperparameters
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = float(config['iql']['learning_rate'])
    tau = config['iql']['tau']
    gamma = config['iql']['gamma']
    alpha = config['iql']['alpha']
    beta = config['iql']['beta']

    # --- Dataset Generation and Loading ---
    NUM_EPISODES = 1000
    STEPS_PER_EPISODE = 30
    DATA_DIR = "data"
    DATASET_FILENAME = "inv_management_dataset.pt"
    DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

    if not os.path.exists(DATASET_PATH):
        logger.info(f"Dataset not found at {DATASET_PATH}. Generating new dataset.")
        generate_dataset(NUM_EPISODES, STEPS_PER_EPISODE, DATASET_PATH)
    else:
        logger.info(f"Dataset found at {DATASET_PATH}. Loading existing dataset.")

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
    
    # IMPORTANT: Use specific seed so Actor script can reproduce the same split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    logger.info(f"Data split: {train_size} Training samples, {val_size} Validation samples")

    # Create DataLoaders
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

    v_optimizer = Adam(v_net.parameters(), lr=learning_rate)
    q_optimizer = Adam(q_net.parameters(), lr=learning_rate)
    actor_optimizer = Adam(actor_net.parameters(), lr=learning_rate)

    agent = IQLAgent(
        device=device, actor=actor_net, q_net=q_net, target_net=target_q_net, v_net=v_net,
        tau=tau, gamma=gamma, alpha=alpha, beta=beta,
        v_optimizer=v_optimizer, q_optimizer=q_optimizer, actor_optimizer=actor_optimizer,
        config=config
    )

    # --- Create Experiment ---
    experimental_name = "inv_management_iql_minmax_run"
    base_path = os.getcwd()
    base_logging_path = os.path.join(base_path, "logs")
    base_checkpoint_path = os.path.join(base_path, "checkpoints")
    
    # We explicitly create the experiment here
    agent._create_new_experimental(
        experimental_name=experimental_name,
        base_logging_path=base_logging_path,
        base_checkpoint_path=base_checkpoint_path
    )
    
    # --- Train Q and V ---
    logger.info(f"Starting Q/V training for {epochs} epochs...")
    q_v_metrics = agent.train_q_and_v(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_v_path=None,
        resume_q_path=None
    )
    
    # Export Diagnostics
    agent.export_diagnostics(q_v_metrics, [], file_path="training_diagnostics_qv.csv")
    logger.info("Q/V training completed.")
    
    print("\n" + "="*50)
    print(f"TRAINING COMPLETE. Experiment ID: {agent.log_path.name}")
    print(f"Use this ID to run train_actor.py: --experiment_id {agent.log_path.name}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()