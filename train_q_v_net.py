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

setup_logging()
logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

def main():
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

    rewards = rewards * reward_scale

    full_dataset = TensorDataset(states, actions, rewards, next_states)

    total_size = len(full_dataset)
    train_size = int(validation_split * total_size)
    val_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    logger.info(f"Data split: {train_size} Training samples, {val_size} Validation samples")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

    experimental_name = "inv_management_iql_minmax_run"
    base_path = os.getcwd()
    base_logging_path = os.path.join(base_path, "logs")
    base_checkpoint_path = os.path.join(base_path, "checkpoints")
    
    agent._create_new_experimental(
        experimental_name=experimental_name,
        base_logging_path=base_logging_path,
        base_checkpoint_path=base_checkpoint_path
    )
    
    logger.info(f"Starting Q/V training for {epochs} epochs...")
    q_v_metrics = agent.train_q_and_v(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_v_path=None,
        resume_q_path=None
    )
    
    agent.export_diagnostics(q_v_metrics, [], file_path="training_diagnostics_qv.csv")
    logger.info("Q/V training completed.")
    
    print("\n" + "="*50)
    print(f"TRAINING COMPLETE. Experiment ID: {agent.log_path.name}")
    print(f"Use this ID to run train_actor.py: --experiment_id {agent.log_path.name}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
