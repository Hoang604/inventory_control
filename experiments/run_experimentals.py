import os
import torch
import logging
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.config_loader import load_config
from utils.logger_config import setup_logging
from src.models.iql.actor import Actor
from src.models.iql.critics import QNet, VNet
from src.models.iql.agent import IQLAgent
from scripts.data_generation.generate_dataset import generate_base_stock_dataset as generate_dataset

setup_logging()
logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


def update_recursive(d, u):
    """Recursively update dictionary d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def run_single_experiment(experiment_name, config_overrides):
    """
    Runs a single IQL training experiment.

    Args:
        experiment_name (str): Unique identifier for the experiment (used for folder names).
        config_overrides (dict): Dictionary containing config parameters to override.
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"STARTING EXPERIMENT: {experiment_name}")
    logger.info(f"{'='*50}")

    config = load_config()
    config = update_recursive(config, config_overrides)

    logger.info(f"Configuration overrides for this run: {config_overrides}")

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
        logger.info(
            f"Dataset not found at {DATASET_PATH}. Generating new dataset.")
        generate_dataset(NUM_EPISODES, STEPS_PER_EPISODE, DATASET_PATH)

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

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

    base_path = os.getcwd()
    base_logging_path = os.path.join(base_path, "logs")
    base_checkpoint_path = os.path.join(base_path, "checkpoints")

    agent._create_new_experimental(
        experimental_name=experiment_name,
        base_logging_path=base_logging_path,
        base_checkpoint_path=base_checkpoint_path
    )

    logger.info(
        f"[{experiment_name}] Starting Q/V training for {epochs} epochs...")
    q_v_metrics = agent.train_q_and_v(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_v_path=None,
        resume_q_path=None
    )
    agent.export_diagnostics(
        q_v_metrics, [], file_path="training_diagnostics_qv.csv")

    logger.info(f"[{experiment_name}] Reloading best Q/V checkpoints...")

    best_q_path = agent.checkpoint_path / "q_net" / "best_loss.pth"
    best_v_path = agent.checkpoint_path / "v_net" / "best_loss.pth"

    if best_q_path.exists():
        agent._load_model('q_net', str(best_q_path))
        agent.target_net.load_state_dict(
            agent.q_net.state_dict())
    else:
        logger.warning(f"Best Q-Net not found. Using final weights.")

    if best_v_path.exists():
        agent._load_model('v_net', str(best_v_path))
    else:
        logger.warning(f"Best V-Net not found. Using final weights.")

    logger.info(f"[{experiment_name}] Starting Actor training...")
    actor_metrics = agent.train_actor(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_training_path=None
    )
    agent.export_diagnostics(
        [], actor_metrics, file_path="training_diagnostics_actor.csv")

    logger.info(f"FINISHED EXPERIMENT: {experiment_name}")


def main():
    experiments = [
        {
            "name": "EXP_01_NET_MEDIUM",
            "config": {"intermediate_dim": 256}
        },
        {
            "name": "EXP_01_NET_LARGE",
            "config": {"intermediate_dim": 512}
        },
        {
            "name": "EXP_01_NET_VERY_LARGE",
            "config": {"intermediate_dim": 1024}
        },
        {
            "name": "EXP_01_NET_HUGE",
            "config": {"intermediate_dim": 2048}
        }
    ]

    logger.info(f"Found {len(experiments)} experiments to run.")

    for i, exp in enumerate(experiments):
        logger.info(
            f"Running Experiment {i+1}/{len(experiments)}: {exp['name']}")
        try:
            run_single_experiment(exp["name"], exp["config"])
        except Exception as e:
            logger.info(f"Experiment {exp['name']} FAILED with error: {e}")
            continue


if __name__ == "__main__":
    main()
