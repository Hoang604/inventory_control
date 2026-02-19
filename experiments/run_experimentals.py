"""
Run IQL experiments across all policy datasets.

Tests the IQL agent on datasets generated from different inventory policies:
- Base-Stock, Min-Max, Lot-for-Lot, Periodic Review, (R,Q), Noisy Base-Stock
"""
import os
import torch
import logging
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from utils.config_loader import load_config
from utils.logger_config import setup_logging
from src.models.iql.actor import Actor
from src.models.iql.critics import QNet, VNet
from src.models.iql.agent import IQLAgent

setup_logging()
logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Available datasets based on files in data/ folder
DATASETS = {
    "basestock": {
        "path": "data/inv_management_basestock.pt",
        "description": "Base-Stock policy",
    },
    "base_stock_alt": {
        "path": "data/inv_management_base_stock.pt",
        "description": "Base-Stock policy (alt name)",
    },
    "minmax": {
        "path": "data/inv_management_minmax.pt",
        "description": "Min-Max (s,S) policy",
    },
    "lotforlot": {
        "path": "data/inv_management_lotforlot.pt",
        "description": "Lot-for-Lot policy",
    },
    "periodic": {
        "path": "data/inv_management_periodic.pt",
        "description": "Periodic Review (T,S) policy",
    },
    "rq": {
        "path": "data/inv_management_rq.pt",
        "description": "(R,Q) Fixed Quantity policy",
    },
    "noisy": {
        "path": "data/inv_management_noisy.pt",
        "description": "Noisy Base-Stock policy",
    },
    "multi_policy": {
        "path": "data/inv_management_multi_policy.pt",
        "description": "Mixed dataset from all policies",
    },
    "continuing": {
        "path": "data/inv_management_continuing.pt",
        "description": "Continuing policy dataset",
    },
}


def update_recursive(d, u):
    """Recursively update dictionary d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def split_by_episodes(states, actions, rewards, next_states, dones, 
                      steps_per_episode=30, val_ratio=0.2, seed=42):
    """Split data at episode level to preserve temporal structure."""
    total_samples = len(states)
    num_episodes = total_samples // steps_per_episode
    truncated = num_episodes * steps_per_episode
    
    # Truncate to exact episode boundaries
    states = states[:truncated]
    actions = actions[:truncated]
    rewards = rewards[:truncated]
    next_states = next_states[:truncated]
    dones = dones[:truncated]
    
    # Shuffle episodes (not samples)
    torch.manual_seed(seed)
    episode_indices = torch.randperm(num_episodes)
    
    num_val_episodes = int(num_episodes * val_ratio)
    val_episode_idx = episode_indices[:num_val_episodes]
    train_episode_idx = episode_indices[num_val_episodes:]
    
    def gather_episodes(episode_idx):
        sample_indices = []
        for ep in episode_idx:
            start = ep * steps_per_episode
            end = start + steps_per_episode
            sample_indices.extend(range(start, end))
        idx = torch.tensor(sample_indices)
        return states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]
    
    train_data = gather_episodes(train_episode_idx)
    val_data = gather_episodes(val_episode_idx)
    
    return train_data, val_data


def find_latest_experiment_path(experiment_name: str, checkpoints_root: Path) -> Path:
    """
    Finds the latest experiment folder matching 'experiment_name_*'.
    """
    if not checkpoints_root.exists():
        return None
        
    candidates = []
    for item in checkpoints_root.iterdir():
        if item.is_dir() and item.name.startswith(experiment_name):
            candidates.append(item)
            
    if not candidates:
        return None
        
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def run_single_experiment(experiment_name: str, dataset_name: str, config_overrides: dict = None):
    """
    Runs a single IQL training experiment on a specific dataset.

    Args:
        experiment_name: Unique identifier for the experiment.
        dataset_name: Key from DATASETS dict.
        config_overrides: Optional config parameters to override.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"DATASET: {dataset_name} - {DATASETS[dataset_name]['description']}")
    logger.info(f"{'='*60}")

    config = load_config()
    if config_overrides:
        config = update_recursive(config, config_overrides)

    # Training parameters
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = float(config['iql']['learning_rate'])
    actor_learning_rate = float(config['iql'].get('actor_learning_rate', learning_rate))
    tau = config['iql']['tau']
    gamma = config['iql']['gamma']
    alpha = config['iql']['alpha']
    beta = config['iql']['beta']
    reward_scale = config['training'].get('reward_scale', 0.1)
    seed = config['training'].get('seed', 42)

    # Load dataset
    dataset_path = PROJECT_ROOT / DATASETS[dataset_name]['path']
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return None

    dataset = torch.load(dataset_path, weights_only=False)
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards'] * reward_scale
    next_states = dataset['next_states']
    dones = dataset['dones']

    logger.info(f"Loaded {len(states)} samples from {dataset_path}")

    # Episode-level split
    train_data, val_data = split_by_episodes(
        states, actions, rewards, next_states, dones,
        steps_per_episode=30, val_ratio=0.2, seed=seed
    )

    train_dataset = TensorDataset(*train_data)
    val_dataset = TensorDataset(*val_data)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    actor_net = Actor(config).to(device)
    q_net = QNet(config).to(device)
    target_q_net = QNet(config).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    v_net = VNet(config).to(device)

    v_optimizer = Adam(v_net.parameters(), lr=learning_rate)
    q_optimizer = Adam(q_net.parameters(), lr=learning_rate)
    actor_optimizer = Adam(actor_net.parameters(), lr=actor_learning_rate)

    agent = IQLAgent(
        device=device, actor=actor_net, q_net=q_net, target_net=target_q_net, v_net=v_net,
        tau=tau, gamma=gamma, alpha=alpha, beta=beta,
        v_optimizer=v_optimizer, q_optimizer=q_optimizer, actor_optimizer=actor_optimizer,
        config=config
    )

    # Setup logging paths
    base_path = PROJECT_ROOT
    base_logging_path = base_path / "logs"
    base_checkpoint_path = base_path / "checkpoints"

    agent._create_new_experimental(
        experimental_name=experiment_name,
        base_logging_path=str(base_logging_path),
        base_checkpoint_path=str(base_checkpoint_path)
    )

    # Train Q/V networks
    logger.info(f"Training Q/V networks for {epochs} epochs...")
    q_v_metrics = agent.train_q_and_v(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_v_path=None,
        resume_q_path=None
    )
    agent.export_diagnostics(q_v_metrics, [], file_path="training_diagnostics_qv.csv")

    # Load best checkpoints
    best_q_path = agent.checkpoint_path / "q_net" / "best_loss.pth"
    best_v_path = agent.checkpoint_path / "v_net" / "best_loss.pth"

    if best_q_path.exists():
        agent._load_model('q_net', str(best_q_path))
        agent.target_q_net.load_state_dict(agent.q_net.state_dict())
    if best_v_path.exists():
        agent._load_model('v_net', str(best_v_path))

    # Train Actor
    logger.info(f"Training Actor for {epochs} epochs...")
    actor_metrics = agent.train_actor(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        resume_training_path=None
    )
    agent.export_diagnostics([], actor_metrics, file_path="training_diagnostics_actor.csv")

    logger.info(f"COMPLETED: {experiment_name}")
    return agent


def main():
    """
    Run Tau Sweep experiments on the basestock dataset.
    """
    tau_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    dataset_key = "basestock"
    
    if dataset_key not in DATASETS:
        logger.error(f"Dataset {dataset_key} not found in DATASETS mapping.")
        return

    logger.info(f"Starting Tau Sweep on {dataset_key} dataset...")
    logger.info(f"Tau values: {tau_values}")

    for tau in tau_values:
        exp_name = f"TAU_SWEEP_{int(tau*100)}"
        overrides = {
            'iql': {'tau': tau}
        }
        
        try:
            run_single_experiment(exp_name, dataset_key, config_overrides=overrides)
        except Exception as e:
            logger.error(f"Tau Sweep for tau={tau} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
