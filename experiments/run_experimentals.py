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

# Available datasets with their properties
DATASETS = {
    "basestock": {
        "path": "data/inv_management_basestock.pt",
        "description": "Base-Stock policy (profit: 346)",
    },
    "minmax": {
        "path": "data/inv_management_minmax.pt",
        "description": "Min-Max (s,S) policy (profit: 350)",
    },
    "lotforlot": {
        "path": "data/inv_management_lotforlot.pt",
        "description": "Lot-for-Lot policy (profit: 351)",
    },
    "periodic": {
        "path": "data/inv_management_periodic.pt",
        "description": "Periodic Review (T,S) policy (profit: 364)",
    },
    "rq": {
        "path": "data/inv_management_rq.pt",
        "description": "(R,Q) Fixed Quantity policy (profit: 431)",
    },
    "noisy": {
        "path": "data/inv_management_noisy.pt",
        "description": "Noisy Base-Stock policy (profit: 364)",
    },
    "multi_policy": {
        "path": "data/inv_management_multi_policy.pt",
        "description": "Mixed dataset from all policies",
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
    
    Returns:
        Path to the folder if found, else None.
    """
    if not checkpoints_root.exists():
        return None
        
    # Pattern: EXP_NAME_DDMMYYYY_HHMMSS
    # We look for folders starting with experiment_name
    candidates = []
    for item in checkpoints_root.iterdir():
        if item.is_dir() and item.name.startswith(experiment_name):
            candidates.append(item)
            
    if not candidates:
        return None
        
    # Sort by modification time (or name which contains timestamp)
    # Timestamp format is DayMonthYear... so string sort might be wrong if crossing months/years without care
    # But usually just taking the one with latest mtime is safest for "latest run"
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
        logger.info(f"Run: python3 scripts/data_generation/generate_{dataset_name}_dataset.py")
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

    # Check for existing checkpoints to resume from
    existing_ckpt_path = find_latest_experiment_path(experiment_name, base_checkpoint_path)
    resume_mode = False
    
    if existing_ckpt_path:
        # Check if Q/V nets are trained
        best_q = existing_ckpt_path / "q_net" / "best_loss.pth"
        best_v = existing_ckpt_path / "v_net" / "best_loss.pth"
        
        if best_q.exists() and best_v.exists():
            logger.info(f"Found existing checkpoint at {existing_ckpt_path}")
            logger.info("Resuming experiment: SKIPPING Q/V training and reusing weights.")
            
            agent.checkpoint_path = existing_ckpt_path
            # Log to the same folder or new? 
            # Reusing same log folder might append weirdly, but keeps things together.
            # Let's use the existing log path matching the checkpoint folder name
            agent.log_path = base_logging_path / existing_ckpt_path.name
            agent.writer = agent.writer = torch.utils.tensorboard.SummaryWriter(log_dir=agent.log_path)
            
            resume_mode = True
        else:
            logger.info(f"Found checkpoint at {existing_ckpt_path} but Q/V weights missing. Starting fresh.")

    if not resume_mode:
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
    else:
        # If resuming, load the weights explicitly
        # agent.checkpoint_path is already set above
        pass

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
    """Run experiments on all available datasets."""
    
    # Define experiments: one per dataset
    experiments = [
        {"name": "EXP_BASESTOCK", "dataset": "basestock"},
        {"name": "EXP_MINMAX", "dataset": "minmax"},
        {"name": "EXP_LOTFORLOT", "dataset": "lotforlot"},
        {"name": "EXP_PERIODIC", "dataset": "periodic"},
        {"name": "EXP_RQ", "dataset": "rq"},
        {"name": "EXP_NOISY", "dataset": "noisy"},
        {"name": "EXP_MULTI_POLICY", "dataset": "multi_policy"},
    ]

    # Check which datasets exist
    available_experiments = []
    for exp in experiments:
        dataset_path = PROJECT_ROOT / DATASETS[exp["dataset"]]["path"]
        if dataset_path.exists():
            available_experiments.append(exp)
        else:
            logger.warning(f"Skipping {exp['name']}: dataset not found at {dataset_path}")

    logger.info(f"Running {len(available_experiments)} experiments")

    for i, exp in enumerate(available_experiments):
        logger.info(f"\n[{i+1}/{len(available_experiments)}] {exp['name']}")
        try:
            run_single_experiment(exp["name"], exp["dataset"])
        except Exception as e:
            logger.error(f"Experiment {exp['name']} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
