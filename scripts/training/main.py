"""
Full Training Pipeline - Orchestrates Q/V and Actor Training

This script runs the complete IQL training pipeline by calling the modular
training scripts instead of duplicating their logic.
"""
from scripts.training import train_actor
from scripts.training import train_q_v_net
from utils.logger_config import setup_logging
import logging

# Import training modules

setup_logging()
logger = logging.getLogger(__name__)


def main():
    """
    Run the full training pipeline:
    1. Train Q and V networks
    2. Train Actor network using the best Q/V checkpoints
    """
    logger.info("="*60)
    logger.info("FULL IQL TRAINING PIPELINE")
    logger.info("="*60)

    # Phase 1: Train Q and V networks
    logger.info("\nPhase 1: Training Q-Network and V-Network")
    experiment_id = train_q_v_net.train_qv_networks()

    if not experiment_id:
        logger.error("Q/V training failed. Aborting pipeline.")
        return

    logger.info(f"\n✓ Q/V training completed. Experiment ID: {experiment_id}")

    # Phase 2: Train Actor network
    logger.info(f"\nPhase 2: Training Actor Network")
    logger.info(f"Using experiment: {experiment_id}")

    success = train_actor.train_actor_network(experiment_id)

    if success:
        logger.info("\n" + "="*60)
        logger.info("FULL TRAINING PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Experiment ID: {experiment_id}")
        logger.info(f"Checkpoints: checkpoints/{experiment_id}/")
        logger.info(f"Logs: logs/{experiment_id}/")
        logger.info("="*60)
    else:
        logger.error("Actor training failed.")


if __name__ == "__main__":
    main()
