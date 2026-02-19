import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import logging
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from src.models.iql.actor import Actor

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class BCAgent:
    """
    Behavior Cloning (BC) agent for imitation learning.
    """

    def __init__(self, device, actor: Actor, actor_optimizer: Optimizer, config):
        self.device = device
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.history = []

        self.log_interval = config['training'].get('log_interval', 100)
        self.checkpoint_interval = config['training'].get('checkpoint_interval', 5)
        self.grad_norm_clip = config['training'].get('grad_norm_clip', 1.0)

        epochs = config['training'].get('epochs', 100)
        eta_min = float(config['iql'].get('eta_min', 1e-6))

        self.scheduler = CosineAnnealingLR(
            self.actor_optimizer, T_max=epochs, eta_min=eta_min)

    def _perform_one_batch_step(self, state_batch, action_batch, batch_step):
        """Calculates and applies the NLL loss for Behavior Cloning."""
        log_probs, entropy, action_mean = self.actor.evaluate(state_batch, action_batch)
        
        # BC Loss is simply the negative log likelihood
        loss = -log_probs.mean()
        entropy_mean = entropy.mean().item()

        if batch_step % self.log_interval == 0:
            self.writer.add_scalar('Loss/bc_loss', loss.item(), batch_step)
            self.writer.add_scalar('Actor/entropy', entropy_mean, batch_step)
            self.writer.add_scalar('Actor/action_mean', action_mean.mean().item(), batch_step)

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm_clip)
        self.actor_optimizer.step()
        
        return loss.item(), entropy_mean

    def _validate(self, val_dataloader):
        """Computes validation loss."""
        self.actor.eval()
        total_loss = 0
        batch_count = 0

        with torch.no_grad():
            for state_batch, action_batch, _, _, _ in val_dataloader:
                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)

                log_probs, _, _ = self.actor.evaluate(state_batch, action_batch)
                loss = -log_probs.mean()
                total_loss += loss.item()
                batch_count += 1

        self.actor.train()
        return total_loss / batch_count

    def train(self, dataloader: DataLoader, val_dataloader: DataLoader, epochs, experimental_name=None):
        """Main training loop for BC."""
        self._create_new_experimental(experimental_name)
        
        best_val_loss = float('inf')
        global_step = 0
        
        for epoch in tqdm(range(epochs), desc="Training BC", unit="epoch"):
            total_loss = 0
            total_entropy = 0
            batch_count = 0

            for state_batch, action_batch, _, _, _ in dataloader:
                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)

                loss, entropy = self._perform_one_batch_step(state_batch, action_batch, global_step)
                
                total_loss += loss
                total_entropy += entropy
                global_step += 1
                batch_count += 1

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('LearningRate/bc', current_lr, epoch)

            val_loss = self._validate(val_dataloader)
            self.writer.add_scalar('Loss/val_bc_loss', val_loss, epoch)

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                tqdm.write(f"Epoch {epoch+1:3d}/{epochs}: Loss={total_loss/batch_count:.4f}, Val-Loss={val_loss:.4f}")

            self.history.append({
                'epoch': epoch,
                'bc_loss': total_loss / batch_count,
                'val_bc_loss': val_loss,
                'entropy': total_entropy / batch_count,
                'lr': current_lr
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, save_best_loss=True, loss=val_loss)
                tqdm.write(f"  -> New best BC model saved! Val-Loss={val_loss:.4f}")

            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch, save_best_loss=False, loss=val_loss)

        self._export_diagnostics()
        return self.history

    def _create_new_experimental(self, experimental_name=None):
        timestamp = datetime.datetime.now().strftime(format='%d%m%Y_%H%M%S')
        if not experimental_name:
            experimental_name = f"bc_{timestamp}"
        else:
            experimental_name = f"bc_{experimental_name}_{timestamp}"

        self.log_path = PROJECT_ROOT / "logs" / experimental_name
        self.checkpoint_path = PROJECT_ROOT / "checkpoints" / experimental_name

        self.log_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.logger.info(f"New BC experiment: {experimental_name}")

    def _save_checkpoint(self, epoch: int, save_best_loss: bool, loss: float):
        checkpoint_name = "best_loss.pth" if save_best_loss else f"checkpoint_epoch_{epoch}.pth"
        checkpoint_dir = self.checkpoint_path / "actor"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'config': self.config,
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, checkpoint_dir / checkpoint_name)

    def _export_diagnostics(self):
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_path / "training_diagnostics_bc.csv", index=False)
        self.logger.info(f"Diagnostics exported to {self.log_path}")
