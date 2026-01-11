import torch
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import logging
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from .critics import VNet, QNet
from .actor import Actor


class IQLAgent:
    """
    Implicit Q-Learning (IQL) agent.
    """

    def __init__(self, device, actor: Actor, q_net: QNet, target_net: QNet, v_net: VNet, tau: float, gamma: float, alpha: float, beta: float, v_optimizer: Optimizer, q_optimizer: Optimizer, actor_optimizer: Optimizer, config):
        self.device = device
        self.actor = actor
        self.q_net = q_net
        self.target_net = target_net
        self.v_net = v_net
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.q_v_history = []
        self.actor_history = []

        self.log_interval = config['training'].get('log_interval', 100)
        self.checkpoint_interval = config['training'].get(
            'checkpoint_interval', 5)
        self.grad_norm_clip = config['training'].get('grad_norm_clip', 1.0)
        self.adv_weight_clip = config['training'].get('adv_weight_clip', 100.0)

        epochs = config['training'].get('epochs', 100)
        eta_min = float(config['iql'].get('eta_min', 1e-6))

        self.v_scheduler = CosineAnnealingLR(
            self.v_optimizer, T_max=epochs, eta_min=eta_min)
        self.q_scheduler = CosineAnnealingLR(
            self.q_optimizer, T_max=epochs, eta_min=eta_min)
        self.actor_scheduler = CosineAnnealingLR(
            self.actor_optimizer, T_max=epochs, eta_min=eta_min)

    def _perform_one_batch_step_for_v_net(self, state_batch: torch.Tensor, target_batch: torch.Tensor, batch_step):
        v_output: torch.Tensor = self.v_net(state_batch)
        v_mean = v_output.mean().item()

        if batch_step % self.log_interval == 0:
            self.writer.add_scalar('Value/avg_v_value', v_mean, batch_step)

        error = target_batch - v_output
        loss = torch.where(error > 0, self.tau, 1 - self.tau) * error ** 2
        mean_loss = loss.mean()

        if batch_step % self.log_interval == 0:
            self.writer.add_scalar('Loss/v_loss', mean_loss.item(), batch_step)

        self.v_optimizer.zero_grad()
        mean_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.v_net.parameters(), max_norm=self.grad_norm_clip)
        self.v_optimizer.step()
        return mean_loss, v_mean

    def _perform_one_batch_step_for_q_net(self, reward_batch: torch.Tensor, state_batch: torch.Tensor, action_batch: torch.Tensor, estimated_v_for_next_state: torch.Tensor, batch_step):
        q_output: torch.Tensor = self.q_net(state_batch, action_batch)
        q_mean = q_output.mean().item()

        if batch_step % self.log_interval == 0:
            q_min = q_output.min().item()
            q_max = q_output.max().item()
            # Only print every 500 batches to reduce clutter
            if batch_step % 500 == 0:
                tqdm.write(f"  Batch {batch_step}: Q-Values - Mean: {q_mean:.4f}, Min: {q_min:.4f}, Max: {q_max:.4f}")
            self.writer.add_scalar('Value/avg_q_value', q_mean, batch_step)

        target_q = reward_batch + self.gamma * estimated_v_for_next_state
        loss: torch.Tensor = (target_q - q_output)**2
        mean_loss = loss.mean()

        if batch_step % self.log_interval == 0:
            self.writer.add_scalar('Loss/q_loss', mean_loss.item(), batch_step)

        self.q_optimizer.zero_grad()
        mean_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), max_norm=self.grad_norm_clip)
        self.q_optimizer.step()
        return mean_loss, q_mean

    def _soft_update_target_net(self):
        with torch.no_grad():
            for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(
                    target_param * (1 - self.alpha) + q_param * self.alpha)

    def _perform_one_batch_step_for_actor_net(self, state_batch, action_batch, batch_step):
        with torch.no_grad():
            target_net_ouput: torch.Tensor = self.target_net(
                state_batch, action_batch)
            v_net_output: torch.Tensor = self.v_net(state_batch)
            advantage = target_net_ouput - v_net_output
            adv_mean = advantage.mean().item()

            if batch_step % self.log_interval == 0:
                self.writer.add_scalar(
                    'Value/advantage', adv_mean, batch_step)

            weight = torch.exp(self.beta * advantage)
            weight = torch.clamp(weight, max=self.adv_weight_clip)

        log_probs, entropy, action_mean = self.actor.evaluate(
            state_batch, action_batch)

        entropy_mean = entropy.mean().item()

        if batch_step % self.log_interval == 0:
            self.writer.add_scalar(
                'Actor/entropy', entropy_mean, batch_step)
            self.writer.add_scalar('Actor/action_mean',
                                   action_mean.mean().item(), batch_step)

        loss: torch.Tensor = - (weight * log_probs)
        mean_loss = loss.mean()

        if batch_step % self.log_interval == 0:
            self.writer.add_scalar(
                'Loss/actor_loss', mean_loss.item(), batch_step)

        self.actor_optimizer.zero_grad()
        mean_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=self.grad_norm_clip)
        self.actor_optimizer.step()
        return mean_loss, entropy_mean, adv_mean

    def _validate_q_v(self, val_dataloader):
        self.v_net.eval()
        self.q_net.eval()
        self.target_net.eval()

        total_v_loss = 0
        total_q_loss = 0

        all_q_outputs = []
        batch_count = 0

        with torch.no_grad():
            for state_batch, action_batch, reward_batch, next_state_batch in val_dataloader:
                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                reward_batch = reward_batch.to(self.device)
                next_state_batch = next_state_batch.to(self.device)

                target_q = self.target_net(state_batch, action_batch)
                v_output = self.v_net(state_batch)
                v_error = target_q - v_output
                v_loss = torch.where(v_error > 0, self.tau,
                                     1 - self.tau) * v_error ** 2
                total_v_loss += v_loss.mean().item()

                estimated_v_next = self.v_net(next_state_batch)
                target_q_val = reward_batch + self.gamma * estimated_v_next
                q_output = self.q_net(state_batch, action_batch)
                q_loss = ((target_q_val - q_output)**2).mean()
                total_q_loss += q_loss.item()

                all_q_outputs.append(q_output.cpu())

                batch_count += 1

        self.v_net.train()
        self.q_net.train()
        self.target_net.train()

        all_q_outputs_tensor = torch.cat(all_q_outputs)
        val_q_mean = all_q_outputs_tensor.mean().item()
        val_q_std = all_q_outputs_tensor.std().item()

        return total_v_loss / batch_count, total_q_loss / batch_count, val_q_mean, val_q_std

    def _validate_actor(self, val_dataloader):
        self.actor.eval()
        self.v_net.eval()
        self.target_net.eval()
        self.q_net.eval()

        total_loss = 0
        total_q_val = 0
        batch_count = 0

        with torch.no_grad():
            for state_batch, action_batch, _, _ in val_dataloader:
                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)

                target_q = self.target_net(state_batch, action_batch)
                v_val = self.v_net(state_batch)
                advantage = target_q - v_val
                weight = torch.exp(self.beta * advantage)
                weight = torch.clamp(weight, max=self.adv_weight_clip)

                log_probs, _, _ = self.actor.evaluate(
                    state_batch, action_batch)
                loss = - (weight * log_probs).mean()
                total_loss += loss.item()

                _, _, pred_action = self.actor.evaluate(
                    state_batch, action_batch)
                pred_q_val = self.q_net(state_batch, pred_action)
                total_q_val += pred_q_val.mean().item()

                batch_count += 1

        self.actor.train()
        self.v_net.train()
        self.target_net.train()
        self.q_net.train()

        return total_loss / batch_count, total_q_val / batch_count

    def train_q_and_v(self, dataloader: DataLoader, val_dataloader: DataLoader, epochs, resume_v_path, resume_q_path):
        self.logger.info("Training Q and V functions...")
        if resume_v_path:
            _, _ = self._load_model(
                model_name='v_net', file_path=resume_v_path)
        if resume_q_path:
            _, _ = self._load_model(
                model_name='q_net', file_path=resume_q_path)

        q_best_loss = torch.inf

        start_epoch = 0

        global_step = 0
        logged_device = False
        epoch_metrics = []

        # Clean progress bar with description
        for epoch in tqdm(range(start_epoch, epochs), desc="Training Q/V", unit="epoch"):
            total_v_loss = 0
            total_q_loss = 0
            total_q_val = 0
            total_v_val = 0
            batch_count = 0

            for state_batch, action_batch, reward_batch, next_state_batch in dataloader:
                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                reward_batch = reward_batch.to(self.device)
                next_state_batch = next_state_batch.to(self.device)

                if not logged_device:
                    tqdm.write(f"Training batch tensors on device: {state_batch.device}")
                    logged_device = True

                with torch.no_grad():
                    target_batch = self.target_net(state_batch, action_batch)

                v_loss, v_val = self._perform_one_batch_step_for_v_net(
                    state_batch, target_batch, global_step)
                total_v_loss += v_loss.item()
                total_v_val += v_val

                with torch.no_grad():
                    estimated_v_for_next_state: torch.Tensor = self.v_net(
                        next_state_batch)

                q_loss, q_val = self._perform_one_batch_step_for_q_net(
                    reward_batch, state_batch, action_batch, estimated_v_for_next_state, global_step)
                total_q_loss += q_loss.item()
                total_q_val += q_val

                self._soft_update_target_net()
                global_step += 1
                batch_count += 1

            self.q_scheduler.step()
            self.v_scheduler.step()

            current_lr = self.q_scheduler.get_last_lr()[0]
            self.writer.add_scalar('LearningRate/q_v_net', current_lr, epoch)

            v_mean_loss = total_v_loss / batch_count
            q_mean_loss = total_q_loss / batch_count
            avg_q_val = total_q_val / batch_count
            avg_v_val = total_v_val / batch_count

            val_v_loss, val_q_loss, val_q_mean, val_q_std = self._validate_q_v(
                val_dataloader)
            self.writer.add_scalar('Loss/val_v_loss', val_v_loss, global_step)
            self.writer.add_scalar('Loss/val_q_loss', val_q_loss, global_step)

            stability_score = val_q_mean - 1.0 * val_q_std
            self.writer.add_scalar('Value/val_q_mean', val_q_mean, global_step)
            self.writer.add_scalar('Value/val_q_std', val_q_std, global_step)
            self.writer.add_scalar(
                'Value/stability_score', stability_score, global_step)

            # Clean epoch summary every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                tqdm.write(
                    f"Epoch {epoch+1:3d}/{epochs}: "
                    f"Q-Loss={q_mean_loss:.4f}, "
                    f"V-Loss={v_mean_loss:.4f}, "
                    f"Val-Q-Loss={val_q_loss:.4f}, "
                    f"Stability={stability_score:.2f}"
                )

            epoch_metrics.append({
                'epoch': epoch,
                'q_loss': q_mean_loss,
                'v_loss': v_mean_loss,
                'val_q_loss': val_q_loss,
                'val_v_loss': val_v_loss,
                'avg_q_val': avg_q_val,
                'avg_v_val': avg_v_val,
                'val_q_mean': val_q_mean,
                'val_q_std': val_q_std,
                'stability_score': stability_score,
                'lr': current_lr
            })

            if val_q_loss < q_best_loss:
                q_best_loss = val_q_loss
                self._save_checkpoint(
                    epoch, save_best_loss=True, model_name='q_net', loss=q_best_loss)
                self._save_checkpoint(
                    epoch, save_best_loss=True, model_name='v_net', loss=val_v_loss)

            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(
                    epoch, save_best_loss=False, model_name='v_net', loss=val_v_loss)
                self._save_checkpoint(
                    epoch, save_best_loss=False, model_name='q_net', loss=val_q_loss)

        return epoch_metrics

    def train_actor(self, dataloader: DataLoader, val_dataloader: DataLoader, epochs, resume_training_path=None):
        self.logger.info("Extracting Policy (training actor)...")
        if resume_training_path:
            _, _ = self._load_model(
                model_name='actor', file_path=resume_training_path)

        best_actor_q_mean = -torch.inf
        start_epoch = 0

        global_step = 0
        epoch_metrics = []

        # Clean progress bar for actor training
        for epoch in tqdm(range(start_epoch, epochs), desc="Training Actor", unit="epoch"):
            total_actor_loss = 0
            total_entropy = 0
            total_advantage = 0
            batch_count = 0

            for state_batch, action_batch, _, _ in dataloader:
                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)

                actor_loss, entropy, adv = self._perform_one_batch_step_for_actor_net(
                    state_batch, action_batch, global_step)

                total_actor_loss += actor_loss.item()
                total_entropy += entropy
                total_advantage += adv
                global_step += 1
                batch_count += 1

            self.actor_scheduler.step()
            current_lr = self.actor_scheduler.get_last_lr()[0]
            self.writer.add_scalar('LearningRate/actor', current_lr, epoch)

            actor_mean_loss = total_actor_loss / batch_count
            avg_entropy = total_entropy / batch_count
            avg_advantage = total_advantage / batch_count

            val_actor_loss, val_actor_q_mean = self._validate_actor(
                val_dataloader)
            self.writer.add_scalar('Loss/val_actor_loss',
                                   val_actor_loss, global_step)
            self.writer.add_scalar(
                'Value/val_actor_q_mean', val_actor_q_mean, global_step)

            # Clean epoch summary every 10 epochs  
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                tqdm.write(
                    f"Epoch {epoch+1:3d}/{epochs}: "
                    f"Actor-Loss={actor_mean_loss:.4f}, "
                    f"Val-Loss={val_actor_loss:.4f}, "
                    f"Val-Q-Mean={val_actor_q_mean:.4f}"
                )

            epoch_metrics.append({
                'epoch': epoch,
                'actor_loss': actor_mean_loss,
                'val_actor_loss': val_actor_loss,
                'val_actor_q_mean': val_actor_q_mean,
                'avg_entropy': avg_entropy,
                'avg_advantage': avg_advantage,
                'lr': current_lr
            })

            if val_actor_q_mean > best_actor_q_mean:
                best_actor_q_mean = val_actor_q_mean
                self._save_checkpoint(
                    epoch, save_best_loss=True, model_name='actor', loss=best_actor_q_mean)

            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(
                    epoch, save_best_loss=False, model_name='actor', loss=val_actor_loss)

        return epoch_metrics

    def export_diagnostics(self, q_v_metrics, actor_metrics, file_path="training_diagnostics.csv"):
        q_v_df = pd.DataFrame(q_v_metrics)
        actor_df = pd.DataFrame(actor_metrics)

        if not q_v_df.empty and not actor_df.empty:
            full_df = pd.merge(q_v_df, actor_df, on='epoch', how='outer')
        elif not q_v_df.empty:
            full_df = q_v_df
        elif not actor_df.empty:
            full_df = actor_df
        else:
            return

        full_df.sort_values(by='epoch', inplace=True)
        full_df.to_csv(self.log_path / file_path, index=False)
        self.logger.info(
            f"Detailed diagnostics exported to {self.log_path / file_path}")

    def train(self, dataloader: DataLoader, val_dataloader: DataLoader, epochs, resume_q_path, resume_v_path, resume_actor_path, experimental_name=None, base_logging_path="/home/hoang/python/inventory_control/logs", base_checkpoint_path="/home/hoang/python/inventory_control/checkpoints"):
        self._create_new_experimental(
            experimental_name, base_logging_path, base_checkpoint_path)

        q_v_metrics = self.train_q_and_v(
            dataloader, val_dataloader, epochs, resume_v_path=resume_v_path, resume_q_path=resume_q_path)

        self.logger.info(
            "Reloading 'best_loss.pth' for Q and V networks before training Actor...")

        best_q_path = self.checkpoint_path / "q_net" / "best_loss.pth"
        best_v_path = self.checkpoint_path / "v_net" / "best_loss.pth"

        if best_q_path.exists():
            self._load_model('q_net', str(best_q_path))
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.logger.info("Target Network synced with Best Q-Network.")
        else:
            self.logger.warning(
                f"Best Q-Net checkpoint not found at {best_q_path}. Using final epoch weights.")

        if best_v_path.exists():
            self._load_model('v_net', str(best_v_path))
        else:
            self.logger.warning(
                f"Best V-Net checkpoint not found at {best_v_path}. Using final epoch weights.")

        actor_metrics = self.train_actor(dataloader, val_dataloader, epochs,
                                         resume_training_path=resume_actor_path)

        self.export_diagnostics(q_v_metrics, actor_metrics)

    def _create_new_experimental(self, experimental_name=None, base_logging_path=None, base_checkpoint_path=None):
        """
        Creates a new experiment by setting up logging and checkpoint directories.

        Args:
            experimental_name: The name of the experiment. If not provided, a name will be generated based on the current timestamp.
            base_logging_path: The base path for logging.
            base_checkpoint_path: The base path for saving checkpoints.
        """
        timestamp = datetime.datetime.now().strftime(format='%d%m%Y_%H%M%S')
        if not experimental_name:
            experimental_name = f"{timestamp}_experimental"
        else:
            experimental_name = f"{experimental_name}_{timestamp}"

        self.log_path = Path(base_logging_path) / experimental_name
        self.checkpoint_path = Path(base_checkpoint_path) / experimental_name

        self.log_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.logger.info(
            f"Create new experimental: {experimental_name}; logging at: {self.log_path}; checkpoint saved at: {self.checkpoint_path}")

    def _save_checkpoint(self, epoch: int, save_best_loss: bool, model_name: str, loss: float):
        """
        Saves a checkpoint of a model.

        Args:
            epoch: The current epoch number.
            save_best_loss: Whether to save the checkpoint as the best loss checkpoint.
            model_name: The name of the model to save ('actor', 'q_net', or 'v_net').
            loss: The loss of the model.
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth" if not save_best_loss else "best_loss.pth"
        if model_name == 'actor':
            model_to_save = self.actor
            optimizer_to_save = self.actor_optimizer
            checkpoint_dir = self.checkpoint_path / "actor"
        elif model_name == 'q_net':
            model_to_save = self.q_net
            optimizer_to_save = self.q_optimizer
            checkpoint_dir = self.checkpoint_path / "q_net"
        elif model_name == 'v_net':
            model_to_save = self.v_net
            optimizer_to_save = self.v_optimizer
            checkpoint_dir = self.checkpoint_path / "v_net"
        else:
            self.logger.error(
                f"The model name can only take 'actor', 'q_net', 'v_net' but got {model_name}")
            return

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file_path = checkpoint_dir / checkpoint_name

        checkpoint = {
            'epoch': epoch,
            'config': self.config,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer_to_save.state_dict(),
            'loss': loss
        }

        torch.save(checkpoint, checkpoint_file_path)

    def _load_model(self, model_name: str, file_path: str):
        """
        Loads a single model and its optimizer state from a file.

        Args:
            model_name (str): The name of the model to load ('actor', 'q_net', 'v_net').
            file_path (str): The full path to the checkpoint file.

        Returns:
            int: The epoch number from the checkpoint.
        """
        if not Path(file_path).is_file():
            self.logger.error(f"Checkpoint file not found: {file_path}")
            return 0

        if model_name == 'actor':
            model_to_load = self.actor
            optimizer_to_load = self.actor_optimizer
        elif model_name == 'q_net':
            model_to_load = self.q_net
            optimizer_to_load = self.q_optimizer
        elif model_name == 'v_net':
            model_to_load = self.v_net
            optimizer_to_load = self.v_optimizer
        else:
            self.logger.error(
                f"Unknown model name: {model_name}. Cannot load.")
            return 0

        checkpoint = torch.load(file_path, map_location=self.device)
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer_to_load.load_state_dict(checkpoint['optimizer_state_dict'])

        self.logger.info(f"Loaded {model_name} from {file_path}")
        return checkpoint["epoch"], checkpoint["loss"]
