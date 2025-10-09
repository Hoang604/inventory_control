import torch
import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import logging
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from .critics import VNet, QNet
from .actor import Actor


class IQLAgent:
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

    def _perform_one_batch_step_for_v_net(self, state_batch: torch.Tensor, target_batch: torch.Tensor, batch_step):
        v_output: torch.Tensor = self.v_net(state_batch)
        self.writer.add_scalar('Value/avg_v_value',
                               v_output.mean().item(), batch_step)
        error = target_batch - v_output
        loss = torch.where(error > 0, self.tau, 1 - self.tau) * error ** 2
        mean_loss = loss.mean()
        self.writer.add_scalar('Loss/v_loss', mean_loss.item(), batch_step)
        self.v_optimizer.zero_grad()
        mean_loss.backward()
        self.v_optimizer.step()
        return mean_loss

    def _perform_one_batch_step_for_q_net(self, reward_batch: torch.Tensor, state_batch: torch.Tensor, action_batch: torch.Tensor, estimated_v_for_next_state: torch.Tensor, batch_step):
        q_input = torch.cat((state_batch, action_batch), dim=1)
        q_output: torch.Tensor = self.q_net(q_input)
        self.writer.add_scalar('Value/avg_q_value',
                               q_output.mean().item(), batch_step)
        target_q = reward_batch + self.gamma * estimated_v_for_next_state
        loss: torch.Tensor = (target_q - q_output)**2
        mean_loss = loss.mean()
        self.writer.add_scalar('Loss/q_loss', mean_loss.item(), batch_step)
        self.q_optimizer.zero_grad()
        mean_loss.backward()
        self.q_optimizer.step()
        return mean_loss

    def _soft_update_target_net(self):
        with torch.no_grad():
            for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(
                    target_param * (1 - self.alpha) + q_param * self.alpha)
                # copy_ copy the thing in (...) to taget_param.data, not make a copy of target_param.data

    def _perform_one_batch_step_for_actor_net(self, state_batch, action_batch, batch_step):
        with torch.no_grad():
            target_net_input = torch.cat((state_batch, action_batch), dim=1)
            target_net_ouput: torch.Tensor = self.target_net(target_net_input)
            v_net_output: torch.Tensor = self.v_net(state_batch)
            advantage = target_net_ouput - v_net_output
            self.writer.add_scalar(
                'Value/advantage', advantage.mean().item(), batch_step)
            weight = torch.exp(self.beta*advantage)

        log_probs, entropy, action_mean = self.actor.evaluate(
            state_batch, action_batch)

        self.writer.add_scalar(
            'Actor/entropy', entropy.mean().item(), batch_step)
        self.writer.add_scalar('Actor/action_mean',
                               action_mean.mean().item(), batch_step)

        # We want to MAXIMIZE the weighted log_probs, so we MINIMIZE its negative
        loss: torch.Tensor = - (weight * log_probs)
        mean_loss = loss.mean()
        self.writer.add_scalar('Loss/actor_loss', mean_loss.item(), batch_step)

        self.actor_optimizer.zero_grad()
        mean_loss.backward()
        self.actor_optimizer.step()
        return mean_loss

    def train_q_and_v(self, dataloader: DataLoader, epochs, resume_v_path, resume_q_path):
        self.logger.info("Training Q and V functions...")
        if resume_v_path:
            v_best_loss, v_continue_epoch = self._load_model(
                model_name='v_net', file_path=resume_v_path)
        else:
            v_best_loss = torch.inf
            v_continue_epoch = 0
        if resume_q_path:
            q_best_loss, q_continue_epoch = self._load_model(
                model_name='q_net', file_path=resume_q_path)
        else:
            q_best_loss = torch.inf
            q_continue_epoch = 0

        start_epoch = min(v_continue_epoch, q_continue_epoch)
        global_step = start_epoch * len(dataloader)
        for epoch in tqdm(range(start_epoch, epochs)):
            v_mean_loss = 0
            q_mean_loss = 0
            for state_batch, action_batch, reward_batch, next_state_batch in enumerate(dataloader):
                with torch.no_grad():
                    target_batch = self.target_net(state_batch, action_batch)
                v_loss = self._perform_one_batch_step_for_v_net(
                    state_batch, target_batch, global_step)
                v_mean_loss = v_mean_loss * global_step / \
                    (global_step + 1) + v_loss / (global_step + 1)
                with torch.no_grad():
                    estimated_v_for_next_state: torch.Tensor = self.v_net(
                        next_state_batch)
                q_loss = self._perform_one_batch_step_for_q_net(
                    reward_batch, state_batch, action_batch, estimated_v_for_next_state, global_step)
                q_mean_loss = q_mean_loss * global_step / \
                    (global_step + 1) + q_loss / (global_step + 1)
                self._soft_update_target_net()
                global_step += 1
            if v_mean_loss < v_best_loss:
                self.logger.info(
                    "New best loss for v_net, update new best loss and saving best model...")
                v_best_loss = v_mean_loss
                self._save_checkpoint(
                    epoch, save_best_loss=True, model_name='v_net', loss=v_best_loss)
            if q_mean_loss < q_best_loss:
                self.logger.info(
                    "New best loss for q_net, update new best loss and saving best model...")
                q_best_loss = q_mean_loss
                self._save_checkpoint(
                    epoch, save_best_loss=True, model_name='q_net', loss=q_best_loss)

            if (epoch + 1) % 5 == 0:
                self.logger.info(
                    f"Saving periodic checkpoint for epoch {epoch + 1}")
                self._save_checkpoint(
                    epoch, save_best_loss=False, model_name='v_net', loss=v_mean_loss)
                self._save_checkpoint(
                    epoch, save_best_loss=False, model_name='q_net', loss=q_mean_loss)

    def train_actor(self, dataloader: DataLoader, epochs, resume_training_path=None):
        self.logger.info("Extracting Policy (training actor)...")
        if resume_training_path:
            start_epoch, actor_best_loss = self._load_model(
                model_name='actor', file_path=resume_training_path)
        else:
            actor_best_loss = torch.inf
            start_epoch = 0

        global_step = start_epoch * len(dataloader)
        for epoch in tqdm(range(start_epoch, epochs)):
            actor_mean_loss = 0
            for state_batch, action_batch, _, _ in enumerate(dataloader):
                actor_loss = self._perform_one_batch_step_for_actor_net(
                    state_batch, action_batch, global_step)
                actor_mean_loss = actor_mean_loss * \
                    global_step / (global_step + 1) + \
                    actor_loss / (global_step + 1)
                global_step += 1
            if actor_mean_loss < actor_best_loss:
                self.logger.info(
                    "New best loss for actor_net, update new best loss and saving best model...")
                actor_best_loss = actor_mean_loss
                self._save_checkpoint(
                    epoch, save_best_loss=True, model_name='actor_net', loss=actor_best_loss)

            if (epoch + 1) % 5 == 0:
                self.logger.info(
                    f"Saving periodic checkpoint for epoch {epoch + 1}")
                self._save_checkpoint(
                    epoch, save_best_loss=False, model_name='actor', loss=actor_mean_loss)

    def train(self, dataloader: DataLoader, epochs, resume_q_path, resume_v_path, resume_actor_path, experimental_name=None, base_logging_path="/home/hoang/python/inventory_control/logs", base_checkpoint_path="/home/hoang/python/inventory_control/checkpoints"):
        self._create_new_experimental(
            experimental_name, base_logging_path, base_checkpoint_path)
        self.train_q_and_v(
            dataloader, epochs, resume_v_path=resume_v_path, resume_q_path=resume_q_path)

        self.train_actor(dataloader, epochs,
                         resume_training_path=resume_actor_path)

    def _create_new_experimental(self, experimental_name=None, base_logging_path=None, base_checkpoint_path=None):
        if not experimental_name:
            experimental_name = f"{datetime.datetime.now().strftime(format="%d%m%Y_%H%M%S")}_experimental"

        self.log_path = Path(base_logging_path) / experimental_name
        self.checkpoint_path = Path(base_checkpoint_path) / experimental_name

        self.log_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.logger.info(
            f"Create new experimental: {experimental_name}; logging at: {self.log_path}; checkpoint saved at: {self.checkpoint_path}")

    def _save_checkpoint(self, epoch: int, save_best_loss: bool, model_name: str, loss: float):
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth" if not save_best_loss else "best_loss.pth"
        if model_name == 'actor':
            model_to_save = self.actor
            optimizer_to_save = self.actor_optimizer
            checkpoint_file_path = self.checkpoint_path / "actor" / checkpoint_name
        elif model_name == 'q_net':
            model_to_save = self.q_net
            optimizer_to_save = self.q_optimizer
            checkpoint_file_path = self.checkpoint_path / "q_net" / checkpoint_name
        elif model_name == 'v_net':
            model_to_save = self.v_net
            optimizer_to_save = self.v_optimizer
            checkpoint_file_path = self.checkpoint_path / "v_net" / checkpoint_name
        else:
            self.logger.error(
                f"The model name can only take 'actor', 'q_net', 'v_net' but got {model_name}")
            return

        checkpoint = {
            'epoch': epoch,
            'config': self.config,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer_to_save.state_dict(),
            'loss': loss
        }

        torch.save(checkpoint, checkpoint_file_path)
        self.logger.info(f"Saved {model_name} model to {checkpoint_file_path}")

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

        # Select the correct model and optimizer to load into
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
