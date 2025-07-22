import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import logging
import argparse

from controller_dataset import ControllerDataModule, normalize_actions
from lstm_step_controller import TactileLSTMController
from lstm_step_test import test_lstm_controller


class LSTMControllerTrainer:
    def __init__(self, controller, data_module, learning_rate=1e-4, weight_decay=1e-6, checkpoint_dir='checkpoint/lstm_controller', device='cuda'):
        self.controller = controller
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = optim.AdamW(
            [p for m in controller.trainable_modules for p in m.parameters()],
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000, eta_min=learning_rate / 10
        )

        self.writer = SummaryWriter(log_dir=self.checkpoint_dir)
        self.logger = self._setup_logger()

        stats_tensors = {key: torch.tensor(value, dtype=torch.float32) for key, value in data_module.stats.items()}
        stats_gpu = {key: value.cuda() for key, value in stats_tensors.items()}
        self.controller.stats = stats_gpu

        self.data_module = data_module


    def _setup_logger(self):
        logger = logging.getLogger('lstm_controller_training')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.checkpoint_dir, 'training.log'))
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _prepare_batch(self, batch):
        context_frames = 2  # or from args/model config

        current_state = batch['states'][:, context_frames - 1] # use the most recent step
        forces = batch['forces'][:, context_frames-1:-1] if self.controller.use_force else None

        vla_actions_n = normalize_actions(batch['vla_actions'], self.controller.stats, 'vla')
        expert_actions_n = normalize_actions(batch['expert_actions'], self.controller.stats, 'expert')

        cam1 = batch['images_cam1'][:, -1] if 'images_cam1' in batch else None
        cam2 = batch['images_cam2'][:, -1] if 'images_cam2' in batch else None

        # Encode the observation (state + images)
        obs_cond = self.controller.encode_observation(
            current_state, cam1, cam2
        )

        # Create batch_dict for diffusion model
        batch_dict = {
            'obs_cond': obs_cond,
            'expert_act': expert_actions_n,  # Target expert action
            'vla_act': vla_actions_n,  # Source VLA action
            'forces': forces,
        }

        return batch_dict

    def train(self, data_module, num_epochs=500, save_interval=50, log_interval=100):
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.controller.train()
            train_loss, global_step = self._train_epoch(train_loader, epoch, global_step, log_interval)

            self.logger.info(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f}")
            self.writer.add_scalar('Epoch/train_loss', train_loss, epoch + 1)

            if (epoch+1)%5==0:
                # Evaluate
                val_loss = self._evaluate(val_loader, epoch+1)
                self.writer.add_scalar('Epoch/val_loss', val_loss, epoch + 1)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best_model")

                # Periodic checkpoint
                if (epoch + 1) % save_interval == 0:
                    self._save_checkpoint(f"epoch_{epoch + 1}")

                # # Run test every N epochs
                # if (epoch + 1) % 50 == 0 or (epoch + 1) == 5:
                #     self._run_test_during_training(val_loader, epoch + 1)

        self.writer.close()

    def _train_epoch(self, train_loader, epoch, global_step, log_interval):
        running_loss = 0.0
        num_batches = len(train_loader)

        for i, batch in enumerate(tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}")):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            batch_dict = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            loss = self.controller.get_loss(batch_dict)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            if global_step % log_interval == 0:
                self.writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

        avg_loss = running_loss / num_batches
        return avg_loss, global_step

    def _evaluate(self, val_loader, epoch):
        self.controller.eval()
        running_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val] Epoch {epoch + 1}"):
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)

                batch_dict = self._prepare_batch(batch)
                loss = self.controller.get_loss(batch_dict)
                running_loss += loss.item()

        avg_loss = running_loss / num_batches
        self.logger.info(f"Epoch {epoch + 1} | Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _run_test_during_training(self, dataloader, epoch):
        """
        Run test on current controller during training.
        """
        import shutil

        temp_checkpoint_dir = os.path.join(self.checkpoint_dir, f"temp_epoch_{epoch}")
        os.makedirs(temp_checkpoint_dir, exist_ok=True)
        self._save_checkpoint(f"temp_epoch_{epoch}")

        self.logger.info(f"[TEST] Running test during training at epoch {epoch}")

        test_results = test_lstm_controller(
            checkpoint_dir=temp_checkpoint_dir,
            data_loader=dataloader,
            context_frames=self.data_module.context_frames,
            horizon=self.data_module.horizon,
            device=self.device,
            num_samples=50,
            state_dim=self.controller.state_dim,
            hidden_dim=self.controller.hidden_dim,
            force_dim=self.controller.force_dim,
            use_force=self.controller.use_force,
            visualize=True,
            save_dir=os.path.join(self.checkpoint_dir, f"test_visualizations_epoch_{epoch}")
        )

        self.writer.add_scalar('Test/avg_error', test_results['avg_error'], epoch)
        self.writer.add_scalar('Test/avg_vla_error', test_results['avg_vla_error'], epoch)
        self.writer.add_scalar('Test/improvement_pct', test_results['improvement'], epoch)

        self.logger.info(
            f"[TEST] Epoch {epoch} | Avg Error: {test_results['avg_error']:.4f} | "
            f"Avg VLA Error: {test_results['avg_vla_error']:.4f} | "
            f"Improvement: {test_results['improvement']:.2f}%"
        )

        shutil.rmtree(temp_checkpoint_dir)

    def _save_checkpoint(self, name):
        path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(path, exist_ok=True)
        self.controller.save(path)

    def load_checkpoint(self, path):
        self.controller.load(path)



def train_lstm_controller_with_dataset(controller, args):
    data_module = ControllerDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        context_frames=args.context_frames,
        horizon=args.horizon,
        image_size=args.image_size,
        val_ratio=0.2,
        stride=args.stride
    )

    trainer = LSTMControllerTrainer(
        controller=controller,
        data_module=data_module,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )

    trainer.train(
        data_module=data_module,
        num_epochs=args.num_epochs,
        save_interval=args.save_interval,
        log_interval=args.log_interval
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM-based controller")

    # Dataset arguments

    parser.add_argument("--context_frames", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=1)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=100)

    # Model arguments
    parser.add_argument("--state_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--image_model", type=str, default="facebook/dinov2-small")
    parser.add_argument("--use_force", type=bool, default=True)
    parser.add_argument("--force_dim", type=int, default=3)

    # Output & device
    parser.add_argument("--task", type=str, default="mango")
    parser.add_argument("--data_dir", type=str, default="./data/datasets/mango_controller_40000")
    parser.add_argument("--checkpoint_dir", type=str, default="residual_controller/checkpoint/lstm_step_controller/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Add timestamp to checkpoint dir
    timestamp = datetime.now().strftime("%m%d_%H%M")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"{args.task}_{timestamp}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    controller = TactileLSTMController(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        image_model_path=args.image_model,
        device=args.device,
        use_force=args.use_force,
        force_dim=args.force_dim
    )

    # Train
    train_lstm_controller_with_dataset(
        controller=controller,
        args = args,
    )

if __name__ == "__main__":
    main()