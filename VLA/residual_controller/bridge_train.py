import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# Import the existing dataset class
from controller_dataset import ControllerDataModule, normalize_actions, denormalize_actions
import argparse
from bridge_controller import DiffusionController
from bridge_test import test_diffusion_controller


def log_info(writer, loss_info, itr):
    # Use proper tensorboard add_scalar method (tag, scalar_value, global_step)
    writer.add_scalar('Loss/v_loss', loss_info['v_loss'].item(), itr)
    writer.add_scalar('Loss/s_loss', loss_info['s_loss'].item(), itr)
    writer.add_scalar('Loss/b_loss', loss_info['b_loss'].item(), itr)


class DiffusionControllerTrainer:
    """
    Trainer for the diffusion controller using the existing ControllerDataset.
    Handles training the StochasticInterpolants model on action refinement data.
    """

    def __init__(
            self,
            controller,
            data_module,
            learning_rate=1e-4,
            weight_decay=1e-6,
            checkpoint_dir='checkpoint/bridge_controller/',
            device='cuda'
    ):
        self.controller = controller
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Setup optimizer for the diffusion model and controller components
        self.optimizer = optim.AdamW(
            list(controller.diffusion_model.net.parameters()) +
            list(controller.state_encoder.parameters()),
            # list(controller.force_decoder.parameters()) +
            # list(controller.force_dynamics_model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000, eta_min=learning_rate / 10
        )

        # Setup logger
        self.logger = self._setup_logger()

        # Setup TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir)

        self.use_force = self.controller.use_force

        self.controller.stats = data_module.stats

        stats_tensors = {key: torch.tensor(value, dtype=torch.float32) for key, value in data_module.stats.items()}
        stats_gpu = {key: value.cuda() for key, value in stats_tensors.items()}

        self.stats = stats_gpu



    def _setup_logger(self):
        """Setup logging configuration."""
        logger = logging.getLogger('diffusion_controller_training')
        logger.setLevel(logging.INFO)

        # Create file handler
        fh = logging.FileHandler(os.path.join(self.checkpoint_dir, 'training.log'))
        fh.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _prepare_batch_for_diffusion(self, batch):
        """
        Convert a batch from ControllerDataset format to the format expected by the diffusion model.

        Args:
            batch: A batch from ControllerDataset with:
                - states: Robot states [batch_size, context_frames + horizon, state_dim]
                - vla_actions: VLA actions [batch_size, horizon, state_dim]
                - expert_actions: Expert actions [batch_size, horizon, state_dim]
                - images_cam1: Front camera images [batch_size, context_frames, H, W, 3]
                - images_cam2: Right camera images [batch_size, context_frames, H, W, 3]

        Returns:
            batch_dict: A batch dictionary for the diffusion model
        """
        # Extract data from batch
        states = batch['states']  # [batch_size, context_frames + horizon, state_dim]
        forces = batch['forces']
        context_frames = self.controller.model_args.get('context_frames', 2)

        # Current state is the last context frame
        current_state = states[:, context_frames - 1]  # [batch_size, state_dim]
        current_forces = forces[:, context_frames - 1] # [batch_size, force_dim]
        future_forces =  forces[:, context_frames:] # [batch_size, horizon, force_dim]

        # VLA actions (source)
        vla_actions = batch['vla_actions']  # [batch_size, horizon, state_dim]
        # Expert actions (target)
        expert_actions = batch['expert_actions']  # [batch_size, horizon, state_dim]

        vla_actions_n = normalize_actions(vla_actions,self.stats,'vla')
        expert_actions_n = normalize_actions(expert_actions,self.stats,'expert')

        # Images (if available)
        images_cam1 = batch.get('images_cam1', None)  # [batch_size, context_frames, H, W, 3]
        images_cam2 = batch.get('images_cam2', None)  # [batch_size, context_frames, H, W, 3]

        # Get the most recent images
        if images_cam1 is not None and images_cam2 is not None:
            current_img_cam1 = images_cam1[:, -1]  # [batch_size, H, W, 3]
            current_img_cam2 = images_cam2[:, -1]  # [batch_size, H, W, 3]
        else:
            current_img_cam1 = None
            current_img_cam2 = None

        # Encode the observation (state + images)
        obs_cond = self.controller.encode_observation(
            current_state, current_img_cam1, current_img_cam2, current_forces
        )

        # Create batch_dict for diffusion model
        batch_dict = {
            'obs_cond': obs_cond,
            'expert_act': expert_actions_n,  # Target expert action
            'vla_act': vla_actions_n,  # Source VLA action
            'forces': future_forces,
            'current_force':current_forces,
        }

        return batch_dict

    def train(
            self,
            data_module,
            num_epochs=100,
            save_interval=25,  # Increased from 10 to 25 to reduce checkpoint frequency
            eval_interval=5,
            log_interval=100,
            diffusion_steps_schedule=None
    ):
        """
        Train the diffusion controller using ControllerDataModule with memory optimization.
        Args:
            data_module: ControllerDataModule for data loading
            num_epochs: Number of epochs to train for
            save_interval: Epoch interval for saving checkpoints
            eval_interval: Epoch interval for evaluation
            log_interval: Step interval for logging
            diffusion_steps_schedule: Optional schedule for diffusion steps during training
        """
        # Get data loaders
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()

        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        last_saved_epoch = 0  # Track when we last saved a checkpoint

        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Train dataset size: {len(data_module.train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(data_module.val_dataset)}")

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train for one epoch - now returns total_loss
            train_total_loss = self._train_epoch(
                train_dataloader,
                global_step,
                log_interval,
                diffusion_steps_schedule,
                epoch + 1
            )
            global_step += len(train_dataloader)

            # Note: No need to log train_loss again here as it's already logged in _train_epoch
            # with more detailed breakdown of all loss components

            # Evaluate if needed
            if (epoch + 1) % eval_interval == 0 or (epoch + 1)==1:
                val_loss = self._evaluate(val_dataloader,epoch)
                self.logger.info(f"Epoch {epoch + 1} | Validation Loss: {val_loss:.4f}")
                self.writer.add_scalar('Epoch/val_loss', val_loss, epoch + 1)

                # Save checkpoint only if validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    # Remove previous best model to save space before saving new one
                    self._remove_checkpoint("best_model")
                    self._save_checkpoint("best_model")
                    last_saved_epoch = epoch + 1
                    self.logger.info(f"Saved best model with validation loss: {val_loss:.4f}")

                # elif epoch - last_saved_epoch > eval_interval * 7:
                #     self.logger.info(f"[EARLY STOP]: Loss not reduced since {last_saved_epoch}")
                #     self.logger.info("Training completed")
                #     self.writer.close()
                #     break

            # Save periodic checkpoint if needed (at reduced frequency)
            # Only save if we haven't already saved based on validation performance
            if (epoch + 1) % save_interval == 0 or (epoch + 1)==1:
                checkpoint_name = f"epoch_{epoch + 1}"

                # Remove previous periodic checkpoint if it exists to save space
                previous_checkpoint = f"epoch_{epoch + 1 - save_interval}"
                self._remove_checkpoint(previous_checkpoint)

                self._save_checkpoint(checkpoint_name)
                self.logger.info(f"Saved periodic checkpoint at epoch {epoch + 1}")

        self.logger.info("Training completed")
        # Close the writer
        self.writer.close()

    def _remove_checkpoint(self, checkpoint_name):
        """
        Remove a checkpoint to free up disk space.
        Args:
            checkpoint_name: Name of the checkpoint to remove
        """
        import os
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                self.logger.info(f"Removed checkpoint: {checkpoint_name}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {checkpoint_name}: {e}")

    def _train_epoch(
            self,
            dataloader,
            global_step,
            log_interval,
            diffusion_steps_schedule,
            epoch
    ):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data
            global_step: Global step counter
            log_interval: Step interval for logging
            diffusion_steps_schedule: Optional schedule for diffusion steps
            epoch: Current epoch number

        Returns:
            epoch_loss: Average loss for the epoch
        """
        self.controller.train()
        epoch_diff_loss = 0.0
        epoch_force_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = len(dataloader)

        # Training loop
        pbar = tqdm(enumerate(dataloader), total=num_batches, desc="Training")
        for batch_idx, batch in pbar:
            step = global_step + batch_idx

            # Update diffusion steps if scheduled
            if diffusion_steps_schedule is not None:
                self.controller.diffusion_steps = diffusion_steps_schedule(step)

            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Prepare batch for diffusion model
            batch_dict = self._prepare_batch_for_diffusion(batch)

            # Zero gradients
            self.optimizer.zero_grad()

            # Get loss from diffusion model
            diff_loss, loss_info = self.controller.diffusion_model.get_loss(
                batch_dict,
                self.device
            )

            if self.use_force:
                force_loss = torch.Tensor([0]).to(self.device)
                total_loss = diff_loss

            else:

                force_loss = torch.Tensor([0]).to(self.device)
                total_loss = diff_loss

            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()

            # Update EMA
            self.controller.diffusion_model.ema.update()

            # Step scheduler
            self.scheduler.step()

            # Update metrics
            epoch_diff_loss += diff_loss.item()
            epoch_force_loss += force_loss.item()
            epoch_total_loss += total_loss.item()

            # Log metrics
            if step % log_interval == 0:
                # Using the log_info function for diffusion loss components
                log_info(self.writer, loss_info, step)

                # Additional logging for dynamics, force, and total losses
                self.writer.add_scalar('Loss/diffusion', diff_loss.item(), step)
                self.writer.add_scalar('Loss/force', force_loss.item(), step)
                self.writer.add_scalar('Loss/total', total_loss.item(), step)

                # Update progress bar with all loss components
                pbar.set_postfix({
                    'diff_loss': diff_loss.item(),
                    'force_loss': force_loss.item(),
                    'total_loss': total_loss.item(),
                    'v_loss': loss_info['v_loss'].item(),
                    's_loss': loss_info['s_loss'].item(),
                    'b_loss': loss_info['b_loss'].item(),
                    # 'lr': self.scheduler.get_last_lr()[0]
                })

        # Calculate average losses for the epoch
        avg_diff_loss = epoch_diff_loss / num_batches
        avg_force_loss = epoch_force_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches

        # Log epoch-level metrics
        self.writer.add_scalar('Epoch/diffusion_loss', avg_diff_loss, epoch)
        self.writer.add_scalar('Epoch/force_loss', avg_force_loss, epoch)
        self.writer.add_scalar('Epoch/total_loss', avg_total_loss, epoch)

        self.logger.info(
            f"Epoch {epoch} | Diffusion Loss: {avg_diff_loss:.4f}")

        return avg_total_loss  # Return the total loss instead of just diffusion loss

    def _evaluate(self, dataloader, epoch):
        """
        Evaluate the model on validation data.

        Args:
            dataloader: DataLoader for validation data

        Returns:
            val_loss: Average validation total loss
        """
        self.controller.eval()
        val_diff_loss = 0.0
        val_dyna_loss = 0.0
        val_force_loss = 0.0
        val_total_loss = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Prepare batch for diffusion model
                batch_dict = self._prepare_batch_for_diffusion(batch)

                # Get loss from diffusion model
                diff_loss, _ = self.controller.diffusion_model.get_loss(
                    batch_dict,
                    self.device
                )

                if self.use_force:
                    # Get dynamics and force losses
                    force_loss = torch.Tensor([0]).to(self.device)
                    total_loss = diff_loss + force_loss
                else:
                    force_loss = torch.Tensor([0]).to(self.device)
                    total_loss = diff_loss

                val_diff_loss += diff_loss.item()
                val_force_loss += force_loss.item()
                val_total_loss += total_loss.item()

        # Calculate average losses
        avg_diff_loss = val_diff_loss / num_batches
        avg_force_loss = val_force_loss / num_batches
        avg_total_loss = val_total_loss / num_batches

        # Log all validation losses
        # epoch = self.writer.get_step() // len(dataloader)  # Estimate current epoch
        self.writer.add_scalar('Validation/diffusion_loss', avg_diff_loss, epoch)
        self.writer.add_scalar('Validation/force_loss', avg_force_loss, epoch)
        self.writer.add_scalar('Validation/total_loss', avg_total_loss, epoch)

        # Log detailed validation losses
        self.logger.info(
            f"Validation | Diffusion Loss: {avg_diff_loss:.4f} | Total Loss: {avg_total_loss:.4f}")

        # if (epoch+1) % 50==0 or (epoch + 1)==1:
        #     self._run_test_during_training(dataloader, epoch+1)

        return avg_total_loss  # Return total loss for model selection and early stopping

    def _run_test_during_training(self, dataloader, epoch):
        """
        Run test during training to evaluate controller performance.

        Args:
            epoch: Current epoch number
        """
        # First save current model as a temporary checkpoint so we can test it
        temp_checkpoint_dir = os.path.join(self.checkpoint_dir, f"temp_epoch_{epoch}")
        os.makedirs(temp_checkpoint_dir, exist_ok=True)
        self._save_checkpoint(f"temp_epoch_{epoch}")

        self.logger.info(f"Running test during training at epoch {epoch}")

        # Run test with only a few samples to keep evaluation efficient during training
        test_results = test_diffusion_controller(
            checkpoint_dir=temp_checkpoint_dir,
            data_loader=dataloader,
            context_frames=self.controller.model_args.get('context_frames', 2),
            horizon=self.controller.model_args.get('horizon', 8),
            diffusion_steps=self.controller.diffusion_steps,
            device=self.device,
            num_samples=50,
            state_dim=self.controller.state_dim,
            hidden_dim=self.controller.hidden_dim,
            force_dim=self.controller.force_dim,
            use_force=self.use_force,
            visualize=True,
            save_dir=os.path.join(self.checkpoint_dir, f"test_visualizations_epoch_{epoch}")
        )

        # Log test results
        self.writer.add_scalar('Test/avg_error', test_results['avg_error'], epoch)
        self.writer.add_scalar('Test/avg_vla_error', test_results['avg_vla_error'], epoch)
        self.writer.add_scalar('Test/improvement_pct', test_results['improvement'], epoch)

        self.logger.info(
            f"Test | Epoch {epoch} | Avg Error: {test_results['avg_error']:.4f} | "
            f"Avg VLA Error: {test_results['avg_vla_error']:.4f} | "
            f"Improvement: {test_results['improvement']:.2f}%")

        # Clean up temporary checkpoint if needed
        import shutil
        shutil.rmtree(temp_checkpoint_dir)

    def _save_checkpoint(self, name):
        """
        Save a checkpoint of the model.

        Args:
            name: Name for the checkpoint
        """
        save_path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(save_path, exist_ok=True)

        # Save controller
        self.controller.save(save_path)

        # # Save optimizer and scheduler
        # torch.save({
        #     'optimizer': self.optimizer.state_dict(),
        #     'scheduler': self.scheduler.state_dict(),
        # }, os.path.join(save_path, 'optimizer.pt'))
        #
        # self.logger.info(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, path):
        """
        Load a checkpoint.

        Args:
            path: Path to the checkpoint
        """
        # Load controller
        self.controller.load(path)

        # Load optimizer and scheduler if available
        optimizer_path = os.path.join(path, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            checkpoint = torch.load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            self.logger.info(f"Loaded checkpoint: {path}")
        else:
            self.logger.warning(f"Optimizer checkpoint not found at: {optimizer_path}")


def train_diffusion_controller_with_dataset(
        controller,
        data_dir,
        context_frames=2,
        horizon=8,
        batch_size=32,
        num_workers=4,
        image_size=384,
        num_epochs=100,
        learning_rate=1e-4,
        weight_decay=1e-6,
        save_interval=10,
        eval_interval=5,
        log_interval=100,
        checkpoint_dir='checkpoints',
        device='cuda',
        val_ratio=0.1,
        stride=1,
        diffusion_steps_schedule=None
):
    """
    Train a diffusion controller using the existing ControllerDataset.

    Args:
        controller: Diffusion controller to train
        data_dir: Directory containing the dataset
        context_frames: Number of past frames to use as context
        horizon: Number of future steps to predict
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Size of images
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        save_interval: Epoch interval for saving checkpoints
        eval_interval: Epoch interval for evaluation
        log_interval: Step interval for logging
        log_dir: Directory for logs
        checkpoint_dir: Directory for checkpoints
        device: Device to use for training
        val_ratio: Ratio of data to use for validation
        stride: Stride for sampling sequences
        diffusion_steps_schedule: Optional schedule for diffusion steps during training
    """
    # Update controller model args with dataset parameters
    controller.model_args.update({
        'context_frames': context_frames,
        'horizon': horizon,
    })

    # Create data module
    data_module = ControllerDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        context_frames=context_frames,
        horizon=horizon,
        image_size=image_size,
        val_ratio=val_ratio,
        stride=stride
    )

    # Create trainer
    trainer = DiffusionControllerTrainer(
        controller=controller,
        data_module = data_module,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    # Train the controller
    trainer.train(
        data_module=data_module,
        num_epochs=num_epochs,
        save_interval=save_interval,
        eval_interval=eval_interval,
        log_interval=log_interval,
        diffusion_steps_schedule=diffusion_steps_schedule
    )

    return controller


def main():

    args = parse_args()
    timestamp = datetime.now().strftime("%m%d_%H%M")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"{args.task}_{timestamp}")
    print(f"SAVE CKPT to : {args.checkpoint_dir}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Define diffusion model arguments
    model_args = {
        'interpolant_type': 'linear',  # Interpolation type for bridge diffusion
        'gamma_type': '2^0.5*t(t-1)',  # Noise schedule for diffusion
        'epsilon_type': '1-t',  # Drift schedule for diffusion # 1-t
        'prior_policy': 'vla',  # Use VLA as prior (source actions)
        'beta_max': args.beta_max,  # Maximum noise scale
        'sde_type': 'vs',  # Use velocity-score SDE
        'action_dim': args.state_dim,  # Dimension of actions
        'obs_dim': args.hidden_dim,  # Dimension of observations
        'obs_horizon': 1,  # Single observation frame
        'net_type': 'unet1D_si',  # Network type
        'pretrain': False,  # Not using pretrained weights
        'context_frames': 2,  # Number of context frames
        'horizon': args.horizon,  # Number of prediction steps
    }

    # Create diffusion controller
    controller = DiffusionController(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        image_model_path=args.image_model,
        diffusion_steps=args.diffusion_steps,
        device=args.device,
        model_args=model_args,
        use_force=args.use_force,
        force_dim=args.force_dim,
    )

    # Train the controller
    train_diffusion_controller_with_dataset(
        controller=controller,
        data_dir=args.data_dir,
        context_frames=args.context_frames,
        horizon=args.horizon,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        stride=args.stride,
        diffusion_steps_schedule=None  # No schedule for diffusion steps
    )




def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion-based controller")

    # Dataset arguments
    parser.add_argument("--context_frames", type=int, default=2,
                        help="Number of past frames to use as context")
    parser.add_argument("--horizon", type=int, default=32,
                        help="Number of future steps to predict")
    parser.add_argument("--image_size", type=int, default=384,
                        help="Size of images")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for sampling sequences")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--num_epochs", type=int, default=400,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay for optimizer")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Epoch interval for saving checkpoints")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Epoch interval for evaluation")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Epoch interval for logging")

    # Model arguments
    parser.add_argument("--state_dim", type=int, default=10,
                        help="Dimension of the robot state and action")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for encoders")
    parser.add_argument("--image_model", type=str, default="facebook/dinov2-small",
                        help="Path to the DINOv2 model to use")
    parser.add_argument("--diffusion_steps", type=int, default=10,
                        help="Number of diffusion steps for sampling")
    parser.add_argument("--beta_max", type=float, default=0.03,
                        help="Maximum noise scale for diffusion")
    parser.add_argument("--use_force", type=bool, default=True,
                        help="Maximum noise scale for diffusion")
    parser.add_argument("--force_dim", type=int, default=3,
                        help="Dimension of the robot state and action")

    # Output arguments
    parser.add_argument("--task", type=str, default="mango",
                        help="Directory containing the dataset")
    parser.add_argument("--data_dir", type=str, default="./data/datasets/mango_controller_40000",
                        help="Directory containing the dataset")

    parser.add_argument("--checkpoint_dir", type=str, default="residual_controller/checkpoint/bridge_controller/",
                        help="Directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")

    return parser.parse_args()

if __name__ == "__main__":

    main()



