#!/usr/bin/env python
# coding: UTF-8
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from visual_encoder import DINOv2Encoder
from controller_dataset import denormalize_actions,normalize_actions


class TactileLSTMController:
    """
    Tactile-aware LSTM Controller that refines VLA actions using force feedback.

    This controller supports two operational modes:
    1. Training mode: Processes entire action chunks with corresponding force sequences
    2. Prediction mode: Processes one step at a time, maintaining internal state

    Both modes use the same underlying LSTM network.
    """

    def __init__(
            self,
            state_dim=10,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            image_model_path="facebook/dinov2-small",
            device="cuda",
            force_dim=3,
            use_force = True,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.force_dim = force_dim
        self.use_force = use_force
        # Image encoder
        self.image_encoder = DINOv2Encoder(model_name=image_model_path, device=device)
        self.latent_obs_dim = self.image_encoder.hidden_size

        # Force encoder (processes force at each timestep)
        self.force_encoder = nn.Sequential(
            nn.Linear(force_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        ).to(device)

        # Total observation dimension (image features + state)
        self.obs_dim = self.latent_obs_dim * 2 + self.state_dim
        # Initial observation encoder (images + state)
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)

        # LSTM input combines only observation encoding and force encoding
        self.lstm_input_dim = hidden_dim // 2 + state_dim  # force encoder output + vla action

        # LSTM network - single common network for both training and inference
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,  # Important: unidirectional for sequential processing
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        ).to(device)

        # Output head for predicting action deltas - takes LSTM output and current VLA action
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # Added state_dim for VLA action
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim)
        ).to(device)

        # Residual connection for VLA actions
        self.use_residual = True

        # For stateful prediction
        self.hidden_state = None
        self.cell_state = None

        self.trainable_modules = [
            self.obs_encoder,
            self.force_encoder,
            self.lstm,
            self.output_head
        ]

    def to(self, device):
        """Move all models to the specified device."""
        self.device = device
        for module in self.trainable_modules:
            module.to(device)
        return self

    def encode_images(self, images_cam1, images_cam2):
        """
        Process images from both cameras through the DINOv2 encoder.

        Args:
            images_cam1: Front camera images [batch_size, H, W, C]
            images_cam2: Right camera images [batch_size, H, W, C]

        Returns:
            (cam1_features, cam2_features): Encoded image features
        """
        # Ensure images are on the correct device
        images_cam1 = images_cam1.to(self.device)
        images_cam2 = images_cam2.to(self.device)

        # Extract features using DINOv2
        cam1_features = self.image_encoder.forward(images_cam1)
        cam2_features = self.image_encoder.forward(images_cam2)

        return cam1_features, cam2_features

    def encode_observation(self, state, images_cam1, images_cam2):
        """
        Encode the static observation (state and images) into a latent representation.

        Args:
            state: Robot state [batch_size, state_dim]
            images_cam1: Front camera images
            images_cam2: Right camera images

        Returns:
            obs_cond: Encoded observation [batch_size, hidden_dim]
        """
        state = state.to(self.device)

        cam1_features, cam2_features = self.encode_images(images_cam1, images_cam2)

        combined_features = torch.cat((cam1_features, cam2_features, state), dim=-1)

        obs_cond = self.obs_encoder(combined_features)

        return obs_cond

    def encode_force(self, force):
        """
        Encode force signals.

        Args:
            force: Force measurements [batch_size, T, force_dim] or [batch_size, force_dim]

        Returns:
            encoded_force: Encoded force [batch_size, T, hidden_dim//2] or [batch_size, hidden_dim//2]
        """
        # Handle both sequence and single step inputs
        orig_shape = force.shape
        if len(orig_shape) == 3:
            # Sequence input: [batch_size, T, force_dim]
            B, T, _ = orig_shape
            force = force.reshape(B * T, -1)
            encoded = self.force_encoder(force)
            return encoded.reshape(B, T, -1)
        else:
            # Single step input: [batch_size, force_dim]
            return self.force_encoder(force)

    def forward(self, batch_dict):
        """
        Forward pass for training mode - processes entire sequences.

        Args:
            batch_dict: Dictionary containing:
                - obs_cond: Encoded observations [batch_size, hidden_dim]
                - vla_act: VLA actions [batch_size, T, state_dim]
                - force_seq: Force measurements [batch_size, T, force_dim]

        Returns:
            refined_actions: Refined action sequence [batch_size, T, state_dim]
        """
        vla_actions = batch_dict['vla_act']  # normalized [B, T, state_dim]
        obs_cond = batch_dict['obs_cond']  # [B, hidden_dim]
        force_seq = batch_dict['forces']  # [B, T, force_dim]

        B, T, _ = vla_actions.shape

        # Encode force sequence
        encoded_force = self.encode_force(force_seq)  # [B, T, hidden_dim//2]

        # LSTM input only has observation and force (no VLA action)
        lstm_input = torch.cat([encoded_force, vla_actions], dim=-1)  # [B, T, lstm_input_dim]

        # Initialize hidden states for training
        h0 = torch.zeros(self.lstm.num_layers, B, self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.lstm.num_layers, B, self.hidden_dim, device=self.device)

        # Process through LSTM
        lstm_output, _ = self.lstm(lstm_input, (h0, c0))  # [B, T, hidden_dim]

        # Combine LSTM output with corresponding VLA action for each timestep
        obs_cond_expanded = obs_cond.unsqueeze(1).repeat(1, T, 1)
        combined = torch.cat([lstm_output, obs_cond_expanded], dim=-1)  # [B, T, hidden_dim + state_dim]

        # Predict action deltas
        delta = self.output_head(combined)  # [B, T, state_dim]

        # Apply residual connection if enabled
        if self.use_residual:
            return vla_actions + delta
        else:
            return delta

    def reset_state(self, batch_size=1):
        """
        Reset the internal LSTM state for a new prediction sequence.
        """
        self.hidden_state = torch.zeros(
            self.lstm.num_layers,
            batch_size,
            self.hidden_dim,
            device=self.device
        )
        self.cell_state = torch.zeros(
            self.lstm.num_layers,
            batch_size,
            self.hidden_dim,
            device=self.device
        )

    def predict(self, obs_cond, vla_action, force, initialize=False):
        """
        Predict a single step of refined action based on the current force feedback.

        Args:
            obs_cond: Encoded observation [batch_size, hidden_dim]
            vla_action: Current VLA action [batch_size, state_dim]
            force: Current force measurement [batch_size, force_dim]
            initialize: Whether to initialize/reset the LSTM state

        Returns:
            refined_action: Refined action for current step [batch_size, state_dim]
        """
        self.eval()

        with torch.no_grad():
            batch_size = vla_action.shape[0]

            # Reset LSTM state if requested or not initialized
            if initialize or self.hidden_state is None:
                self.reset_state(batch_size)

            # Ensure inputs are on the correct device
            vla_action = vla_action.to(self.device)
            force = force.to(self.device)

            # Encode force
            encoded_force = self.encode_force(force)  # [B, hidden_dim//2]

            # Prepare LSTM input (only obs_cond and force)
            lstm_input = torch.cat([
                encoded_force,
                vla_action
            ], dim=-1).unsqueeze(1)  # Add time dimension: [B, 1, lstm_input_dim]

            # Run through LSTM with state
            lstm_output, (self.hidden_state, self.cell_state) = self.lstm(
                lstm_input,
                (self.hidden_state, self.cell_state)
            )  # output: [B, 1, hidden_dim]

            # Combine LSTM output with current VLA action
            combined = torch.cat([
                lstm_output.squeeze(1),  # [B, hidden_dim]
                obs_cond  # [B, state_dim]
            ], dim=-1)  # [B, hidden_dim + state_dim]

            # Predict action delta
            delta = self.output_head(combined)  # [B, state_dim]

            # Apply residual connection if enabled
            if self.use_residual:
                return denormalize_actions(vla_action + delta, self.stats, 'expert')
            else:
                return denormalize_actions(delta, self.stats, 'expert')

    def predict_sequence(self, obs_cond, vla_actions, force_seq):
        """
        Predict a sequence by running step-by-step, using the maintained state.

        Args:
            obs_cond: Encoded observation [batch_size, hidden_dim]
            vla_actions: VLA action sequence [batch_size, T, state_dim]
            force_seq: Force measurements sequence [batch_size, T, force_dim]

        Returns:
            refined_actions: Refined action sequence [batch_size, T, state_dim]
        """
        B, T, _ = vla_actions.shape
        refined_actions = []

        # Reset state for a new sequence
        self.reset_state(batch_size=B)

        vla_actions_n = normalize_actions(vla_actions, self.stats, 'vla')

        # Process one step at a time
        for t in range(T):
            refined_action = self.predict(
                obs_cond=obs_cond,
                vla_action=vla_actions_n[:, t],
                force=force_seq[:, t],
                initialize=(t == 0)
            )
            refined_actions.append(refined_action)

        # Stack actions into a sequence
        return torch.stack(refined_actions, dim=1)  # [B, T, state_dim]

    def get_loss(self, batch_dict):
        """
        Calculate loss during training.

        Args:
            batch_dict: Dictionary containing:
                - obs_cond: Encoded observations
                - vla_act: VLA actions
                - force_seq: Force measurements
                - expert_act: Expert actions (ground truth)

        Returns:
            loss: MSE loss between predictions and expert actions
        """
        pred = self.forward(batch_dict)
        target = batch_dict['expert_act']
        return F.mse_loss(pred, target)

    def train(self, mode=True):
        """Set the model to training mode."""
        for module in self.trainable_modules:
            module.train(mode)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        for module in self.trainable_modules:
            module.eval()
        return self

    def save(self, path):
        """Save model checkpoints."""
        state_dict = {
            'stats': self.stats,
            'model_args': getattr(self, 'model_args', None),
            'modules': {
                'obs_encoder': self.obs_encoder.state_dict(),
                'force_encoder': self.force_encoder.state_dict(),
                'lstm': self.lstm.state_dict(),
                'output_head': self.output_head.state_dict(),
            }
        }
        torch.save(state_dict, f"{path}/tactile_controller.pt")

    def load(self, path):
        """Load model from checkpoint."""
        checkpoint = torch.load(f"{path}/tactile_controller.pt", map_location=self.device)
        modules = checkpoint['modules']

        self.obs_encoder.load_state_dict(modules['obs_encoder'])
        self.force_encoder.load_state_dict(modules['force_encoder'])
        self.lstm.load_state_dict(modules['lstm'])
        self.output_head.load_state_dict(modules['output_head'])

        self.stats = {
            key: torch.tensor(value, dtype=torch.float32).to(self.device)
            for key, value in checkpoint['stats'].items()
        }
        self.model_args = checkpoint.get('model_args', None)


def load_lstm_controller():
    controller = TactileLSTMController(
        state_dim=state_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        device=device,
        force_dim=force_dim,
    )
    return controller

# Example usage
if __name__ == "__main__":
    # Settings
    batch_size = 8
    horizon = 64
    state_dim = 10
    force_dim = 3
    image_height, image_width = 384, 384
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy inputs
    state = torch.randn(batch_size, state_dim).to(device)
    images_cam1 = torch.randn(batch_size, 3, image_height, image_width).to(device)
    images_cam2 = torch.randn(batch_size, 3, image_height, image_width).to(device)

    # Create controller
    controller = TactileLSTMController(
        state_dim=state_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        device=device,
        force_dim=force_dim,
    )

    # Set dummy stats for denormalization
    controller.stats = {
        'vla_mins': torch.zeros(state_dim, device=device),
        'vla_range': torch.ones(state_dim, device=device),
        'action_mins': torch.zeros(state_dim, device=device),
        'action_range': torch.ones(state_dim, device=device)
    }

    print("Testing controller...")

    # Test observation encoding
    obs_cond = controller.encode_observation(state, images_cam1, images_cam2)
    print(f"Observation encoding shape: {obs_cond.shape}")

    # Test training mode (entire sequence)
    vla_actions = torch.randn(batch_size, horizon, state_dim).to(device)
    force_seq = torch.randn(batch_size, horizon, force_dim).to(device)
    expert_actions = torch.randn(batch_size, horizon, state_dim).to(device)

    batch_dict = {
        'obs_cond': obs_cond,
        'vla_act': vla_actions,
        'forces': force_seq,
        'expert_act': expert_actions
    }

    refined_actions = controller.forward(batch_dict)
    print(f"Training mode output shape: {refined_actions.shape}")

    # Test prediction mode (step by step)
    controller.reset_state(batch_size)

    # Predict first step
    s1 = time.time()
    first_action = controller.predict(
        obs_cond=obs_cond,
        vla_action=vla_actions[:, 0],
        force=force_seq[:, 0],
        initialize=True
    )
    e1 = time.time()
    print(f"Single step prediction shape: {first_action.shape}, takes time:{e1-s1}")

    # Test full sequence prediction
    s2 = time.time()
    sequence_pred = controller.predict_sequence(obs_cond, vla_actions, force_seq)
    e2 = time.time()
    print(f"Sequence prediction shape: {sequence_pred.shape}, takes time:{e2-s2}")

    print("All tests passed! ✅")