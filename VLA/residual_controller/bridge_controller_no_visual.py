import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from visual_encoder import DINOv2Encoder
from bridge.bridge_model import StochasticInterpolants
from controller_dataset import denormalize_actions, normalize_actions
class DiffusionController:
    """
    Diffusion-based controller that refines VLA actions using a stochastic interpolants model.
    This controller uses a bridge diffusion approach to transition from VLA actions to expert-like actions.
    """

    def __init__(
            self,
            state_dim=10,
            hidden_dim=256,
            image_model_path="facebook/dinov2-small",
            diffusion_steps=10,
            device="cuda",
            model_args=None,
            use_force=True,
            force_dim = 3,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.diffusion_steps = diffusion_steps

        self.image_encoder = None
        self.latent_obs_dim = 0
        self.use_force = use_force
        self.force_dim = force_dim
        self.model_args = model_args
        self.stats = None

        if self.use_force:

            self.obs_dim = self.state_dim + self.force_dim
            self.state_encoder = nn.Sequential(
                nn.Linear(self.obs_dim , hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ).to(device)

            self.force_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, int(hidden_dim/2)),
                nn.GELU(),
                nn.Linear(int(hidden_dim/2), force_dim)
            ).to(device)

        else:

            self.obs_dim = self.state_dim
            self.state_encoder = nn.Sequential(
                nn.Linear(self.obs_dim , hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ).to(device)

        # Initialize the diffusion model
        self.diffusion_model = StochasticInterpolants()
        if self.model_args:
            self.diffusion_model.load_model(self.model_args, device)

        # Move models to device
        self.to(device)

    def to(self, device):
        """Move all models to the specified device."""
        self.device = device
        self.state_encoder.to(device)

        if self.use_force:
            self.force_decoder.to(device)
        return self


    def encode_observation(self, state, images_cam1=None, images_cam2=None, forces=None):
        """
        Encode state and (optional) image observations into a conditioning vector.

        Args:
            state: Robot state [batch_size, state_dim]
            images_cam1: Optional front camera image [batch_size, H, W, C]
            images_cam2: Optional right camera image [batch_size, H, W, C]

        Returns:
            obs_cond: Encoded observation for conditioning [batch_size, cond_dim]
        """
        # Ensure state is on the correct device
        state = state.to(self.device)
        if self.use_force:
            state = torch.cat((state,forces),dim=-1)

        obs_cond = self.state_encoder(state)

        return obs_cond


    def predict(self, state, vla_actions, images_cam1=None, images_cam2=None, forces = None):
        """
        Predict refined actions using the diffusion model.

        Args:
            state: Current robot state [batch_size, state_dim]
            vla_actions: VLA actions to refine [batch_size, horizon, state_dim]
            images_cam1: Optional front camera image [batch_size, H, W, C]
            images_cam2: Optional right camera image [batch_size, H, W, C]
            diffuse_steps: Number of diffusion steps (default: self.diffusion_steps)

        Returns:
            refined_actions: Refined actions [batch_size, state_dim]
        """
        # Set model to evaluation mode
        self.eval()

        obs_cond = self.encode_observation(state, images_cam1, images_cam2, forces)
        vla_actions_n = normalize_actions(vla_actions, self.stats, 'vla')

        # Sample from the diffusion model
        with torch.no_grad():
            refined_actions_n = self.diffusion_model.sample(
                x_prior=vla_actions_n,
                cond=obs_cond,
                diffuse_step=self.diffusion_steps
            )

        refined_actions = denormalize_actions(refined_actions_n,self.stats,'expert')

        return refined_actions


    def train(self):
        """Set models to training mode."""

        self.state_encoder.train()
        self.diffusion_model.train()
        if self.use_force:
            self.force_decoder.train()
        return self

    def eval(self):
        """Set models to evaluation mode."""

        self.state_encoder.eval()
        self.diffusion_model.eval()
        if self.use_force:
            self.force_decoder.eval()
        return self

    def save(self, path):
        """
        Save the controller model.

        Args:
            path: Path to save the model
        """
        state_dict = {
            'state_encoder': self.state_encoder.state_dict(),
            'model_args': self.model_args,
            'stats': self.stats
        }
        if self.use_force:
            state_dict['force_decoder'] = self.force_decoder.state_dict()

        # Save controller components
        torch.save(state_dict, f"{path}/controller.pt")

        # Save diffusion model separately
        self.diffusion_model.save_model(path)



    def load(self, path):
        """
        Load the controller model.

        Args:
            path: Path to load the model from
        """
        # Load controller components
        checkpoint = torch.load(f"{path}/controller.pt", map_location=self.device)

        self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        if self.use_force:
            self.force_decoder.load_state_dict(checkpoint['force_decoder'])

        self.model_args = checkpoint['model_args']
        self.stats = {key: torch.tensor(value, dtype=torch.float32).cuda() for key, value in checkpoint['stats'].items()}

        # Load diffusion model
        self.diffusion_model.load_model({**self.model_args, 'ckpt_path': path, 'pretrain': True}, self.device)

def load_bridge_controller():

    model_args = {
        'interpolant_type': 'linear',  # Interpolation type for bridge diffusion
        'gamma_type': '2^0.5*t(t-1)',  # Noise schedule for diffusion
        'epsilon_type': '1-t',  # Drift schedule for diffusion
        'prior_policy': 'vla',  # Use VLA as prior (source actions)
        'beta_max': 0.03,  # Maximum noise scale
        'sde_type': 'vs',  # Use velocity-score SDE
        'action_dim': 10,  # Dimension of actions
        'obs_dim': 256,  # Dimension of observations
        'obs_horizon': 1,  # Single observation frame
        'net_type': 'unet1D_si',  # Network type
        'pretrain': False,  # Not using pretrained weights
        'context_frames': 2,  # Number of context frames
        'horizon': 16,  # Number of prediction steps
    }

    controller = DiffusionController(
        state_dim=10,
        hidden_dim=256,
        image_model_path="facebook/dinov2-small",
        diffusion_steps=10,
        model_args=model_args,
        force_dim=3,
    )

    return controller



def test_bridge_tensors():
    """Simple test to verify the BridgeLSTM controller tensor passing"""
    print("Testing BridgeLSTM controller tensor passing...")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    batch_size = 2
    state_dim = 10
    force_dim = 3
    horizon = 16
    hidden_dim = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create mock model_args
    model_args = {
        'action_dim': state_dim,
        'horizon': horizon,
        'interpolant_type': 'linear',  # Interpolation type for bridge diffusion
        'gamma_type': '2^0.5*t(t-1)',  # Noise schedule for diffusion
        'epsilon_type': '1-t',  # Drift schedule for diffusion
        'prior_policy': 'vla',  # Use VLA as prior (source actions)
        'beta_max': 0.03,  # Maximum noise scale
        'sde_type': 'vs',  # Use velocity-score SDE
        'obs_dim': 256,  # Dimension of observations
        'obs_horizon': 1,  # Single observation frame
        'net_type': 'unet1D_si',  # Network type
        'pretrain': False,  # Not using pretrained weights
        'context_frames': 2,  # Number of context frames
    }

    # Create controller
    controller = DiffusionController(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        device=device,
        model_args=model_args,
        force_dim=force_dim,
        use_force = True
    )

    # Mock the stats
    controller.stats = {
        'vla_mins': torch.zeros(state_dim, device=device),
        'vla_maxs': torch.ones(state_dim, device=device),
        'vla_range': torch.ones(state_dim, device=device),
        'action_mins': torch.zeros(state_dim, device=device),
        'action_maxs': torch.ones(state_dim, device=device),
        'action_range': torch.ones(state_dim, device=device)
    }

    # VLA trajectory: [batch_size, horizon, state_dim]
    state = torch.randn(batch_size, state_dim, device=device)
    vla_actions = torch.randn(batch_size, horizon, state_dim, device=device)
    forces = torch.randn(batch_size, force_dim, device=device)
    # obs_cond = torch.rand(batch_size, 384, device=device)

    image_height, image_width = 384,384
    # Camera images: [batch_size, channels, height, width]
    # Note: DINOv2 expects RGB images
    images_cam1 = torch.rand(batch_size, 3, image_height, image_width, device=device)
    images_cam2 = torch.rand(batch_size, 3, image_height, image_width, device=device)


    with torch.no_grad():
        s = time.time()
        refined_trajectory = controller.predict(
            state=state,
            vla_actions=vla_actions,
            images_cam1=images_cam1,
            images_cam2=images_cam2,
            forces = forces
        )
        e = time.time()
        print(f"inference time:{e-s}")

        print(state.shape)
        print(vla_actions.shape)
        print(refined_trajectory.shape)


if __name__ == "__main__":

    test_bridge_tensors()
