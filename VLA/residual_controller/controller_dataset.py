#!/usr/bin/env python
# coding: UTF-8
"""
Dataset class for training a controller that maps VLA actions to expert actions.
This loads proprioception, images, and VLA actions from the augmented dataset.
"""

import os
import fnmatch
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import re
import tqdm
from scripts.utils_eef import *

def natural_sort_filenames(file_list):
    """
    Sort filenames naturally, so that episode_2.h5 comes before episode_10.h5
    """

    def extract_number(filename):
        match = re.search(r'episode_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0  # Default if no number found

    return sorted(file_list, key=extract_number)

class ControllerDataset(Dataset):
    """Dataset for training a controller that maps VLA actions to expert actions"""

    def __init__(self,
                 data_dir,
                 file_paths=None,
                 context_frames=2,  # Number of frames to use as context
                 horizon=8,  # Number of future steps to predict
                 use_images=False,  # Whether to include images
                 image_size=384,  # Size of images
                 stride=1):  # Stride for sampling sequences
        """
        Initialize the dataset for training the controller.

        Args:
            data_dir: Directory containing the augmented dataset with VLA actions
            context_frames: Number of past frames to include as context
            horizon: Number of future steps to predict
            use_images: Whether to include images in the dataset
            image_size: Size of images
            stride: Stride for sampling sequences
        """
        self.data_dir = data_dir
        self.context_frames = context_frames
        self.horizon = horizon
        self.use_images = use_images
        self.image_size = image_size
        self.stride = stride

        # Find all h5 files
        if file_paths is None:
            self.file_paths = []
            for root, _, files in os.walk(data_dir):
                for filename in natural_sort_filenames(fnmatch.filter(files, '*.h5')):
                    self.file_paths.append(os.path.join(root, filename))
        else:
            self.file_paths = file_paths

        # Create an index mapping for efficient sampling
        self.create_index_mapping()
        self.stats = self.get_normalization_stats()

    def create_index_mapping(self):
        """Create an index mapping to efficiently sample from episodes"""
        self.episode_indices = []
        self.total_samples = 0

        for file_idx, file_path in enumerate(self.file_paths):
            with h5py.File(file_path, 'r') as f:
                # Get episode length
                EPS = 1e-2
                qpos = f['ee_poses']
                qpos_length = qpos.shape[0]

                qpos_delta = np.abs(qpos - qpos[0:1])
                indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]

                if len(indices) == 0:
                    print(f"Warning: No movement detected in file {file_path}. Skipping.")
                    continue

                # Only use sequences where we have enough frames for context and prediction
                valid_start_indices = range(indices[0], qpos_length - (self.context_frames + self.horizon - 1), self.stride)

                for start_idx in valid_start_indices:
                    self.episode_indices.append((file_idx, start_idx))
                    self.total_samples += 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Get a training sample at the specified index.

        Returns a dictionary containing:
            states: Robot states for the context frames [context_frames, state_dim]
            vla_actions: VLA actions for the horizon [horizon, state_dim]
            expert_actions: Expert actions (ground truth) for the horizon [horizon, state_dim]
            images_cam1: Front camera images for context frames [context_frames, H, W, 3]
            images_cam2: Right camera images for context frames [context_frames, H, W, 3]
        """
        # Map the index to episode and start index
        file_idx, start_idx = self.episode_indices[idx]
        file_path = self.file_paths[file_idx]

        # Load data from the h5 file
        with h5py.File(file_path, 'r') as f:

            # convert to 10D representation
            qpos = converted_ee_pose_with_gripper(f)
            qpos = qpos[start_idx:start_idx + self.context_frames + self.horizon]

            future_states = qpos[self.context_frames:]
            future_states[:,-1]/=255 # only rescale actions, not observations

            # Get VLA actions for the current state
            # Note: Each timestep in vla_action contains predictions for multiple future steps
            vla_action_chunk = f['vla_action'][start_idx + self.context_frames]
            vla_actions = vla_action_chunk[:self.horizon]
            vla_actions[:,-1]/=255

            forces = f['gelsight_force']['forces'][start_idx:start_idx + self.context_frames + self.horizon]
            displacements = f['gelsight_force']['displacement'][start_idx:start_idx + self.context_frames + self.horizon]

            # Load images if requested
            images_cam1 = None
            images_cam2 = None

            if self.use_images:

                img_front = f['camera1_resized'][start_idx:start_idx + self.context_frames]
                img_right = f['camera2_resized'][start_idx:start_idx + self.context_frames]

                # Convert to numpy arrays
                images_cam1 = np.array(img_front)
                images_cam2 = np.array(img_right)

        # Convert to tensors
        qpos = torch.as_tensor(qpos, dtype=torch.float32)
        future_states = torch.as_tensor(future_states, dtype=torch.float32)
        vla_actions = torch.as_tensor(vla_actions, dtype=torch.float32)

        forces = torch.as_tensor(forces, dtype=torch.float32)
        disps = torch.as_tensor(displacements, dtype=torch.float32)



        result = {
            'states': qpos,
            'vla_actions': vla_actions,
            'expert_actions': future_states,
            'forces':forces,
            'disps':disps,
        }

        if self.use_images:
            result['images_cam1'] = torch.as_tensor(images_cam1, dtype=torch.float32) / 255.0
            result['images_cam2'] = torch.as_tensor(images_cam2, dtype=torch.float32) / 255.0

        return result

    def get_normalization_stats(self):
        """
        Calculate statistics for normalizing actions to [-1, 1] range using all trajectories.
        This method computes min/max values across all data points to create a proper normalization
        scale for both VLA actions and expert actions (future states).

        Returns:
            dict: Dictionary containing normalization statistics:
                - action_mins: Minimum values for each action dimension
                - action_maxs: Maximum values for each action dimension
                - vla_mins: Minimum values for each VLA action dimension
                - vla_maxs: Maximum values for each VLA action dimension
                - action_range: Range for each action dimension
                - vla_range: Range for each VLA action dimension
        """
        # Initialize min/max values for both expert actions and VLA actions
        # We're focusing on the position/rotation components (first dimensions)
        action_dims = 10  # Assuming full dimensionality (pos, rot, gripper)
        action_mins = np.array([float('inf')] * action_dims)
        action_maxs = np.array([float('-inf')] * action_dims)

        vla_mins = np.array([float('inf')] * action_dims)
        vla_maxs = np.array([float('-inf')] * action_dims)

        print(f"Computing normalization statistics across {len(self.file_paths)} files...")

        for file_idx, file_path in enumerate(self.file_paths):
            with h5py.File(file_path, 'r') as f:

                expert_actions = converted_ee_pose_with_gripper(f)
                expert_actions[:,-1] /=255
                vla_actions = f['vla_action'][:]
                vla_actions[:,:,-1]/=255

                # Update min/max for expert actions
                action_mins = np.minimum(action_mins, np.min(expert_actions, axis=0))
                action_maxs = np.maximum(action_maxs, np.max(expert_actions, axis=0))

                # Update min/max for VLA actions - apply min/max across both time and batch dims
                vla_mins = np.minimum(vla_mins, np.min(vla_actions, axis=(0, 1)))
                vla_maxs = np.maximum(vla_maxs, np.max(vla_actions, axis=(0, 1)))

        # Add small epsilon to avoid division by zero for dimensions with no range
        eps = 1e-6
        action_range = action_maxs - action_mins
        action_range[action_range < eps] = 1.0

        vla_range = vla_maxs - vla_mins
        vla_range[vla_range < eps] = 1.0

        stats = {
            'action_mins': action_mins,
            'action_maxs': action_maxs,
            'vla_mins': vla_mins,
            'vla_maxs': vla_maxs,
            'action_range': action_range,
            'vla_range': vla_range
        }

        # print("Normalization statistics computed:")
        # print(f"Action mins: {action_mins}")
        # print(f"Action maxs: {action_maxs}")
        # print(f"Action range: {action_range}")

        return stats


# def normalize_actions(actions, stats, action_type='expert'):
#     """
#     Normalize actions to [-1, 1] range based on precomputed statistics.
#     Args:
#         actions: Actions to normalize
#         stats: Statistics from get_normalization_stats
#         action_type: Either 'expert' or 'vla' (default: 'expert')
#     Returns:
#         Normalized actions in range [-1, 1]
#     """
#     # Select the appropriate min/max values based on action type
#     if action_type == 'expert':
#         mins = stats['action_mins']
#         # maxs = stats['action_maxs']
#         range_values = stats['action_range']
#     elif action_type == 'vla':
#         mins = stats['vla_mins']
#         # maxs = stats['vla_maxs']
#         range_values = stats['vla_range']
#     else:
#         raise ValueError(f"Unknown action_type: {action_type}. Use 'expert' or 'vla'.")
#
#     # Normalize to [0, 1] first, using the range to avoid division by zero
#     normalized = (actions - mins) / range_values
#
#     # Then scale to [-1, 1]
#     normalized = normalized * 2.0 - 1.0
#
#     return normalized
#
#
# def denormalize_actions(normalized_actions, stats, action_type='expert'):
#     """
#     Convert normalized actions back to original scale.
#
#     Args:
#         normalized_actions: Normalized actions in [-1, 1] range
#         stats: Statistics from get_normalization_stats
#         action_type: Either 'expert' or 'vla'
#
#     Returns:
#         Actions in original scale
#     """
#     if action_type == 'expert':
#         mins = stats['action_mins']
#         # maxs = stats['action_maxs']
#         range_values = stats['action_range']
#     elif action_type == 'vla':
#         mins = stats['vla_mins']
#         # maxs = stats['vla_maxs']
#         range_values = stats['vla_range']
#     else:
#         raise ValueError(f"Unknown action_type: {action_type}. Use 'expert' or 'vla'.")
#
#
#     # First convert from [-1, 1] to [0, 1]
#     unnormalized = (normalized_actions + 1.0) / 2.0
#
#     # Then scale back to original range
#     unnormalized = unnormalized * range_values + mins
#
#     return unnormalized


def normalize_actions(actions, stats, action_type='expert', padding_factor=1.4):
    """
    Normalize actions with padded min/max range to handle out-of-distribution data.

    Args:
        actions: Actions to normalize
        stats: Statistics from get_normalization_stats
        action_type: Either 'expert' or 'vla'
        padding_factor: Factor to expand range by (e.g., 1.2 adds 20% padding)

    Returns:
        Normalized actions guaranteed to be within [-1, 1] for in-range data,
        and likely to be within [-1, 1] for reasonable OOD data
    """
    if action_type == 'expert':
        mins = stats['action_mins']
        maxs = stats['action_maxs']
    elif action_type == 'vla':
        mins = stats['vla_mins']
        maxs = stats['vla_maxs']
    else:
        raise ValueError(f"Unknown action_type: {action_type}")

    # Calculate range
    orig_range = maxs - mins

    # Calculate padded mins and maxs
    # For each dimension, expand range by padding_factor
    padded_range = orig_range * padding_factor

    # Center the padded range around the original range center
    center = (mins + maxs) / 2
    padded_mins = center - padded_range / 2
    padded_maxs = center + padded_range / 2

    # Ensure no division by zero
    eps = 1e-6
    safe_range = padded_maxs - padded_mins
    safe_range[safe_range < eps] = 1.0

    # Normalize to [-1, 1] using padded range
    normalized = 2.0 * (actions - padded_mins) / safe_range - 1.0

    return normalized


def denormalize_actions(normalized_actions, stats, action_type='expert', padding_factor=1.4):
    """
    Denormalize actions that were normalized with padded_normalize_actions.

    Args:
        normalized_actions: Normalized actions from padded_normalize_actions
        stats: Statistics from get_normalization_stats
        action_type: Either 'expert' or 'vla'
        padding_factor: Same factor used in normalization

    Returns:
        Actions in original scale
    """
    if action_type == 'expert':
        mins = stats['action_mins']
        maxs = stats['action_maxs']
    elif action_type == 'vla':
        mins = stats['vla_mins']
        maxs = stats['vla_maxs']
    else:
        raise ValueError(f"Unknown action_type: {action_type}")

    # Calculate range
    orig_range = maxs - mins

    # Calculate padded mins and maxs (must match normalization)
    padded_range = orig_range * padding_factor
    center = (mins + maxs) / 2
    padded_mins = center - padded_range / 2
    padded_maxs = center + padded_range / 2

    # Denormalize from [-1, 1]
    safe_range = padded_maxs - padded_mins
    unnormalized = (normalized_actions + 1.0) / 2.0 * safe_range + padded_mins

    return unnormalized

class ControllerDataModule:
    """Data module for training, validation, and testing"""

    def __init__(
            self,
            data_dir,
            batch_size=32,
            num_workers=4,
            context_frames=2,
            horizon=8,
            use_images=True,
            image_size=384,
            val_ratio=0.1,
            stride=1
    ):
        """
        Initialize the data module.

        Args:
            data_dir: Directory containing the dataset
            batch_size: Batch size
            num_workers: Number of workers for data loading
            context_frames: Number of past frames to use as context
            horizon: Number of future steps to predict
            use_images: Whether to include images
            image_size: Size of images
            val_ratio: Ratio of data to use for validation
            stride: Stride for sampling sequences
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.context_frames = context_frames
        self.horizon = horizon
        self.use_images = use_images
        self.image_size = image_size
        self.val_ratio = val_ratio
        self.stride = stride

        # Split files into train and validation sets
        self.setup()

    def setup(self):
        """Set up the data module"""
        # Find all h5 files

        print(f"loading dataset from {self.data_dir} ..")
        file_paths = []
        for root, _, files in os.walk(self.data_dir):
            for filename in natural_sort_filenames(fnmatch.filter(files, '*.h5')):
                file_paths.append(os.path.join(root, filename))

        # Split files into train and validation sets
        num_val = max(1, int(len(file_paths) * self.val_ratio))
        val_indices = np.random.choice(len(file_paths), num_val, replace=False)

        train_files = [file_paths[i] for i in range(len(file_paths)) if i not in val_indices]
        val_files = [file_paths[i] for i in val_indices]

        # Create train and validation datasets
        self.train_dataset = ControllerDataset(
            data_dir=self.data_dir,
            context_frames=self.context_frames,
            horizon=self.horizon,
            use_images=self.use_images,
            image_size=self.image_size,
            stride=self.stride
        )
        self.train_dataset.file_paths = train_files
        self.train_dataset.create_index_mapping()

        self.val_dataset = ControllerDataset(
            data_dir=self.data_dir,
            context_frames=self.context_frames,
            horizon=self.horizon,
            use_images=self.use_images,
            image_size=self.image_size,
            stride=self.stride
        )
        self.val_dataset.file_paths = val_files
        self.val_dataset.create_index_mapping()

        # Calculate normalization statistics
        self.stats = self.train_dataset.get_normalization_stats()

    def train_dataloader(self):
        """Get the training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        """Get the validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )



if __name__ == "__main__":
    # Example usage
    data_dir = "./data/datasets/watercup_controller"

    # Create a dataset
    dataset = ControllerDataset(
        data_dir=data_dir,
        context_frames=2,
        horizon=64,
        use_images=True,
        stride=1
    )

    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    # Create a data module
    data_module = ControllerDataModule(
        data_dir=data_dir,
        batch_size=16,
        num_workers=4
    )

    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")

    # Check a batch
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        for key, value in batch.items():
            print(f"  {key}: shape={value.shape}")

        if batch_idx >= 0:  # Just check the first batch
            break