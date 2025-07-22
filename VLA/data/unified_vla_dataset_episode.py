import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING
from docs.test_6drot import convert_quaternion_to_orthod6d
import re


def natural_sort_filenames(file_list):
    """
    Sort filenames naturally, so that episode_2.h5 comes before episode_10.h5

    Args:
        file_list: List of filenames to sort

    Returns:
        Sorted list of filenames
    """

    def extract_number(filename):
        # Extract the numeric part from the filename
        match = re.search(r'episode_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0  # Default if no number found

    # Sort based on the extracted number
    return sorted(file_list, key=extract_number)


def pad_and_resize_for_siglip(images, target_size=384):
    """
    First pad images to make them square, then resize to the target size.
    This approach preserves more information from the original image.

    Parameters:
    - images: numpy array of shape (batch, height, width, channels)
    - target_size: the target size for SigLIP (default: 384)

    Returns:
    - Processed images of shape (batch, target_size, target_size, channels)
    """
    batch_size, height, width, channels = images.shape
    processed_images = np.zeros((batch_size, target_size, target_size, channels), dtype=images.dtype)

    for i in range(batch_size):
        img = images[i]

        # Find the larger dimension to create a square
        max_dim = max(height, width)

        # Create a square canvas filled with zeros
        square_img = np.zeros((max_dim, max_dim, channels), dtype=img.dtype)

        # Calculate padding to center the image
        pad_height = (max_dim - height) // 2
        pad_width = (max_dim - width) // 2

        # Place the original image in the center of the square canvas
        square_img[pad_height:pad_height + height, pad_width:pad_width + width, :] = img

        # Resize the square image to the target size
        resized_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        processed_images[i] = resized_img

    return processed_images


def converted_ee_pose_with_gripper(data_dict):
    """
    Convert end-effector pose with gripper.
    Works with both HDF5 and NPZ data.

    Args:
        data_dict: Dictionary containing 'ee_poses' and 'gripper_pos'

    Returns:
        qpos: Combined pose and gripper data
    """
    # [ee_pos, ee_6d_ori, gripper_open] #(batch, 10)
    ee_pos = data_dict['ee_poses'][:, :3]
    ee_ori = data_dict['ee_poses'][:, 3:]
    ee_6d = convert_quaternion_to_orthod6d(ee_ori)

    grip_pos = data_dict['gripper_pos'][:].reshape(-1, 1)

    qpos = np.concatenate((ee_pos, ee_6d, grip_pos), axis=-1)

    return qpos


class UnifiedDataInterface:
    """
    Base class for unified data access between HDF5 and NPZ.
    Provides methods to read data from both formats.
    """

    @staticmethod
    def get_file_extension(file_path):
        """Get the file extension of a file path"""
        _, ext = os.path.splitext(file_path)
        return ext.lower()

    @staticmethod
    def is_hdf5_file(file_path):
        """Check if a file is an HDF5 file"""
        return UnifiedDataInterface.get_file_extension(file_path) == '.h5'

    @staticmethod
    def is_npz_file(file_path):
        """Check if a file is a NPZ file"""
        return UnifiedDataInterface.get_file_extension(file_path) == '.npz'

    @staticmethod
    def open_file(file_path):
        """Open a file based on its extension"""
        if UnifiedDataInterface.is_hdf5_file(file_path):
            return h5py.File(file_path, 'r')
        elif UnifiedDataInterface.is_npz_file(file_path):
            return np.load(file_path, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    @staticmethod
    def get_item(data, key, default=None):
        """Get an item from either HDF5 or NPZ data"""
        try:
            if isinstance(data, h5py.File):
                # HDF5 data
                return data[key]
            else:
                # NPZ data
                if key in data:
                    return data[key]
                elif f"{key}_images" in data:  # Special handling for image data in NPZ
                    return data[f"{key}_images"]
                else:
                    return default
        except KeyError:
            return default

    @staticmethod
    def get_metadata(data):
        """Get metadata from either HDF5 or NPZ data"""
        if isinstance(data, h5py.File):
            # For HDF5, metadata is stored as attributes
            metadata = {}
            for key in data.attrs:
                metadata[key] = data.attrs[key]
            return metadata
        else:
            # For NPZ, metadata is stored as a JSON string in the 'metadata' key
            if 'metadata' in data:
                try:
                    return json.loads(data['metadata'].item())
                except:
                    return {}
            return {}


class UnifiedVLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in either HDF5 or NPZ format.
    """

    def __init__(self) -> None:
        # [Modify] The path to the dataset directories
        # Each file contains one episode
        self.DATASET_NAME = "mango"
        HDF5_DIR = f"data/datasets/{self.DATASET_NAME}_hdf5_gelsight/"
        NPZ_DIR = None #"data/datasets/mahjong_stack_npz/"

        self.file_paths = []
        # Collect HDF5 files
        if HDF5_DIR:
            if os.path.exists(HDF5_DIR):
                for root, _, files in os.walk(HDF5_DIR):
                    filename_list = natural_sort_filenames(fnmatch.filter(files, '*.h5'))
                    for filename in filename_list:
                        file_path = os.path.join(root, filename)
                        self.file_paths.append(file_path)
        if NPZ_DIR:
            # Collect NPZ files
            if os.path.exists(NPZ_DIR):
                for root, _, files in os.walk(NPZ_DIR):
                    filename_list = natural_sort_filenames(fnmatch.filter(files, '*.npz'))
                    for filename in filename_list:
                        file_path = os.path.join(root, filename)
                        self.file_paths.append(file_path)

        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            _, epi_len = self.parse_file_state_only(file_path)
            if epi_len is not None:
                episode_lens.append(epi_len)

        self.total_episode_lengths = np.sum(episode_lens)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

    def __len__(self):
        return len(self.episode_sample_weights)

    def get_totol_episode_lengths(self):
        return self.total_episode_lengths

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            sample, _ = self.parse_file(file_path) \
                if not state_only else self.parse_file_state_only(file_path)
            if sample:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))

    def parse_file(self, file_path):
        """Parse a file (HDF5 or NPZ) to generate a training sample at a random timestep.

        Args:
            file_path (str): the path to the file

        Returns:
            dict: a dictionary containing the training sample, or None if the episode is invalid.
            int: number of steps in the episode, or None if the episode is invalid.
        """
        data_interface = UnifiedDataInterface()

        with data_interface.open_file(file_path) as episode_data:
            is_hdf5 = isinstance(episode_data, h5py.File)

            # Extract data for different file types
            if is_hdf5:
                # HDF5 data extraction
                qpos = converted_ee_pose_with_gripper(episode_data)

                # Load instruction embedding
                instruction_embedding = episode_data['instruct_embeddings'][:][0]
            else:
                # NPZ data extraction
                # For NPZ, we need to construct a dict that mimics the HDF5 structure
                data_dict = {
                    'ee_poses': episode_data['ee_poses'] if 'ee_poses' in episode_data else None,
                    'gripper_pos': episode_data['gripper_pos'] if 'gripper_pos' in episode_data else None
                }

                # If we're missing required data, try to find it with alternative naming
                if data_dict['ee_poses'] is None and 'ee_poses_data' in episode_data:
                    data_dict['ee_poses'] = episode_data['ee_poses_data']
                if data_dict['gripper_pos'] is None and 'gripper_pos_data' in episode_data:
                    data_dict['gripper_pos'] = episode_data['gripper_pos_data']

                # Check if we have all required data
                if data_dict['ee_poses'] is None or data_dict['gripper_pos'] is None:
                    print(f"Missing required data in NPZ file: {file_path}")
                    return None, None

                qpos = converted_ee_pose_with_gripper(data_dict)

                # Load instruction embedding
                instruction_embedding = episode_data['instruct_embeddings'][
                    0] if 'instruct_embeddings' in episode_data else np.zeros(768)

            num_steps = qpos.shape[0]

            # [Optional] We drop too-short episodes
            if num_steps < 32:
                return None, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                print(f"Found no qpos that exceeds the threshold in {file_path}")
                return None, None

            # We randomly sample a timestep
            step_id = np.random.randint(first_idx - 1, num_steps - int(self.CHUNK_SIZE / 2))
            action_id = step_id + 2

            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction_embedding": instruction_embedding
            }

            # Rescale gripper to [0, 1]
            qpos = qpos / np.array(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 255]]
            )

            target_qpos = qpos[action_id:action_id + self.CHUNK_SIZE]

            # Parse the state and action
            state = qpos[step_id:step_id + 1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos ** 2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1))
                ], axis=0)

            state = self.fill_in_state(state)
            state_indicator = self.fill_in_state(np.ones_like(state_std))
            state_std = self.fill_in_state(state_std)
            state_mean = self.fill_in_state(state_mean)
            state_norm = self.fill_in_state(state_norm)
            actions = self.fill_in_state(actions)

            # Parse images
            cam_high = self.parse_img('camera1', step_id, episode_data, is_hdf5)
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = self.parse_img('camera2', step_id, episode_data, is_hdf5)
            cam_right_wrist_mask = cam_high_mask.copy()

            # Return the resulting sample
            return {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }, num_steps

    def parse_img(self, key, step_id, data, is_hdf5):
        """Parse images from either HDF5 or NPZ data"""
        if is_hdf5:
            # HDF5 image parsing
            if key in data:
                imgs = data[key][key][max(step_id - self.IMG_HISORY_SIZE + 1, 0): step_id + 1]
                imgs = pad_and_resize_for_siglip(imgs)
            else:
                return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
        else:
            # NPZ image parsing
            img_key = f"{key}_images"
            if img_key in data:
                imgs = data[img_key][max(step_id - self.IMG_HISORY_SIZE + 1, 0): step_id + 1]
                imgs = pad_and_resize_for_siglip(imgs)
            else:
                return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))

        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            # Pad the images using the first image
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)
        return imgs

    def parse_file_state_only(self, file_path):
        """Parse a file (HDF5 or NPZ) to generate a state trajectory.

        Args:
            file_path (str): the path to the file

        Returns:
            dict: a dictionary containing the state trajectory, or None if the episode is invalid.
            int: number of steps in the episode, or None if the episode is invalid.
        """
        data_interface = UnifiedDataInterface()

        with data_interface.open_file(file_path) as data:
            state, epi_len = self.get_episodic_data_state_only(data)
            return state, epi_len

    def get_episodic_data_state_only(self, data):
        """Extract state-only data from either HDF5 or NPZ format"""
        is_hdf5 = isinstance(data, h5py.File)

        if is_hdf5:
            # HDF5 data extraction
            qpos = converted_ee_pose_with_gripper(data)
        else:
            # NPZ data extraction
            # For NPZ, we need to construct a dict that mimics the HDF5 structure
            data_dict = {
                'ee_poses': data['ee_poses'] if 'ee_poses' in data else None,
                'gripper_pos': data['gripper_pos'] if 'gripper_pos' in data else None
            }

            # If we're missing required data, try to find it with alternative naming
            if data_dict['ee_poses'] is None and 'ee_poses_data' in data:
                data_dict['ee_poses'] = data['ee_poses_data']
            if data_dict['gripper_pos'] is None and 'gripper_pos_data' in data:
                data_dict['gripper_pos'] = data['gripper_pos_data']

            # Check if we have all required data
            if data_dict['ee_poses'] is None or data_dict['gripper_pos'] is None:
                return None, 0

            qpos = converted_ee_pose_with_gripper(data_dict)

        num_steps = qpos.shape[0]

        if num_steps < 32:
            print(f"drop short episode with {num_steps} steps < 32 steps ")
            return None, 0

        # [Optional] We skip the first few still steps
        EPS = 1e-2
        # Get the idx of the first qpos whose delta exceeds the threshold
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            print("Found no qpos that exceeds the threshold.")
            return None, 0

        # Rescale gripper to [0, 1]
        qpos = qpos / np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 255]]
        )

        # Parse the state
        state = qpos[first_idx - 1:]
        state = self.fill_in_state(state)

        return {"state": state}, len(state)

    # Fill the state/action into the unified vector
    def fill_in_state(self, values):
        UNI_STATE_INDICES = [
                                STATE_VEC_IDX_MAPPING["eef_pos_x"]
                            ] + [
                                STATE_VEC_IDX_MAPPING["eef_pos_y"]
                            ] + [
                                STATE_VEC_IDX_MAPPING["eef_pos_z"]
                            ] + [
                                STATE_VEC_IDX_MAPPING[f"eef_angle_{i}"] for i in range(6)
                            ] + [
                                STATE_VEC_IDX_MAPPING["right_gripper_open"]
                            ]

        uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
        uni_vec[..., UNI_STATE_INDICES] = values
        return uni_vec


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Test the dataset
    ds = UnifiedVLADataset()
    print(f"Total episodes: {ds.__len__()}")

    # Test with a random episode
    random_idx = np.random.randint(0, len(ds))
    print(f"Testing with random episode {random_idx}...")
    data = ds.get_item(random_idx)
    print(f"Sample State: {data['state'][0]}")

    img = data["cam_high"][-1] #cam_right_wrist

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Camera High RGB Image Sample")
    plt.axis('off')
    plt.show()