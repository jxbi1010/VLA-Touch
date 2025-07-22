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


def converted_ee_pose_with_gripper(epi_data):
    # [ee_pos, ee_6d_ori, gripper_open] #(batch, 10)
    ee_pos = epi_data['ee_poses'][:, :3]
    ee_ori = epi_data['ee_poses'][:, 3:]
    ee_6d = convert_quaternion_to_orthod6d(ee_ori)

    grip_pos = epi_data['gripper_pos'][:].reshape(-1, 1)

    qpos = np.concatenate((ee_pos, ee_6d, grip_pos), axis=-1)

    return qpos

class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        # HDF5_DIR = "data/datasets/openx_embod/berkeley_rpt_converted_externally_to_rlds/0.1.0/"
        HDF5_DIR = "data/datasets/mahjong_stack_hdf5/"
        self.DATASET_NAME = "mahjong_stack"
        
        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            filename_list = natural_sort_filenames(fnmatch.filter(files, '*.h5'))
            for filename in filename_list:
                file_path = os.path.join(root, filename)
                # print(f"file_path:{file_path}")
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
            _, epi_len = self.parse_hdf5_file_state_only(file_path)
            episode_lens.append(epi_len)

        self.total_episode_lengths = np.sum(episode_lens)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.episode_sample_weights)

    def get_totol_episode_lengths(self):
        return self.total_episode_lengths

    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
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
                # print(f"file_paths:{self.file_paths}")
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            sample,_ = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if sample:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))

    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as episode:

            qpos = converted_ee_pose_with_gripper(episode)
            num_steps = qpos.shape[0]

            # [Optional] We drop too-short episode
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
                raise ValueError("Found no qpos that exceeds the threshold.")

            # We randomly sample a timestep
            step_id = np.random.randint(first_idx - 1, num_steps - int(self.CHUNK_SIZE/2))
            action_id = step_id +2

            # Load the corresponding insturction embedding from pre-computed
            instruction_embedding = episode['instruct_embeddings'][:][0]

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

            # def rescale_qpos_with_gripper(qpos):
            #     eef = qpos[:,-1] / np.array(
            #         [[1, 1, 1, 1, 1, 1, 1, 1, 1]]
            #     )
            #     gripper = (qpos[:,-1]-80)/150.0
            #
            #     return np.concatenate((eef,gripper),axis=-1)
            #
            # qpos = rescale_qpos_with_gripper(qpos)


            target_qpos = qpos[action_id:action_id + self.CHUNK_SIZE] / np.array(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 255]]
            )

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
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = self.fill_in_state(actions)

            # `cam_high` is the external camera image
            cam_high = self.parse_img('camera1',step_id, episode)
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = self.parse_img('camera2',step_id, episode)
            cam_right_wrist_mask = cam_high_mask.copy()

            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
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

    def parse_img(self, key, step_id, episode):
        imgs = episode[key][key][max(step_id - self.IMG_HISORY_SIZE + 1, 0): step_id + 1]
        imgs = pad_and_resize_for_siglip(imgs)

        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            # Pad the images using the first image
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)
        return imgs

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            state, epi_len = self.get_episodic_data_state_only(f)

        return state, epi_len

    def get_episodic_data_state_only(self, epi_data):

        qpos = converted_ee_pose_with_gripper(epi_data)
        num_steps = qpos.shape[0]

        if num_steps < 32:
            print(f"drop shot episode with {num_steps} steps < 32 steps ")
            return None, 0

        # [Optional] We skip the first few still steps
        EPS = 1e-2
        # Get the idx of the first qpos whose delta exceeds the threshold
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.")

        # Rescale gripper to [0, 1]
        qpos = qpos / np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1,255]]
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
    ds = HDF5VLADataset()
    print(ds.__len__())
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        data = ds.get_item(i)
        print(data)
