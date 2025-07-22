#!/usr/bin/env python
# coding: UTF-8
"""
Script to record VLA actions for a dataset by running the VLA model on each observation.
This will augment the dataset with VLA_action field to be used for controller training.
"""

import os
import sys
import argparse
import h5py
import yaml
import numpy as np
import torch
import cv2
from tqdm import tqdm
from collections import deque
from PIL import Image as PImage

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.franka_model_eef import create_model
from scripts.utils_eef import *
from data.unified_vla_dataset_episode import UnifiedVLADataset as HDF5VLADataset

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']


def make_policy(args):
    """Initialize the VLA model"""
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=10,  # Default frequency
    )
    return model


def get_config(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


def update_observation_window(observation_window, img_front, img_right, qpos, device='cuda'):
    """Update the global observation window with new images and state"""

    # JPEG transformation to align with training
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    # Process the images
    img_front = jpeg_mapping(img_front)
    img_right = jpeg_mapping(img_right)
    qpos_tensor = torch.from_numpy(qpos).float().to(device)

    # Append the observation to the window
    observation_window.append(
        {
            'qpos': qpos_tensor,
            'images': {
                CAMERA_NAMES[0]: img_front,
                CAMERA_NAMES[1]: img_right,
                CAMERA_NAMES[2]: None,
            },
        }
    )

    return observation_window


def init_observation_window():
    """Initialize a new observation window at the start of an episode"""
    observation_window = deque(maxlen=2)

    # Add the first dummy frame
    observation_window.append(
        {
            'qpos': None,
            'images': {
                CAMERA_NAMES[0]: None,
                CAMERA_NAMES[1]: None,
                CAMERA_NAMES[2]: None,
            },
        }
    )

    return observation_window


def run_vla_inference(policy, observation_window, lang_embeddings, camera_names):
    """Run VLA model inference on the observation"""
    # Fetch images in sequence [front, right]
    image_arrs = [
        observation_window[-2]['images'][camera_names[0]],
        observation_window[-2]['images'][camera_names[1]],
        observation_window[-2]['images'][camera_names[2]],
        observation_window[-1]['images'][camera_names[0]],
        observation_window[-1]['images'][camera_names[1]],
        observation_window[-1]['images'][camera_names[2]]
    ]

    images = [PImage.fromarray(arr) if arr is not None else None
              for arr in image_arrs]

    # Get last qpos
    proprio = observation_window[-1]['qpos']
    proprio = proprio.unsqueeze(0)

    # Get VLA actions
    with torch.inference_mode():
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings
        ).squeeze(0).cpu().numpy()

    return actions


def record_vla_actions_for_dataset(args):
    """Main function to record VLA actions for the dataset"""
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize VLA model
    policy = make_policy(args)

    # Initialize the dataset
    dataset = HDF5VLADataset()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each episode in the dataset
    for idx in range(105,len(dataset)):
        print(f"Processing episode {idx + 1}/{len(dataset)}...")

        # Get the original file path
        file_path = dataset.file_paths[idx]
        file_name = os.path.basename(file_path)
        output_path = os.path.join(args.output_dir, file_name)

        # Skip if output file already exists
        if os.path.exists(output_path):
            print(f"File {output_path} already exists, skipping...")
            continue

        # Initialize a new observation window for this episode
        observation_window = init_observation_window()

        # Open input file
        with h5py.File(file_path, 'r') as input_file:
            # Create output file with same structure
            with h5py.File(output_path, 'w') as output_file:
                # Copy all original datasets
                for key in input_file.keys():
                    if isinstance(input_file[key], h5py.Group):
                        group = output_file.create_group(key)
                        for subkey in input_file[key].keys():
                            input_file.copy(f"{key}/{subkey}", group)
                    else:
                        input_file.copy(key, output_file)

                # Get the timesteps and images
                qpos = converted_ee_pose_with_gripper(input_file)
                num_steps = qpos.shape[0]
                lang_embeddings = torch.Tensor(input_file['instruct_embeddings'][:])

                # Create VLA actions dataset
                vla_actions = output_file.create_dataset('vla_action',
                                                         shape=(num_steps, args.chunk_size, 10),
                                                         dtype=np.float32)

                camera1_resized = output_file.create_dataset('camera1_resized',
                                                         shape=(num_steps, 384,384,3),
                                                         dtype=np.uint8)
                camera2_resized = output_file.create_dataset('camera2_resized',
                                                         shape=(num_steps, 384,384,3),
                                                         dtype=np.uint8)

                # Process each timestep in sequence, maintaining the observation window
                for t in tqdm(range(0, num_steps),
                              desc=f"Processing episode {idx + 1}"):
                    # Get images for this timestep
                    img_front = input_file['camera1']['camera1'][t]
                    img_right = input_file['camera2']['camera2'][t]

                    # Resize for SigLIP
                    img_front = pad_and_resize_for_siglip(img_front)
                    img_right = pad_and_resize_for_siglip(img_right)

                    # Update the observation window with the current frame
                    observation_window = update_observation_window(
                        observation_window, img_front, img_right, qpos[t], device=device
                    )

                    # Run VLA inference only when we have a complete window (2 frames)
                    actions = run_vla_inference(policy, observation_window, lang_embeddings, CAMERA_NAMES)

                    vla_actions[t] = actions
                    camera1_resized[t] = img_front
                    camera2_resized[t] = img_right

        print(f"Successfully processed episode {idx + 1} and saved to {output_path}")


if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Record VLA actions for the dataset')
        parser.add_argument('--config_path', type=str, default="./configs/base.yaml",
                            help='Path to the config file')
        parser.add_argument('--pretrained_model_name_or_path', type=str,
                            default="./checkpoints/rdt_ckpt/rdt-finetune-1b-mango-rgb/checkpoint-40000",
                            help='Path to the pretrained VLA model')
        parser.add_argument('--output_dir', type=str, default="./data/datasets/mango_controller_40000",
                            help='Directory to save the augmented dataset')
        parser.add_argument('--use_gpu', action='store_true', default=True,
                            help='Use GPU for inference')
        parser.add_argument('--chunk_size', type=int, default=64,
                            help='Action chunk size')
        return parser.parse_args()

    args = parse_arguments()
    record_vla_actions_for_dataset(args)