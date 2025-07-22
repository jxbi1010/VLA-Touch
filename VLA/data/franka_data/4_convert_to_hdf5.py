import os
import h5py
import numpy as np
import json
from pathlib import Path
import cv2
import re
import torch

def get_file_number(filename):
    # Extract the number between "rgb_" and ".jpg"
    match = re.search(r'rgb_(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return 0


def convert_episode_to_hdf5(episode_path, output_path):
    """
    Convert a single episode's data to HDF5 format, handling both images and numpy files

    Args:
        episode_path (str): Path to the episode directory
        output_path (str): Path where the HDF5 file will be saved
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create HDF5 file
    with h5py.File(output_path, 'w') as hf:
        # Iterate through all items in the episode directory
        for item_name in os.listdir(episode_path):
            item_path = os.path.join(episode_path, item_name)

            # Handle numpy files directly in the episode directory
            if os.path.isfile(item_path) and item_path.endswith('.npy'):
                # Load numpy data
                data = np.load(item_path,allow_pickle=True)
                # Create dataset with the base filename (without extension)
                base_name = os.path.splitext(item_name)[0]
                hf.create_dataset(
                    base_name,
                    data=data,
                    compression='lzf'
                )
                print(f"  Saved numpy file: {item_name}, with data size {data.shape} ")
                continue

            # handle .pt file, convert to numpy and save
            elif os.path.isfile(item_path) and item_path.endswith('.pt'):
                embedding_tensor = torch.load(item_path)
                embedding_np = embedding_tensor.to(torch.float32).numpy()

                hf.create_dataset(
                    "instruct_embeddings",
                    data=embedding_np,
                    compression='lzf'
                )

                print(f"  Saved pt file: {item_name}, with data size {embedding_tensor.shape} ")
                continue


            # Skip non-directories and non-npy files
            if not os.path.isdir(item_path):
                continue

            # Create a group for each subfolder (sensor type)
            sensor_group = hf.create_group(item_name)

            # Get all files in the sensor directory
            files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]

            # Process image files
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_files.sort(key=get_file_number)

            if image_files:
                all_images = []

                # First attempt: Try to collect all images as-is
                try:
                    for file_name in image_files:
                        file_path = os.path.join(item_path, file_name)
                        img = cv2.imread(file_path)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            all_images.append(img_rgb)

                    # Try to convert to numpy array
                    if all_images:
                        images_array = np.array(all_images)

                        # Create a single dataset with all images
                        sensor_group.create_dataset(
                            item_name,
                            data=images_array,
                            compression='lzf'
                        )
                        print(f"  Saved {len(all_images)} images as a unified array from {item_name}, with data size {images_array.shape}")

                except ValueError as e:
                    # If we get here, images have different dimensions
                    print(f"  Images in {item_name} have different dimensions. Resizing to smallest resolution...")

                    # Find the smallest dimension among all images
                    min_height = float('inf')
                    min_width = float('inf')

                    # First pass to determine minimum dimensions
                    for file_name in image_files:
                        file_path = os.path.join(item_path, file_name)
                        img = cv2.imread(file_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            min_height = min(min_height, h)
                            min_width = min(min_width, w)

                    # Second pass to resize and collect images
                    resized_images = []
                    for file_name in image_files:
                        file_path = os.path.join(item_path, file_name)
                        img = cv2.imread(file_path)
                        if img is not None:
                            resized = cv2.resize(img, (min_width, min_height), interpolation=cv2.INTER_AREA)
                            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            resized_images.append(resized_rgb)

                    # Convert resized images to array and save
                    if resized_images:
                        resized_array = np.array(resized_images)
                        sensor_group.create_dataset(
                            item_name,
                            data=resized_array,
                            compression='lzf'
                        )

                        # Store the resize information as attributes
                        sensor_group.attrs['resized'] = True
                        sensor_group.attrs['target_width'] = min_width
                        sensor_group.attrs['target_height'] = min_height

                        print(
                            f"  Saved {len(resized_images)} resized images ({min_width}x{min_height}) from {item_name}")

            # Process numpy files in subfolders
            npy_files = [f for f in files if f.endswith('.npy')]
            if npy_files:
                # Sort numpy files if they have numeric indices
                npy_files.sort(key=get_file_number)

                for file_name in npy_files:
                    file_path = os.path.join(item_path, file_name)

                    # Load numpy data
                    data = np.load(file_path)

                    # Use the file name (without extension) as the dataset name
                    base_name = os.path.splitext(file_name)[0]
                    sensor_group.create_dataset(
                        base_name,
                        data=data,
                        compression='lzf'
                    )
                print(f"  Saved {len(npy_files)} numpy files from {item_name}")
                


def convert_dataset_to_hdf5(input_dir, output_dir):
    """
    Convert all episodes in the dataset to HDF5 format
    
    Args:
        input_dir (str): Directory containing the episode folders
        output_dir (str): Directory where HDF5 files will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    # Get all episode directories
    episode_dirs = [f for f in os.listdir(input_dir) 
                   if os.path.isdir(os.path.join(input_dir, f))]
    episode_dirs.sort(key=lambda x: int(x.split('_')[1]))

    for episode_dir in episode_dirs:
        input_path = os.path.join(input_dir, episode_dir)
        output_path = os.path.join(output_dir, f"{episode_dir}.h5")
        
        print(f"Converting {episode_dir} to HDF5...")
        convert_episode_to_hdf5(input_path, output_path)
        print(f"Converted {episode_dir} successfully")

if __name__ == "__main__":
    # Define input and output directories

    current_path = os.getcwd()

    input_directory = os.path.join(current_path, "data/datasets/water_cup_new")
    output_directory = os.path.join(current_path, "data/datasets/water_cup_new_hdf5_gelsight")
    # Convert all episodes
    convert_dataset_to_hdf5(input_directory, output_directory)
    print("Conversion completed!")