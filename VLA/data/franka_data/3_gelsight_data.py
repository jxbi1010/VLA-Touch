import os
import cv2
import numpy as np
import re
from pathlib import Path
from residual_controller.tactile.marker.marker_tracker import EnhancedMarkerTracker

def get_file_number(filename):
    # Extract the number between "gel_" and ".jpg"
    match = re.search(r'gel_(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return 0


def process_image_sequence(folder_path, tracker):
    """
    Process all images in a folder using the provided tracker

    Args:
        folder_path (str): Path to folder containing GelSight images
        tracker: Instance of EnhancedMarkerTracker

    Returns:
        dict: Dictionary containing tracking results
    """
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and "gel_" in f]
    image_files.sort(key=get_file_number)

    # Process each image
    results = {
        'frames': [],
        'displacements': [],
        'forces': []
    }

    for file_name in image_files:
        file_path = os.path.join(folder_path, file_name)
        # Get timestamp from filename (assuming format includes timestamp)
        timestamp = get_file_number(file_name)

        # Read and process image
        frame = cv2.imread(file_path)
        if frame is not None:

            # Store results
            # Apply the tracking algorithm
            displacement = tracker.get_marker_state(frame)
            force_magnitude, force_direction = tracker.estimate_force(displacement)

            # Store results
            results['frames'].append(timestamp)
            results['displacements'].append(displacement)
            results['forces'].append((force_magnitude, force_direction[0], force_direction[1]))

    dtype = [
        ('frame', np.float64),
        ('displacement', (np.float64, (63,2))),  # This creates a 2-element array field
        ('forces', np.float64,3),
    ]

    # Create a structured array
    n_frames = len(results['frames'])
    structured_data = np.zeros(n_frames, dtype=dtype)

    # Fill the array
    structured_data['frame'] = results['frames']
    structured_data['displacement'] = results['displacements']
    structured_data['forces'] = results['forces']

    return structured_data


def process_all_episodes(dataset_dir, calibration_img_path, output_dir=None):
    """
    Process GelSight images for all episodes in the dataset

    Args:
        dataset_dir (str): Path to the dataset directory
        output_dir (str, optional): Directory to save NPY files. If None, save in episode directories.
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get all episode directories
    episode_dirs = [f for f in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, f)) and "episode_" in f]
    episode_dirs.sort(key=lambda x: int(x.split('_')[1]))

    for episode_dir in episode_dirs:
        episode_path = os.path.join(dataset_dir, episode_dir)
        gelsight_path = os.path.join(episode_path, "gelsight")

        # Skip if no gelsight folder
        if not os.path.exists(gelsight_path):
            print(f"No GelSight folder found for {episode_dir}, skipping...")
            continue

        print(f"Processing GelSight images for {episode_dir}...")

        # Get calibration image
        if not os.path.exists(calibration_img_path):
            print(f"No calibration image found for {episode_dir}, skipping...")
            continue

        calibration_img = cv2.imread(calibration_img_path)

        # Initialize tracker
        tracker = EnhancedMarkerTracker(
            grid_rows=7,
            grid_cols=9,
            calibration_frame=calibration_img,
            gelsight_version='standard'
        )

        # Process image sequence
        results = process_image_sequence(gelsight_path, tracker)

        # Save results
        if output_dir:
            save_path = os.path.join(output_dir, f"{episode_dir}_gelsight.npy")
        else:
            save_path = os.path.join(episode_path, "gelsight_force.npy")

        np.save(save_path, results)
        print(f"Saved GelSight results for {episode_dir} to {save_path}")


if __name__ == "__main__":
    # Define path to dataset
    current_path = os.getcwd()
    dataset_dir = os.path.join(current_path, "data/datasets/water_cup_new")
    calibration_img_path = os.path.join(dataset_dir,"episode_0/gelsight/gel_0.jpg")

    # Process all episodes
    process_all_episodes(dataset_dir,calibration_img_path)

    print("GelSight processing completed!")