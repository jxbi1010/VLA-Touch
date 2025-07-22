import os
import h5py
import numpy as np


def add_npy_to_h5(npy_file_path, h5_file_path):
    """
    Add data from NPY file to an existing H5 file

    Args:
        npy_file_path (str): Path to the NPY file with GelSight data
        h5_file_path (str): Path to the existing H5 file

    Returns:
        bool: True if successful, False otherwise
    """
    # Check if files exist
    if not os.path.exists(npy_file_path):
        print(f"NPY file not found: {npy_file_path}")
        return False

    if not os.path.exists(h5_file_path):
        print(f"H5 file not found: {h5_file_path}")
        return False

    try:
        # Load NPY data
        gelsight_data = np.load(npy_file_path)

        # Add to H5 file
        with h5py.File(h5_file_path, 'a') as hf:
            # Check if gelsight group already exists
            if 'gelsight_state' in hf:
                print(f"Updating existing gelsight data in {h5_file_path}")
                del hf['gelsight_state']  # Remove existing group

            # Create a new group for gelsight data
            gelsight_group = hf.create_group('gelsight_state')

            gelsight_group.create_dataset(
                'gelsight_state',
                data=gelsight_data,
                compression='lzf'
            )

        print(f"Successfully added GelSight data to {h5_file_path}")
        return True

    except Exception as e:
        print(f"Error adding GelSight data to H5 file: {e}")
        return False


def process_all_episodes(dataset_dir, h5_dir):
    """
    Process all episodes and add GelSight data to corresponding H5 files

    Args:
        dataset_dir (str): Directory with raw episode data containing NPY files
        h5_dir (str): Directory containing H5 files
    """
    # Get all episode directories
    episode_dirs = [f for f in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, f)) and "episode_" in f]
    episode_dirs.sort(key=lambda x: int(x.split('_')[1]))

    success_count = 0
    fail_count = 0

    for episode_dir in episode_dirs:
        episode_num = episode_dir.split('_')[1]
        episode_path = os.path.join(dataset_dir, episode_dir)
        npy_file_path = os.path.join(episode_path, "gelsight.npy")
        h5_file_path = os.path.join(h5_dir, f"{episode_dir}.h5")

        print(f"Processing {episode_dir}...")
        success = add_npy_to_h5(npy_file_path, h5_file_path)

        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"Processing complete. Successfully processed: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    # Define paths
    current_path = os.getcwd()
    dataset_dir = os.path.join(current_path, "data/datasets/cup")
    h5_dir = os.path.join(current_path, "data/datasets/cup_hdf5")

    # Process all episodes
    process_all_episodes(dataset_dir, h5_dir)

    # # TEST: Open an H5 file with GelSight data
    # with h5py.File('data/datasets/cup_hdf5/episode_0.h5', 'r') as f:
    #     # Access GelSight marker positions
    #     gelsight_forces = f['gelsight_state/gelsight_state']['forces']
    #
    #     gel_displacement = f['gelsight_state/gelsight_state']['displacement']
    #
    #     # Access timestamps
    #     timestamps = f['gelsight_state/gelsight_state']['frame']
    #
    #     # Process the data
    #     print(f"Found {len(timestamps)} frames of GelSight data")
    #     print(f"gel_displacement shape: {gel_displacement.shape}")
    #     print(f"gelsight_forces shape: {gelsight_forces.shape}")