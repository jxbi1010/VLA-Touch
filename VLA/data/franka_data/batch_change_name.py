import os
import shutil
from pathlib import Path


def batch_rename_files(dataset_dir, episode_range, old_filename, new_filename, subfolder=None):
    """
    Rename a specific file across multiple episodes

    Args:
        dataset_dir (str): Directory containing episode folders
        episode_range (tuple): Range of episodes (start, end) inclusive
        old_filename (str): Original filename to change
        new_filename (str): New filename to use
        subfolder (str, optional): Subfolder within episode folder (e.g., 'camera1')
    """
    start_episode, end_episode = episode_range
    files_renamed = 0
    files_not_found = 0

    # Get all episode directories
    all_episode_dirs = [f for f in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, f))]

    # Sort episode directories numerically
    all_episode_dirs.sort(
        key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else float('inf'))

    if not all_episode_dirs:
        print(f"No episode directories found in {dataset_dir}")
        return

    print(f"Found {len(all_episode_dirs)} total episodes")
    print(f"Renaming files in episodes {start_episode} to {end_episode}...")

    for episode_dir in all_episode_dirs:
        # Extract episode number
        try:
            episode_num = int(episode_dir.split('_')[1])
        except (IndexError, ValueError):
            print(f"Skipping directory with invalid format: {episode_dir}")
            continue

        # Check if episode is in the specified range
        if start_episode <= episode_num <= end_episode:
            episode_path = os.path.join(dataset_dir, episode_dir)

            # Determine the path to the file
            if subfolder:
                file_dir = os.path.join(episode_path, subfolder)
                if not os.path.exists(file_dir):
                    print(f"Subfolder '{subfolder}' not found in episode {episode_dir}")
                    files_not_found += 1
                    continue
            else:
                file_dir = episode_path

            # Create paths for old and new files
            old_file_path = os.path.join(file_dir, old_filename)
            new_file_path = os.path.join(file_dir, new_filename)

            # Check if old file exists
            if os.path.exists(old_file_path):
                # Rename the file
                try:
                    shutil.move(old_file_path, new_file_path)
                    print(f"Renamed in episode {episode_dir}: {old_filename} → {new_filename}")
                    files_renamed += 1
                except Exception as e:
                    print(f"Error renaming file in episode {episode_dir}: {e}")
            else:
                print(f"File '{old_filename}' not found in episode {episode_dir}")
                files_not_found += 1

    print(f"\nBatch renaming completed. Renamed {files_renamed} files.")
    print(f"Files not found: {files_not_found}")


if __name__ == "__main__":
    # Define directory
    current_path = os.getcwd()
    dataset_directory = os.path.join(current_path, "data/datasets/wipe")

    print("Batch File Renamer")
    print("=" * 40)
    print(f"Dataset Directory: {dataset_directory}")
    print("=" * 40)

    # Interactive mode
    while True:
        print("\nBatch Renaming Options:")
        print("1. Rename files across episodes")
        print("2. Exit")

        choice = input("Enter your choice (1/2): ").strip()

        if choice == '2':
            break
        elif choice == '1':
            # Get episode range from user
            try:
                start = int(input("Enter starting episode number: "))
                end = int(input("Enter ending episode number: "))
                if start > end:
                    print("Error: Starting episode must be less than or equal to ending episode.")
                    continue
            except ValueError:
                print("Please enter valid numbers.")
                continue

            # Get file names
            old_filename = input("Enter the original filename to change: ")
            new_filename = input("Enter the new filename: ")

            # Ask for subfolder (optional)
            use_subfolder = input("Is the file in a subfolder? (y/n): ").lower()
            subfolder = None
            if use_subfolder == 'y':
                subfolder = input("Enter subfolder name (e.g., camera1): ")

            # Confirm before proceeding
            subfolder_text = f" in subfolder '{subfolder}'" if subfolder else ""
            print(f"\nAbout to rename '{old_filename}' to '{new_filename}'{subfolder_text}")
            print(f"For episodes {start} to {end}.")
            confirm = input("Continue? (y/n): ").lower()

            if confirm == 'y':
                batch_rename_files(dataset_directory, (start, end), old_filename, new_filename, subfolder)
            else:
                print("Operation cancelled.")
        else:
            print("Invalid choice. Please try again.")

    print("\nBatch file renamer completed!")