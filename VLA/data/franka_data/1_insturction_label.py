import os
import numpy as np
import cv2
import time
import re
from pathlib import Path


def get_file_number(filename):
    # Extract the number between "rgb_" and ".jpg"
    match = re.search(r'rgb_(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return 0


def play_episode_and_annotate(episode_folder_path, fps=10):
    """
    Play camera1 images from original episode folder and collect annotation at the end

    Args:
        episode_folder_path (str): Path to the episode directory
        fps (int): Frames per second for playback
    """
    episode_name = os.path.basename(episode_folder_path)
    print(f"\nPlaying episode: {episode_name}")

    # Check if camera1 folder exists
    camera_folder = os.path.join(episode_folder_path, "camera1")
    if not os.path.exists(camera_folder):
        print(f"No camera1 folder found in {episode_name}")
        return False

    # Get all image files in the camera folder
    image_files = [f for f in os.listdir(camera_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"No camera images found in {episode_name}/camera1")
        return False

    # Sort the files based on their numeric index
    image_files.sort(key=get_file_number)
    num_frames = len(image_files)
    print(f"Found {num_frames} frames")

    # Create a window for display
    window_name = f"Episode: {episode_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    # Play the images
    frame_delay = 1.0 / fps  # in seconds

    print("Playing... Press 'q' to quit, any other key to pause/resume")

    paused = False
    for i, file_name in enumerate(image_files):
        file_path = os.path.join(camera_folder, file_name)

        # Read the image
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Error reading frame: {file_path}")
            continue

        # Add frame counter on the image
        frame_with_counter = frame.copy()
        cv2.putText(frame_with_counter, f"{i + 1}/{num_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(window_name, frame_with_counter)

        # Handle key presses
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
            elif key != 255:  # Any key press
                paused = not paused

            if not paused:
                break

            # When paused, show additional info
            info_frame = frame.copy()
            cv2.putText(info_frame, f"Frame: {i + 1}/{num_frames} [PAUSED]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(info_frame, f"File: {file_name}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, info_frame)

        # Wait for the next frame
        if not paused:
            time.sleep(frame_delay)

    # Close the window after playback
    cv2.destroyAllWindows()

    # Ask for instruction
    print("\nPlease type your instruction for this episode (Press Enter when done):")
    instruction = input("> ")

    # Save instruction to a text file in the episode folder
    instruction_file = os.path.join(episode_folder_path, "instruction.txt")
    with open(instruction_file, 'w') as f:
        f.write(instruction)

    print(f"Instruction saved to {instruction_file}")
    return True


def process_all_episodes(dataset_dir):
    """
    Process all episodes in the dataset

    Args:
        dataset_dir (str): Directory containing episode folders
    """
    # Get all episode directories
    episode_dirs = [f for f in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, f))]

    # Sort episode directories numerically
    episode_dirs.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else float('inf'))

    if not episode_dirs:
        print(f"No episode directories found in {dataset_dir}")
        return

    print(f"Found {len(episode_dirs)} episodes to process")

    for episode_dir in episode_dirs:
        episode_path = os.path.join(dataset_dir, episode_dir)

        # Play episode and collect annotation
        continue_process = play_episode_and_annotate(episode_path)

        if not continue_process:
            print("Playback stopped.")
            break

        print(f"Completed episode {episode_dir}")
        print("-" * 50)

        # Ask if user wants to continue to next episode
        if episode_dir != episode_dirs[-1]:  # Not the last directory
            response = input("Continue to next episode? (y/n): ")
            if response.lower() != 'y':
                print("Annotation process stopped by user.")
                break


if __name__ == "__main__":
    # Define directory
    current_path = os.getcwd()
    dataset_directory = os.path.join(current_path, "data/datasets/cup")

    print("Episode Playback and Annotation Tool")
    print("=" * 40)
    print(f"Dataset Directory: {dataset_directory}")
    print("=" * 40)

    # Process all episodes
    process_all_episodes(dataset_directory)

    print("\nAnnotation process completed!")



