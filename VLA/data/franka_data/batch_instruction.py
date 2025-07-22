import os
from pathlib import Path


def batch_create_instructions(dataset_dir, episode_range, instruction):
    """
    Create instruction.txt files for a batch of episodes without viewing them

    Args:
        dataset_dir (str): Directory containing episode folders
        episode_range (tuple): Range of episodes (start, end) inclusive
        instruction (str): Instruction text to write in all files
    """
    start_episode, end_episode = episode_range
    episodes_processed = 0

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
    print(f"Creating instructions for episodes {start_episode} to {end_episode}...")

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

            # Create instruction file path
            instruction_file = os.path.join(episode_path, "instruction.txt")

            # Save instruction to the file
            with open(instruction_file, 'w') as f:
                f.write(instruction)

            print(f"Created instruction for episode {episode_dir}")
            episodes_processed += 1

    print(f"\nBatch processing completed. Processed {episodes_processed} episodes.")
    print(f"Instruction: \"{instruction}\"")


if __name__ == "__main__":
    # Define directory
    current_path = os.getcwd()
    dataset_directory = os.path.join(current_path, "data/datasets/water_cup")

    print("Batch Instruction Generator")
    print("=" * 40)
    print(f"Dataset Directory: {dataset_directory}")
    print("=" * 40)

    # Interactive mode
    while True:
        print("\nBatch Instruction Options:")
        print("1. Create instructions for range of episodes")
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

            # Get instruction text
            instruction = input("Enter instruction text for all episodes in range: ")

            # Confirm before proceeding
            print(f"\nAbout to create instruction.txt with text: \"{instruction}\"")
            print(f"For episodes {start} to {end}.")
            confirm = input("Continue? (y/n): ").lower()

            if confirm == 'y':
                batch_create_instructions(dataset_directory, (start, end), instruction)
            else:
                print("Operation cancelled.")
        else:
            print("Invalid choice. Please try again.")

    print("\nBatch instruction generator completed!")