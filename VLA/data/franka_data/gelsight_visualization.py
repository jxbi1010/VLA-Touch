import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path
from residual_controller.tactile.marker.marker_tracker import EnhancedMarkerTracker


def get_file_number(filename):
    # Extract the number between "gel_" and ".jpg"
    match = re.search(r'gel_(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return 0


def create_visualization_frame(frame, current_markers, displacement=None, force_magnitude=0, force_direction=None):
    """
    Create a visualization frame with markers and displacement vectors.

    Args:
        frame: Original image frame
        current_markers: Detected marker positions
        displacement: Displacement vectors for markers
        force_magnitude: Magnitude of the estimated force
        force_direction: Direction of the estimated force

    Returns:
        viz_frame: Visualization frame with overlays
    """
    # Create visualization frame (copy of original)
    viz_frame = frame.copy()

    # Draw detected markers
    for marker in current_markers:
        cv2.circle(viz_frame, tuple(marker.astype(int)), 3, (0, 255, 0), -1)

    # Draw displacement vectors if provided
    if displacement is not None:
        for i, marker in enumerate(current_markers):
            if i < len(displacement) and np.any(displacement[i] != 0):
                # Scale displacement for visibility
                scale = 3
                end_point = marker + scale * displacement[i]

                # Draw arrow
                cv2.arrowedLine(
                    viz_frame,
                    tuple(marker.astype(int)),
                    tuple(end_point.astype(int)),
                    (0, 0, 255),
                    1,
                    tipLength=0.3
                )

    # Draw force vector (global)
    if force_magnitude > 0.5 and force_direction is not None:  # Only draw if magnitude is significant
        # Get frame center
        center_x, center_y = viz_frame.shape[1] // 2, viz_frame.shape[0] // 2

        # Scale and draw global force vector
        scale = 50
        force_end = (
            center_x + int(scale * force_magnitude * force_direction[0]),
            center_y + int(scale * force_magnitude * force_direction[1])
        )

        cv2.arrowedLine(
            viz_frame,
            (center_x, center_y),
            force_end,
            (255, 0, 0),
            2,
            tipLength=0.3
        )

        # Add force magnitude text
        cv2.putText(
            viz_frame,
            f"Force: {force_magnitude:.2f}",
            (10, viz_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    return viz_frame


def process_image_sequence_with_viz(folder_path, tracker, viz_output_path):
    """
    Process all images in a folder using the provided tracker and save visualizations

    Args:
        folder_path (str): Path to folder containing GelSight images
        tracker: Instance of EnhancedMarkerTracker
        viz_output_path (str): Path to save visualization images

    Returns:
        dict: Dictionary containing tracking results
    """
    # Create visualization directory if it doesn't exist
    os.makedirs(viz_output_path, exist_ok=True)

    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and "gel_" in f]
    image_files.sort(key=get_file_number)

    # Process each image
    results = {
        'frames': [],
        'displacements': [],
        'forces': []
    }

    # Use the first frame for calibration if not already calibrated
    if image_files and tracker.baseline_markers is None:
        first_img_path = os.path.join(folder_path, image_files[0])
        first_frame = cv2.imread(first_img_path)
        if first_frame is not None:
            tracker.calibrate(first_frame)

    for i, file_name in enumerate(image_files):
        file_path = os.path.join(folder_path, file_name)
        # Get timestamp/frame number from filename
        frame_num = get_file_number(file_name)

        # Read and process image
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Warning: Could not read image {file_path}")
            continue

        # Process frame to detect markers
        processed_frame = tracker.preprocess_frame(frame)
        current_markers = tracker.detect_markers(processed_frame)

        # Apply the tracking algorithm
        displacement = tracker.get_marker_state(frame)
        force_magnitude, force_direction = tracker.estimate_force(displacement)

        # Store results
        results['frames'].append(frame_num)
        results['displacements'].append(displacement)
        results['forces'].append((force_magnitude, force_direction[0], force_direction[1]))

        # Create visualization frame
        viz_frame = create_visualization_frame(
            frame,
            current_markers,
            displacement,
            force_magnitude,
            force_direction
        )

        # Add frame number to visualization
        # cv2.putText(
        #     viz_frame,
        #     f"Frame: {frame_num}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2
        # )

        # Save visualization frame
        viz_file_path = os.path.join(viz_output_path, f"viz_{frame_num:04d}.jpg")
        cv2.imwrite(viz_file_path, viz_frame)

        # Print progress
        if (i + 1) % 10 == 0 or i == len(image_files) - 1:
            print(f"Processed {i + 1}/{len(image_files)} images, current force: {force_magnitude:.2f}")

    # Create and save force plot
    if results['frames']:
        plt.figure(figsize=(10, 6))
        frames = results['frames']
        forces = [f[0] for f in results['forces']]  # Extract magnitudes

        plt.plot(frames, forces, 'b-', linewidth=2)
        plt.scatter(frames, forces, color='red', s=30)

        plt.title('Force Magnitude Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Force Magnitude')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(viz_output_path, "force_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Force plot saved to {plot_path}")

    # Create a structured numpy array for the results
    dtype = [
        ('frame', np.float64),
        ('displacement', (np.float64, (63, 2))),  # This creates a 2-element array field
        ('forces', np.float64, 3),
    ]

    # Create and fill the structured array
    n_frames = len(results['frames'])
    structured_data = np.zeros(n_frames, dtype=dtype)

    structured_data['frame'] = results['frames']
    structured_data['displacement'] = results['displacements']
    structured_data['forces'] = results['forces']

    return structured_data


def create_force_direction_visualization(viz_output_path, results):
    """
    Create a visualization of force directions over time

    Args:
        viz_output_path (str): Path to save the visualization
        results (dict): Dictionary with force data
    """
    # if not results['frames'] or len(results['forces']) < 2:
    #     return

    # Extract force directions
    frames = results['frames']
    fx = [f[1] for f in results['forces']]  # x-component
    fy = [f[2] for f in results['forces']]  # y-component
    magnitudes = [f[0] for f in results['forces']]  # magnitudes

    # Create a scatter plot with arrows
    plt.figure(figsize=(12, 8))

    # Create a colormap based on frame number
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(frames)))

    # Plot force directions with colors representing time and size representing magnitude
    for i in range(len(frames)):
        # Skip points with very small magnitude
        if magnitudes[i] < 0.1:
            continue

        # Scale arrow size by magnitude
        arrow_scale = magnitudes[i] * 0.5

        plt.arrow(
            0, 0,  # Start at origin
            fx[i] * arrow_scale,
            fy[i] * arrow_scale,
            color=colors[i],
            alpha=0.7,
            width=0.005,
            head_width=0.03
        )

        print(magnitudes[-1],fx[-1],fy[-1])

    # Create a colorbar to show the time progression
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(frames), max(frames)))
    # sm.set_array([])
    # cbar = plt.colorbar(sm)
    # cbar.set_label('Frame Number')

    # Set plot properties
    plt.title('Force Direction Over Time')
    plt.xlabel('X-Component')
    plt.ylabel('Y-Component')
    plt.grid(True)
    plt.axis('equal')  # Equal scale

    # Add a circle to represent unit force
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    plt.gca().add_patch(circle)

    # Set limits
    max_range = 1.2
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    # Save the plot
    direction_plot_path = os.path.join(viz_output_path, "force_direction_plot.png")
    plt.savefig(direction_plot_path, dpi=300)
    plt.close()

    print(f"Force direction visualization saved to {direction_plot_path}")


def process_all_episodes_with_viz(dataset_dir, calibration_img_path, output_dir=None):
    """
    Process GelSight images for all episodes in the dataset with visualizations

    Args:
        dataset_dir (str): Path to the dataset directory
        calibration_img_path (str): Path to the calibration image
        output_dir (str, optional): Directory to save NPY files. If None, save in episode directories.
    """
    # Get all episode directories
    episode_dirs = [f for f in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, f)) and "episode_" in f]
    episode_dirs.sort(key=lambda x: int(x.split('_')[1]))

    # Load calibration image
    if not os.path.exists(calibration_img_path):
        print(f"Error: Calibration image not found at {calibration_img_path}")
        return

    calibration_img = cv2.imread(calibration_img_path)
    if calibration_img is None:
        print(f"Error: Could not read calibration image at {calibration_img_path}")
        return

    # Initialize tracker
    tracker = EnhancedMarkerTracker(
        grid_rows=7,
        grid_cols=9,
        calibration_frame=calibration_img,
        gelsight_version='standard'
    )

    # Process each episode
    for episode_dir in episode_dirs:
        episode_path = os.path.join(dataset_dir, episode_dir)
        gelsight_path = os.path.join(episode_path, "gelsight")

        # Skip if no gelsight folder
        if not os.path.exists(gelsight_path):
            print(f"No GelSight folder found for {episode_dir}, skipping...")
            continue

        print(f"\nProcessing GelSight images for {episode_dir}...")

        # Create visualization output directory
        viz_output_path = os.path.join(episode_path, "gelsight_viz")
        os.makedirs(viz_output_path, exist_ok=True)

        # Process image sequence with visualizations
        results = process_image_sequence_with_viz(gelsight_path, tracker, viz_output_path)

        # Create force direction visualization
        create_force_direction_visualization(viz_output_path, {
            'frames': results['frame'],
            'forces': [(f[0], f[1], f[2]) for f in results['forces']]
        })

        # Save results to npy file
        if output_dir:
            save_path = os.path.join(output_dir, f"{episode_dir}_gelsight.npy")
        else:
            save_path = os.path.join(episode_path, "gelsight_force.npy")

        np.save(save_path, results)
        print(f"Saved GelSight results for {episode_dir} to {save_path}")
        print(f"Visualizations saved to {viz_output_path}")


def create_summary_video(episode_path, fps=10):
    """
    Create a summary video from the visualization images

    Args:
        episode_path (str): Path to episode directory
        fps (int): Frames per second for the video
    """
    viz_path = os.path.join(episode_path, "gelsight_viz")
    if not os.path.exists(viz_path):
        print(f"No visualization directory found at {viz_path}")
        return

    # Get all visualization images
    viz_files = [f for f in os.listdir(viz_path) if f.startswith("viz_") and f.endswith(".jpg")]
    viz_files.sort(key=lambda x: int(re.search(r'viz_(\d+)\.jpg', x).group(1)))

    if not viz_files:
        print(f"No visualization images found in {viz_path}")
        return

    # Read first image to get dimensions
    first_img = cv2.imread(os.path.join(viz_path, viz_files[0]))
    height, width, layers = first_img.shape

    # Create video writer
    video_path = os.path.join(viz_path, "marker_tracking.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Add each image to video
    for file in viz_files:
        img_path = os.path.join(viz_path, file)
        img = cv2.imread(img_path)
        video.write(img)

    # Release video writer
    video.release()
    print(f"Created summary video at {video_path}")


if __name__ == "__main__":
    # Define path to dataset
    current_path = os.getcwd()
    dataset_dir = os.path.join(current_path, "data/datasets/temp/cup")
    calibration_img_path = os.path.join(dataset_dir, "episode_0/gelsight/gel_0.jpg")

    # Process all episodes with visualizations
    process_all_episodes_with_viz(dataset_dir, calibration_img_path)

    # Create summary videos for each episode
    episode_dirs = [f for f in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, f)) and "episode_" in f]

    for episode_dir in episode_dirs:
        episode_path = os.path.join(dataset_dir, episode_dir)
        if os.path.exists(os.path.join(episode_path, "gelsight_viz")):
            create_summary_video(episode_path)

    print("\nGelSight processing and visualization completed!")