import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt


class EnhancedMarkerTracker:
    def __init__(self, grid_rows=7, grid_cols=9, calibration_frame=None, gelsight_version='standard'):
        """
        Initialize the marker tracker with optional calibration frame and grid dimensions.

        Parameters:
        -----------
        grid_rows : int, optional
            Number of rows in the marker grid
        grid_cols : int, optional
            Number of columns in the marker grid
        calibration_frame : ndarray, optional
            A reference frame for calibration (no force applied)
        gelsight_version : str
            Version of the GelSight sensor ('standard' or 'HSR')
        """
        self.grid_dims = (grid_rows, grid_cols) if grid_rows and grid_cols else None
        self.expected_markers = grid_rows*grid_cols
        self.baseline_markers = None
        self.gelsight_version = gelsight_version

        if calibration_frame is not None:
            print("Calibrate with a reference frame.")
            self.calibrate(calibration_frame)

    def calibrate(self, calibration_frame):
        """
        Establish baseline marker positions from a calibration frame.

        Parameters:
        -----------
        calibration_frame : ndarray
            A reference frame for calibration (no force applied)
        """
        # Process the calibration frame to detect markers
        processed_frame = self.preprocess_frame(calibration_frame)
        markers = self.detect_markers(processed_frame)

        # Store as baseline
        self.baseline_markers = markers

        # If grid dimensions not provided, estimate them
        if self.grid_dims is None:
            n_markers = len(markers)
            grid_size = int(np.sqrt(n_markers))
            self.grid_dims = (grid_size, n_markers // grid_size)

            print(f"Estimated grid dimensions: {self.grid_dims}, detected {n_markers} markers")

        # Create ideal grid for reference
        self.ideal_grid = self.create_ideal_grid(markers)

        return markers

    def preprocess_frame(self, frame):
        """
        Preprocess frame based on GelSight version.

        Parameters:
        -----------
        frame : ndarray
            Input image frame

        Returns:
        --------
        processed_frame : ndarray
            Preprocessed frame ready for marker detection
        """
        if self.gelsight_version == 'HSR':
            return self.init_HSR(frame)
        else:
            return self.init_standard(frame)

    def init_standard(self, frame):
        """
        Standard initialization for marker detection.

        Parameters:
        -----------
        frame : ndarray
            Input image frame

        Returns:
        --------
        processed_frame : ndarray
            Preprocessed frame
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding to handle uneven lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Perform morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return cleaned

    def init_HSR(self, frame):
        """
        HSR-specific initialization for marker detection.

        Parameters:
        -----------
        frame : ndarray
            Input image frame

        Returns:
        --------
        processed_frame : ndarray
            Preprocessed frame
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Invert the image (assuming markers are dark on light background for HSR)
        gray = 255 - gray

        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

        # Apply binary thresholding
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        # Clean up with morphology
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return cleaned

    def detect_markers(self, processed_frame,filter_coords=None, filter_threshold=5): #filter_coords = [18,109]
        """
        Detect markers in the processed frame.

        Parameters:
        -----------
        processed_frame : ndarray
            Preprocessed binary image

        Returns:
        --------
        markers : ndarray
            Array of marker centroid coordinates, shape (n, 2)
        """
        # Find contours in the binary image
        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size to remove noise
        min_area = 10  # Min area threshold
        max_area = 500  # Max area threshold
        filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

        # Calculate centroids of valid contours
        markers = []
        for contour in filtered_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                markers.append([cx, cy])

        candidate_markers = np.array(markers)

        # Filter out markers based on provided coordinates
        if filter_coords is not None and len(markers) > 0:
            filter_coords = np.array(filter_coords)

            # Keep only markers that are not close to any filter coordinates
            keep_mask = np.ones(len(candidate_markers), dtype=bool)

            for i, marker in enumerate(candidate_markers):
                # Calculate distance to each filter coordinate
                distances = np.sqrt(np.sum((filter_coords - marker) ** 2))
                # If any distance is below threshold, mark for removal
                if np.any(distances < filter_threshold):
                    keep_mask[i] = False

            # Apply mask to keep only wanted markers
            candidate_markers = candidate_markers[keep_mask]

        # Handle the case when we have too many or too few markers
        expected_count = self.expected_markers

        if len(candidate_markers) == expected_count:
            # Perfect case - return as is
            return candidate_markers

        elif len(candidate_markers) > expected_count:
            # Too many markers - select the most reliable ones

            # Method 1: Use spatial clustering to identify marker groups
            # This works well when the "extra" markers are close to real markers
            from sklearn.cluster import KMeans

            # Use KMeans to cluster markers into the expected number of groups
            kmeans = KMeans(n_clusters=expected_count, random_state=0).fit(candidate_markers)

            # For each cluster, select the marker closest to the centroid
            refined_markers = []
            for i in range(expected_count):
                cluster_points = candidate_markers[kmeans.labels_ == i]
                if len(cluster_points) > 0:
                    centroid = kmeans.cluster_centers_[i]
                    distances = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))
                    closest_idx = np.argmin(distances)
                    refined_markers.append(cluster_points[closest_idx])

            return np.array(refined_markers)

        elif 0 < len(candidate_markers) < expected_count:
            # Too few markers - warn but return what we found
            print(f"Warning: Expected {expected_count} markers but found only {len(candidate_markers)}")
            return candidate_markers

        else:
            # No markers found
            print("Warning: No markers detected")
            return np.array([])



    def create_ideal_grid(self, markers):
        """
        Create an ideal grid based on detected markers.

        Parameters:
        -----------
        markers : ndarray
            Array of detected marker positions

        Returns:
        --------
        grid_points : ndarray
            Array of ideal grid points
        """
        rows, cols = self.grid_dims

        # Get bounding box of markers
        x_min, y_min = np.min(markers, axis=0)
        x_max, y_max = np.max(markers, axis=0)

        # Create grid
        x = np.linspace(x_min, x_max, cols)
        y = np.linspace(y_min, y_max, rows)

        # Generate grid points
        grid_points = []
        for i in y:
            for j in x:
                grid_points.append([j, i])

        return np.array(grid_points)

    def get_marker_state(self, frame):
        """
        Get marker state from current frame.

        Parameters:
        -----------
        frame : ndarray
            Current image frame

        Returns:
        --------
        displacement : ndarray
            Displacement vectors for each marker
        """
        # Process frame and detect markers
        processed_frame = self.preprocess_frame(frame)
        current_markers = self.detect_markers(processed_frame)

        # If this is the first frame and no baseline exists, set it as baseline
        if self.baseline_markers is None:
            self.calibrate(frame)
            # Return zero displacement for the baseline frame
            return np.zeros((len(current_markers), 2))

        # Match current markers to baseline markers
        # (using a simple nearest neighbor approach)
        # This is a simplified matching - real implementation may need more robust matching
        displacement = self.match_and_compute_displacement(current_markers)

        return displacement

    def match_and_compute_displacement(self, current_markers):
        """
        Match current markers to baseline and compute displacement.

        Parameters:
        -----------
        current_markers : ndarray
            Array of current marker positions

        Returns:
        --------
        displacement : ndarray
            Array of displacement vectors
        """
        # Make sure we have detected markers
        if len(current_markers) == 0:
            return np.array([])

        # Simple matching approach: match each current marker to closest baseline marker
        # Using kNN for efficient matching
        from scipy.spatial import cKDTree

        # Build KD-tree for baseline markers
        tree = cKDTree(self.baseline_markers)

        # Find nearest baseline marker for each current marker
        distances, indices = tree.query(current_markers, k=1)

        # Compute displacement vectors
        matched_baseline = self.baseline_markers[indices]
        displacement = current_markers - matched_baseline

        return displacement

    def estimate_force(self, displacement):
        """
        Estimate force from marker displacement.

        Parameters:
        -----------
        displacement : ndarray
            Array of displacement vectors

        Returns:
        --------
        force_magnitude : float
            Estimated magnitude of applied force
        force_direction : ndarray
            Estimated direction of applied force [x, y]
        """
        if len(displacement) == 0:
            return 0, np.array([0, 0])

        # Calculate average displacement vector
        avg_displacement = np.mean(displacement, axis=0)

        # Calculate force magnitude (L2 norm of average displacement)
        force_magnitude = np.linalg.norm(avg_displacement)

        # Calculate force direction (unit vector)
        if force_magnitude > 0:
            force_direction = avg_displacement / force_magnitude
        else:
            force_direction = np.array([0, 0])

        return force_magnitude, force_direction


def process_image_sequence(folder_path, tracker, prefix='gel_', extension='.jpg'):
    """
    Process a sequence of images from a folder in order.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing image sequence
    tracker : EnhancedMarkerTracker
        Instance of the EnhancedMarkerTracker class
    prefix : str
        Prefix of the image filenames (default: 'gel_')
    extension : str
        Extension of the image files (default: '.jpg')

    Returns:
    --------
    results : dict
        Dictionary containing processed results:
        - 'frames': list of frame numbers
        - 'displacements': list of displacement arrays
        - 'forces': list of (magnitude, direction) tuples
    """
    # Get all image files that match the pattern
    image_files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith(extension)]

    # Extract the number from each filename for sorting
    def extract_number(filename):
        # Extract the number between prefix and extension
        match = re.search(f'{prefix}(\d+){extension}', filename)
        if match:
            return int(match.group(1))
        return float('inf')  # Files that don't match the pattern go to the end

    # Sort the files based on their numerical order
    image_files.sort(key=extract_number)

    # Process each image
    results = {
        'frames': [],
        'displacements': [],
        'forces': []
    }

    # Use the first frame for calibration
    if image_files:
        first_img_path = os.path.join(folder_path, image_files[0])
        first_frame = cv2.imread(first_img_path)
        if first_frame is not None:
            tracker.calibrate(first_frame)

    # Process all frames
    for img_file in image_files:
        frame_num = extract_number(img_file)
        img_path = os.path.join(folder_path, img_file)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        # Apply the tracking algorithm
        displacement = tracker.get_marker_state(frame)
        force_magnitude, force_direction = tracker.estimate_force(displacement)
        print(f"num markers:{displacement.shape[0]},force:{force_magnitude, force_direction}")
        # Store results
        results['frames'].append(frame_num)
        results['displacements'].append(displacement)
        results['forces'].append((force_magnitude, force_direction))

        print(f"Processed {img_file}, detected {len(displacement)} markers, "
              f"force magnitude: {force_magnitude:.2f}")

    return results


def visualize_image_sequence(folder_path, tracker, prefix='gel_', extension='.jpg', delay=100):
    """
    Visualize the marker tracking on a sequence of images.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing image sequence
    tracker : EnhancedMarkerTracker
        Instance of the EnhancedMarkerTracker class
    prefix : str
        Prefix of the image filenames (default: 'gel_')
    extension : str
        Extension of the image files (default: '.jpg')
    delay : int
        Delay between frames in milliseconds (default: 100)
    """
    # Get and sort image files
    image_files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith(extension)]

    def extract_number(filename):
        match = re.search(f'{prefix}(\d+){extension}', filename)
        if match:
            return int(match.group(1))
        return float('inf')

    image_files.sort(key=extract_number)

    # Use the first frame for calibration
    if image_files:
        first_img_path = os.path.join(folder_path, image_files[0])
        first_frame = cv2.imread(first_img_path)
        if first_frame is not None:
            tracker.calibrate(first_frame)

    # Process and visualize each frame
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        # Process frame
        processed_frame = tracker.preprocess_frame(frame)
        current_markers = tracker.detect_markers(processed_frame)

        # Compute displacement if we have a baseline
        if tracker.baseline_markers is not None:
            displacement = tracker.match_and_compute_displacement(current_markers)
            force_magnitude, force_direction = tracker.estimate_force(displacement)
        else:
            displacement = np.zeros((len(current_markers), 2))
            force_magnitude, force_direction = 0, np.array([0, 0])

        # Create visualization frame (copy of original)
        viz_frame = frame.copy()

        # Draw detected markers
        for marker in current_markers:
            cv2.circle(viz_frame, tuple(marker.astype(int)), 3, (0, 255, 0), -1)

        # Draw displacement vectors if not zero
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
        if force_magnitude > 0.5:  # Only draw if magnitude is significant
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

        # Display frame number
        cv2.putText(
            viz_frame,
            f"Frame: {extract_number(img_file)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Show processed binary image in corner
        h, w = processed_frame.shape
        scale = 0.25
        small_processed = cv2.resize(processed_frame, (int(w * scale), int(h * scale)))

        # Convert to BGR for overlay
        small_processed_bgr = cv2.cvtColor(small_processed, cv2.COLOR_GRAY2BGR)

        # Overlay in top-right corner
        viz_frame[10:10 + small_processed_bgr.shape[0],
        viz_frame.shape[1] - 10 - small_processed_bgr.shape[1]:viz_frame.shape[1] - 10] = small_processed_bgr

        # Display the frame
        cv2.imshow('Enhanced Marker Tracking', viz_frame)

        # Wait for key press or delay
        key = cv2.waitKey(delay)

        # Exit if 'q' is pressed
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def plot_force_over_time(results):
    """
    Plot force magnitude over time.

    Parameters:
    -----------
    results : dict
        Results dictionary from process_image_sequence
    """
    frames = results['frames']
    forces = [f[0] for f in results['forces']]  # Extract magnitudes

    plt.figure(figsize=(10, 6))
    plt.plot(frames, forces, 'b-', linewidth=2)
    plt.scatter(frames, forces, color='red', s=30)

    plt.title('Force Magnitude Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Force Magnitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create tracker
    calibration_img_path = "data/datasets/wipe/episode_0/gelsight/gel_0.jpg"
    calibration_img = cv2.imread(calibration_img_path)
    tracker = EnhancedMarkerTracker(grid_rows=7, grid_cols=9, calibration_frame=calibration_img, gelsight_version='standard')

    # Folder with image sequence
    folder_path = "data/datasets/wipe/episode_70/gelsight"

    # Process images and get results
    results = process_image_sequence(folder_path, tracker)
    plot_force_over_time(results)

    # Visualize the sequence
    # visualize_image_sequence(folder_path, tracker, delay=100)