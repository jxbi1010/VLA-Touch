import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
# Import the existing dataset class
from controller_dataset import ControllerDataModule
import argparse
from bridge_controller import load_bridge_controller


# Enhanced test function with improved visualization
def test_diffusion_controller(
        checkpoint_dir,
        data_dir=None,
        data_loader=None,
        context_frames=2,
        horizon=32,
        image_size=384,
        device='cuda',
        num_samples=10,
        visualize=True,
        save_dir=None
):
    """
    Test a trained diffusion controller by sampling trajectories from the dataset
    and running inference with enhanced visualization.

    Args:
        checkpoint_dir: Directory containing the checkpoint to load
        data_dir: Directory containing the dataset
        context_frames: Number of past frames to use as context
        horizon: Number of future steps to predict
        image_size: Size of images
        diffusion_steps: Number of diffusion steps for sampling
        device: Device to use for inference
        num_samples: Number of samples to test
        state_dim: Dimension of the robot state and action
        hidden_dim: Hidden dimension for encoders
        force_dim: Dimension of the force
        use_force: Whether to use force information
        image_model_path: Path to the DINOv2 model to use
        visualize: Whether to create visualizations
        save_dir: Directory to save visualizations (defaults to checkpoint_dir/test_visualizations)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_diffusion_controller")
    logger.info(f"Testing diffusion controller from checkpoint: {checkpoint_dir}")



    if data_loader:
        test_dataloader = data_loader
    else:
        # Create data module and sample trajectories
        data_module = ControllerDataModule(
            data_dir=data_dir,
            batch_size=32,  # Process one trajectory at a time
            num_workers=4,  # No parallel workers for testing
            context_frames=context_frames,
            horizon=horizon,
            image_size=image_size,
            val_ratio=0.2,
            stride=1
        )
        # Use validation dataset for testing
        test_dataloader = data_module.val_dataloader()

    # Create and load diffusion controller
    controller = load_bridge_controller()

    controller.load(checkpoint_dir)
    logger.info(f"Loaded model from {checkpoint_dir}")

    # Set controller to evaluation mode
    controller.eval()

    # Setup visualization directory
    if visualize:
        if save_dir is None:
            save_dir = os.path.join(checkpoint_dir, "test_visualizations")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Visualizations will be saved to {save_dir}")

    # Prepare for visualization
    import matplotlib.pyplot as plt
    import random
    from mpl_toolkits.mplot3d import Axes3D

    # Loop through test samples
    test_errors = []
    test_vla_errors = []  # Track VLA errors for comparison


    sample_indices = []
    total_samples = 0
    # First, count total samples across all batches
    for i, batch in enumerate(test_dataloader):
        batch_size = batch['states'].shape[0]  # Get actual batch size
        total_samples += batch_size
        # Store mapping from global sample index to (batch_idx, within_batch_idx)
        for j in range(batch_size):
            sample_indices.append((i, j))

    logger.info(f"Total available test samples: {total_samples}")

    # Randomly sample indices
    if num_samples <= total_samples:
        chosen_indices = random.sample(range(len(sample_indices)), num_samples)
    else:
        logger.warning(f"Requested {num_samples} samples but only {total_samples} are available")
        chosen_indices = list(range(len(sample_indices)))

    # We need to load the batches that contain our samples
    batches_to_load = set([sample_indices[idx][0] for idx in chosen_indices])
    loaded_batches = {}

    # Load only the needed batches
    for i, batch in enumerate(test_dataloader):
        if i in batches_to_load:
            loaded_batches[i] = batch

    # Now process each selected sample
    for sample_num, idx in enumerate(chosen_indices):
        batch_idx, within_batch_idx = sample_indices[idx]
        batch = loaded_batches[batch_idx]

        logger.info(
            f"Testing random sample {sample_num + 1}/{len(chosen_indices)} (batch {batch_idx}, index {within_batch_idx})")

        # Extract data for this specific sample
        states = batch['states'].to(device) # [1, context_frames + horizon, state_dim]
        forces = batch['forces'].to(device)
        context_frames = controller.model_args.get('context_frames', 2)

        # Current state is the last context frame
        current_state = states[:, context_frames - 1]  # [1, state_dim]
        future_states = states[:, context_frames:]  # [1, horizon, state_dim]

        # Get images if available
        images_cam1 = batch.get('images_cam1', None).to(device)
        images_cam2 = batch.get('images_cam2', None).to(device)

        if images_cam1 is not None and images_cam2 is not None:
            current_img_cam1 = images_cam1[:, -1]  # [1, H, W, 3]
            current_img_cam2 = images_cam2[:, -1]  # [1, H, W, 3]
        else:
            current_img_cam1 = None
            current_img_cam2 = None

        # Get forces if using them
        current_forces = forces[:, context_frames - 1]
        future_forces = forces[:, context_frames:]


        # Get VLA actions (source) and expert actions (target)
        vla_actions = batch['vla_actions'].to(device)  # [1, horizon, state_dim]
        expert_actions = batch['expert_actions'].to(device)  # [1, horizon, state_dim]

        # Run inference
        with torch.no_grad():
            # Generate action trajectory using the diffusion controller
            # take unnormalized action data as input, output unnormalized action
            predicted_actions = controller.predict(
                current_state,
                vla_actions,  # Using VLA as prior
                current_img_cam1,
                current_img_cam2,
                current_forces
            )

            # Calculate error metrics
            action_mse = torch.mean((predicted_actions - expert_actions) ** 2).item()
            vla_mse = torch.mean((vla_actions - expert_actions) ** 2).item()

            test_errors.append(action_mse)
            test_vla_errors.append(vla_mse)

            logger.info(f"Sample {sample_num + 1} - Action MSE: {action_mse:.4f} | VLA MSE: {vla_mse:.4f}")

            if visualize:
                # Create enhanced visualizations
                create_trajectory_visualizations(
                    expert_actions[0].cpu().numpy(),
                    vla_actions[0].cpu().numpy(),
                    predicted_actions[0].cpu().numpy(),
                    horizon,
                    os.path.join(save_dir, f"sample_{sample_num + 1}.png")
                )

    # Report average performance
    avg_error = sum(test_errors) / len(test_errors)
    avg_vla_error = sum(test_vla_errors) / len(test_vla_errors)
    improvement = (1.0 - avg_error / avg_vla_error) * 100 if avg_vla_error > 0 else 0

    logger.info(f"===== Test Results =====")
    logger.info(f"Average Action MSE: {avg_error:.4f}")
    logger.info(f"Average VLA MSE: {avg_vla_error:.4f}")
    logger.info(f"Improvement over VLA: {improvement:.2f}%")

    if visualize:
        logger.info(f"Visualizations saved to {save_dir}")

    return {
        'avg_error': avg_error,
        'avg_vla_error': avg_vla_error,
        'improvement': improvement,
        'test_errors': test_errors,
        'test_vla_errors': test_vla_errors
    }



def create_trajectory_visualizations(expert, vla, diffusion, horizon, save_path):
    """
    Create enhanced visualizations of trajectories.

    Args:
        expert: Expert actions [horizon, state_dim]
        vla: VLA actions [horizon, state_dim]
        diffusion: Diffusion model actions [horizon, state_dim]
        horizon: Number of timesteps
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 12))

    # 1. 3D trajectory plot (first 3 dimensions)
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(expert[:, 0], expert[:, 1], expert[:, 2], 'g-', linewidth=2, label='Expert')
    ax1.plot(vla[:, 0], vla[:, 1], vla[:, 2], 'r--', linewidth=2, label='VLA (Prior)')
    ax1.plot(diffusion[:, 0], diffusion[:, 1], diffusion[:, 2], 'b-', linewidth=2, label='Diffusion')

    # Add markers for start and end points
    ax1.scatter(expert[0, 0], expert[0, 1], expert[0, 2], c='g', s=100, marker='o', label='Start')
    ax1.scatter(expert[-1, 0], expert[-1, 1], expert[-1, 2], c='g', s=100, marker='*', label='End')

    ax1.set_xlabel('Dim 0', fontsize=12)
    ax1.set_ylabel('Dim 1', fontsize=12)
    ax1.set_zlabel('Dim 2', fontsize=12)
    ax1.set_title('3D Action Trajectories (First 3 Dimensions)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True)

    # 2. Error over time plot
    ax2 = fig.add_subplot(222)
    timesteps = list(range(horizon))

    # Calculate error at each timestep
    vla_errors = np.sqrt(np.sum((vla - expert) ** 2, axis=1))
    diff_errors = np.sqrt(np.sum((diffusion - expert) ** 2, axis=1))

    ax2.plot(timesteps, vla_errors, 'r-', linewidth=2, label='VLA Error')
    ax2.plot(timesteps, diff_errors, 'b-', linewidth=2, label='Diffusion Error')
    ax2.fill_between(timesteps, vla_errors, diff_errors, where=(diff_errors < vla_errors),
                     color='lightgreen', alpha=0.5, label='Improvement')
    ax2.fill_between(timesteps, diff_errors, vla_errors, where=(diff_errors > vla_errors),
                     color='lightcoral', alpha=0.5, label='Degradation')

    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Error (Euclidean Distance)', fontsize=12)
    ax2.set_title('Action Error Over Time', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True)

    # 3. Per-dimension error analysis
    ax3 = fig.add_subplot(223)

    # Calculate average error per dimension
    dim_count = min(10, expert.shape[1])  # Show at most the first 10 dimensions
    vla_dim_errors = np.mean(np.abs(vla[:, :dim_count] - expert[:, :dim_count]), axis=0)
    diff_dim_errors = np.mean(np.abs(diffusion[:, :dim_count] - expert[:, :dim_count]), axis=0)

    x = np.arange(dim_count)
    width = 0.35

    ax3.bar(x - width / 2, vla_dim_errors, width, label='VLA')
    ax3.bar(x + width / 2, diff_dim_errors, width, label='Diffusion')

    ax3.set_xlabel('Dimension', fontsize=12)
    ax3.set_ylabel('Average Absolute Error', fontsize=12)
    ax3.set_title('Error Analysis by Dimension', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, axis='y')
    ax3.set_xticks(x)

    # 4. Prediction vs Expert (selected dimensions over time)
    ax4 = fig.add_subplot(224)
    dims_to_plot = min(3, expert.shape[1])  # Plot first 3 dimensions

    # Line styles and colors for different dimensions
    styles = ['-', '--', '-.']
    colors = ['g', 'r', 'b']

    for d in range(dims_to_plot):
        ax4.plot(timesteps, expert[:, d], f'{colors[0]}{styles[d]}',
                 linewidth=2, label=f'Expert Dim {d}')
        ax4.plot(timesteps, diffusion[:, d], f'{colors[2]}{styles[d]}',
                 linewidth=2, label=f'Diffusion Dim {d}')

    ax4.set_xlabel('Timestep', fontsize=12)
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Trajectory Comparison (First 3 Dimensions)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def test(epoch=None):

    use_force = True
    task = 'wipe'
    if epoch:
        model = f"epoch_{epoch}"
    else:
        model = 'best_model'
    checkpoint_dir = f"residual_controller/checkpoint/bridge_controller/{task}_0421_1945/{model}"
    data_dir = f"./data/datasets/{task}_controller_40000"
    result = test_diffusion_controller(
        checkpoint_dir=checkpoint_dir,
        data_dir = data_dir,
        num_samples = 50,
        visualize=False,
    )

    return result

if __name__ == "__main__":


    imp = []

    for epoch in [300]:
        result = test(epoch)
        imp.append(result['improvement'])

    result = test()
    imp.append(result['improvement'])
    print(imp)



