import os
import torch
import numpy as np
import logging
from tqdm import tqdm
from controller_dataset import ControllerDataModule,normalize_actions,denormalize_actions
from lstm_step_controller import TactileLSTMController
from bridge_test import create_trajectory_visualizations  # reuse your viz func


def test_lstm_controller(
        checkpoint_dir,
        data_dir=None,
        data_loader=None,
        context_frames=2,
        horizon=32,
        image_size=384,
        device='cuda',
        num_samples=10,
        state_dim=10,
        hidden_dim=256,
        force_dim=3,
        use_force=False,
        image_model_path="facebook/dinov2-small",
        visualize=True,
        save_dir=None
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_lstm_controller")
    logger.info(f"Testing LSTM controller from checkpoint: {checkpoint_dir}")

    # Load data
    if data_loader:
        test_dataloader = data_loader
    else:
        data_module = ControllerDataModule(
            data_dir=data_dir,
            batch_size=32,
            num_workers=4,
            context_frames=context_frames,
            horizon=horizon,
            image_size=image_size,
            val_ratio=0.2,
            stride=1
        )
        test_dataloader = data_module.val_dataloader()

    # Init model and load
    controller = TactileLSTMController(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        image_model_path=image_model_path,
        device=device,
        use_force=use_force,
        force_dim=force_dim
    )
    controller.load(checkpoint_dir)
    controller.eval()

    if visualize:
        if save_dir is None:
            save_dir = os.path.join(checkpoint_dir, "test_visualizations")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Visualizations will be saved to {save_dir}")

    # Prepare sampling
    import random
    sample_indices = []
    total_samples = 0
    for i, batch in enumerate(test_dataloader):
        batch_size = batch['states'].shape[0]
        total_samples += batch_size
        for j in range(batch_size):
            sample_indices.append((i, j))

    logger.info(f"Total available test samples: {total_samples}")
    chosen_indices = random.sample(range(len(sample_indices)), min(num_samples, len(sample_indices)))
    batches_to_load = set(sample_indices[i][0] for i in chosen_indices)
    loaded_batches = {}

    for i, batch in enumerate(test_dataloader):
        if i in batches_to_load:
            loaded_batches[i] = batch

    # Run selected test samples
    test_errors, test_vla_errors = [], []

    for sample_num, idx in enumerate(chosen_indices):
        batch_idx, within_batch_idx = sample_indices[idx]
        batch = loaded_batches[batch_idx]

        logger.info(f"Testing sample {sample_num + 1}/{len(chosen_indices)} (batch {batch_idx}, idx {within_batch_idx})")


        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)


        # Extract single sample
        # sample = {}
        # for k in batch:
        #     if isinstance(batch[k], torch.Tensor):
        #         sample[k] = batch[k][within_batch_idx:within_batch_idx + 1].to(device)
        #     else:
        #         sample[k] = batch[k]
        # Get last context frame state and images
        current_state = batch['states'][:, context_frames - 1].to(device)

        # Get forces sequence - should be aligned with prediction horizon
        forces = batch['forces'][:, context_frames - 1:context_frames - 1 + horizon].to(device)

        # Get VLA and expert actions for the prediction horizon
        vla_actions = batch['vla_actions'][:, :horizon].to(device)
        # Normalize actions for model input
        # vla_actions_n = normalize_actions(vla_actions, controller.stats, 'vla')

        # Get images from the last context frame
        cam1 = batch['images_cam1'][:, context_frames - 1].to(device)
        cam2 = batch['images_cam2'][:, context_frames - 1].to(device)

        # Encode the observation (state + images)
        obs_cond = controller.encode_observation(
            current_state, cam1, cam2
        )

        vla_actions_n = normalize_actions(batch['vla_actions'], controller.stats, 'vla')
        expert_actions_n = normalize_actions(batch['expert_actions'],controller.stats, 'expert')

        batch_dict = {
            'obs_cond': obs_cond,
            'expert_act': expert_actions_n,  # Target expert action
            'vla_act': vla_actions_n,  # Source VLA action
            'forces': forces,
        }


        with torch.no_grad():
            # Predict action sequence
            predicted_actions = controller.predict_sequence(
                obs_cond=obs_cond,
                vla_actions=vla_actions,
                force_seq=forces
            )

            # predicted_actions_f = controller.forward(batch_dict)
            # predicted_actions_denorm = denormalize_actions(predicted_actions_f, controller.stats,'expert')

        # Evaluate
        vla_actions = batch['vla_actions'].to(device)
        expert_actions = batch['expert_actions'].to(device)

        action_mse = torch.mean((predicted_actions - expert_actions) ** 2).item()
        vla_mse = torch.mean((vla_actions - expert_actions) ** 2).item()

        test_errors.append(action_mse)
        test_vla_errors.append(vla_mse)
        logger.info(f"Sample {sample_num+1}: LSTM MSE: {action_mse:.4f} | VLA MSE: {vla_mse:.4f}")

        if visualize:
            create_trajectory_visualizations(
                expert_actions[0].cpu().numpy(),
                vla_actions[0].cpu().numpy(),
                predicted_actions[0].cpu().numpy(),
                horizon,
                os.path.join(save_dir, f"sample_{sample_num + 1}.png")
            )

    avg_error = np.mean(test_errors)
    avg_vla_error = np.mean(test_vla_errors)
    improvement = (1 - avg_error / avg_vla_error) * 100 if avg_vla_error > 0 else 0

    logger.info(f"===== LSTM Test Results =====")
    logger.info(f"Average LSTM MSE: {avg_error:.4f}")
    logger.info(f"Average VLA MSE: {avg_vla_error:.4f}")
    logger.info(f"Improvement over VLA: {improvement:.2f}%")

    return {
        'avg_error': avg_error,
        'avg_vla_error': avg_vla_error,
        'improvement': improvement,
        'test_errors': test_errors,
        'test_vla_errors': test_vla_errors
    }


def test(epoch=None):
    use_force = True
    if epoch:
        model = f"epoch_{epoch}"
    else:
        model = "best_model"

    task = 'wipe'
    checkpoint_dir = f"residual_controller/checkpoint/lstm_step_controller/{task}_0423_2218/{model}"
    data_dir = f"./data/datasets/{task}_controller_40000"
    result = test_lstm_controller(
        checkpoint_dir=checkpoint_dir,
        data_dir=data_dir,
        num_samples=50,
        use_force=use_force,
        visualize= False
    )
    return result


if __name__ == "__main__":

    imp = []
    # for epoch in [100,150,200]:
    #     result = test(epoch)
    #     imp.append(result['improvement'])

    result = test()
    imp.append(result['improvement'])
    print(imp)
