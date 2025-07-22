import numpy as np
import cv2
from docs.test_6drot import *

def pad_and_resize_for_siglip_batch(images, target_size=384):
    """
    First pad images to make them square, then resize to the target size.
    This approach preserves more information from the original image.

    Parameters:
    - images: numpy array of shape (batch, height, width, channels)
    - target_size: the target size for SigLIP (default: 384)

    Returns:
    - Processed images of shape (batch, target_size, target_size, channels)
    """
    batch_size, height, width, channels = images.shape
    processed_images = np.zeros((batch_size, target_size, target_size, channels), dtype=images.dtype)

    for i in range(batch_size):
        img = images[i]

        # Find the larger dimension to create a square
        max_dim = max(height, width)

        # Create a square canvas filled with zeros
        square_img = np.zeros((max_dim, max_dim, channels), dtype=img.dtype)

        # Calculate padding to center the image
        pad_height = (max_dim - height) // 2
        pad_width = (max_dim - width) // 2

        # Place the original image in the center of the square canvas
        square_img[pad_height:pad_height + height, pad_width:pad_width + width, :] = img

        # Resize the square image to the target size
        resized_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        processed_images[i] = resized_img

    return processed_images


def pad_and_resize_for_siglip(image, target_size=384):
    """
    First pad image to make it square, then resize to the target size.
    This approach preserves more information from the original image.

    Parameters:
    - image: numpy array of shape (height, width, channels)
    - target_size: the target size for SigLIP (default: 384)

    Returns:
    - Processed image of shape (target_size, target_size, channels)
    """
    if image is None:
        return None
    else:
        height, width, channels = image.shape

        # Find the larger dimension to create a square
        max_dim = max(height, width)

        # Create a square canvas filled with zeros
        square_img = np.zeros((max_dim, max_dim, channels), dtype=image.dtype)

        # Calculate padding to center the image
        pad_height = (max_dim - height) // 2
        pad_width = (max_dim - width) // 2

        # Place the original image in the center of the square canvas
        square_img[pad_height:pad_height + height, pad_width:pad_width + width, :] = image

        # Resize the square image to the target size
        resized_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return resized_img


def converted_ee_pose_with_gripper(epi_data):
    # [ee_pos, ee_6d_ori, gripper_open] #(batch, 10)
    ee_pos = epi_data['ee_poses'][:, :3]
    ee_ori = epi_data['ee_poses'][:, 3:]
    ee_6d = convert_quaternion_to_orthod6d(ee_ori)

    grip_pos = epi_data['gripper_pos'][:].reshape(-1, 1)

    qpos = np.concatenate((ee_pos, ee_6d, grip_pos), axis=-1)

    return qpos


def convert_quaternion_to_orthod6d(quat):
    if len(quat.shape)==1:
        quat = quat[None,:]
    euler = convert_quaternion_to_euler(quat)
    rotmat = convert_euler_to_rotation_matrix(euler)
    ortho6d = compute_ortho6d_from_rotation_matrix(rotmat)

    return ortho6d[0] if len(quat.shape)==1 else ortho6d


def convert_orthod6d_to_quaternion(ortho6d):
    ortho6d = ortho6d[None,:]
    rotmat = compute_rotation_matrix_from_ortho6d(ortho6d)
    euler = convert_rotation_matrix_to_euler(rotmat)
    quat = convert_euler_to_quaternion(euler)

    return quat[0]


def ee_pose_9D_to_7D(ee_9d):
    pos =  ee_9d[:3]
    quat_4d = convert_orthod6d_to_quaternion(ee_9d[3:])

    return  np.concatenate((pos,quat_4d),axis=-1)

def ee_pose_7D_to_9D(ee_6d):
    pos =  ee_6d[:3]
    quat_6d = convert_quaternion_to_orthod6d(ee_6d[3:])
    return np.concatenate((pos, quat_6d), axis=-1)

