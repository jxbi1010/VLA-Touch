#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import sys
import os
import threading
import time
import termios
import tty
import yaml
from collections import deque
from math import acos
import numpy as np
import rospy
import torch

from cv_bridge import CvBridge
from PIL import Image as PImage
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Header, String
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.franka_model_eef import create_model
from utils_eef import *
from residual_controller.lstm_step_controller import load_lstm_controller
from residual_controller.bridge_controller import load_bridge_controller

from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output as GripperOutput
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_input as GripperInput
from controller_dataset import normalize_actions, denormalize_actions
# from pyquaternion import Quaternion

import select

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

observation_window = None
lang_embeddings = None
preload_images = None

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


# Initialize the model (unchanged)
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config

    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 10,
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config


def jpeg_mapping(img):
    img = cv2.imencode('.jpg', img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    return img


# Get the observation from the ROS topic
def get_ros_observation(args, ros_operator):
    delay = 1.0 / args.publish_rate  # Use the desired frequency (e.g., from args)
    print_flag = True

    while not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            time.sleep(delay)
            continue
        print_flag = True
        (img_front, img_right, qpos, gelsight) = result

        # print(f"sync success when get_ros_observation")
        return (img_front, img_right, qpos, gelsight)


# Update the observation window buffer
def update_observation_window(args, config, ros_operator):
    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)

        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'images':
                    {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                        config["camera_names"][2]: None,
                    },
                'gelsight': None
            }
        )

    img_front, img_right, qpos, gelsight = get_ros_observation(args, ros_operator)
    img_front = jpeg_mapping(img_front)
    img_right = jpeg_mapping(img_right)

    qpos = torch.from_numpy(qpos).float().cuda()
    gelsight = torch.from_numpy(gelsight).float().cuda()
    observation_window.append(
        {
            'qpos': qpos,
            'images':
                {
                    config["camera_names"][0]: img_front,
                    config["camera_names"][1]: img_right,
                    config["camera_names"][2]: None,
                },
            'gelsight': gelsight
        }
    )


# RDT inference
def inference_fn(config, policy):
    global observation_window
    global lang_embeddings
    global lstm_states
    lstm_states = None

    # print(f"Start inference_thread_fn: t={t}")
    while not rospy.is_shutdown():
        # fetch images in sequence [front, right]
        image_arrs = [
            observation_window[-2]['images'][config['camera_names'][0]],
            observation_window[-2]['images'][config['camera_names'][1]],
            observation_window[-2]['images'][config['camera_names'][2]],

            observation_window[-1]['images'][config['camera_names'][0]],
            observation_window[-1]['images'][config['camera_names'][1]],
            observation_window[-1]['images'][config['camera_names'][2]]
        ]

        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]

        # get last qpos in shape [10, ]
        proprio = observation_window[-1]['qpos']
        proprio = proprio.unsqueeze(0)

        # gelsight = observation_window[-1]['gelsight']

        # actions shaped as [1, 64, 10]
        vla_tensor = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings
        )
        vla_actions = vla_tensor.squeeze(0).cpu().numpy()

        return vla_actions, vla_tensor


class RosOperator:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()

        # Initialize deques
        self.img_right_deque = deque(maxlen=2000)
        self.img_front_deque = deque(maxlen=2000)

        self.eef_pose_deque = deque(maxlen=2000)
        self.gripper_state_deque = deque(maxlen=2000)
        self.gelsight_state_deque = deque(maxlen=2000)

        self.arm_publish_lock = threading.Lock()
        self.arm_publish_thread = None
        self.init_ros()
        self.init_gripper()

    def init_ros(self):
        # Publishers
        self.arm_right_publisher = rospy.Publisher(self.args.arm_right_topic, PoseStamped, queue_size=10)
        self.gripper_publisher = rospy.Publisher(self.args.gripper_cmd_topic, GripperOutput, queue_size=10)

        # Subscribers
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=10)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=10)

        rospy.Subscriber(self.args.arm_eef_right_topic, PoseStamped, self.eef_pose_callback, queue_size=10)
        rospy.Subscriber(self.args.gripper_state_topic, GripperInput, self.gripper_state_callback, queue_size=10)
        rospy.Subscriber(self.args.gelsight_state_topic, Twist, self.gelsight_state_callback, queue_size=10)

    def init_gripper(self):
        """Initialize the Robotiq gripper"""
        command = GripperOutput()
        command.rACT = 1  # Activate gripper
        command.rGTO = 100  # Go to position
        command.rSP = 20  # Speed (255 is maximum)
        command.rFR = 10  # Force (150 is a good default)
        command.rPR = 3
        self.gripper_publisher.publish(command)
        print("Initialize Gripper")
        rospy.sleep(1.0)  # Wait for activation

    def arm_publish(self, right):
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"

        if len(right) == 10:
            # convert 9D eef pose to 7D
            eef_pose_7D = ee_pose_9D_to_7D(right[:9])
            gripper_pos_raw = right[-1]
        else:
            eef_pose_7D = right[:7]
            gripper_pos_raw = right[-1]

        target_gripper_pos = int(np.clip(gripper_pos_raw, 10, 240))  # Position (255 is fully closed)
        current_gripper_state = self.gripper_state_deque[-1]

        # Parameters
        alpha = 1  # Smoothing factor (0-1)
        deadband = 2  # No movement within this range of target

        # Calculate filtered target (smooths oscillating inputs)
        filtered_target = current_gripper_state + alpha * (target_gripper_pos - current_gripper_state)

        # Apply deadband to prevent tiny movements
        if abs(filtered_target - current_gripper_state) < deadband:
            gripper_pos = current_gripper_state
        else:
            gripper_pos = filtered_target

        # print(f"gripper_pos_raw:{gripper_pos_raw},gripper_pos:{gripper_pos}")

        # Set position (first 3 elements of 'right')
        pose_msg.pose.position.x = float(eef_pose_7D[0])
        pose_msg.pose.position.y = float(eef_pose_7D[1])
        pose_msg.pose.position.z = float(eef_pose_7D[2])

        # Set orientation as quaternion (next 4 elements of 'right')
        pose_msg.pose.orientation.x = float(eef_pose_7D[3])
        pose_msg.pose.orientation.y = float(eef_pose_7D[4])
        pose_msg.pose.orientation.z = float(eef_pose_7D[5])
        pose_msg.pose.orientation.w = float(eef_pose_7D[6])

        self.arm_right_publisher.publish(pose_msg)

        gripper_cmd_msg = GripperOutput()
        gripper_cmd_msg.rACT = 1  # Gripper activated
        gripper_cmd_msg.rGTO = 1  # Go to position
        gripper_cmd_msg.rSP = 20  # Speed
        gripper_cmd_msg.rFR = 20  # Force
        gripper_cmd_msg.rPR = int(gripper_pos)

        self.gripper_publisher.publish(gripper_cmd_msg)

    def img_right_callback(self, msg):
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        self.img_front_deque.append(msg)

    def gripper_state_callback(self, msg):
        self.gripper_state_deque.append(msg.gPO)
        # self.gripper_position = msg.gPO  # Current position
        # self.gripper_status = msg.gSTA   # Gripper status
        # self.object_detected = msg.gOBJ  # Object detection status

    def gelsight_state_callback(self, msg):
        # Process the Gelsight state message here
        force_tripplet = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
        self.gelsight_state_deque.append(force_tripplet)

    def eef_pose_callback(self, msg):

        # append eef pos and quaternion
        self.eef_pose_deque.append(np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        ]))

    def get_frame(self):
        if len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0:
            return False

        # Get latest frames
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'bgr8')
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'bgr8')

        img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

        # Resize and pad images for SigLIP
        img_front_resized = pad_and_resize_for_siglip(img_front)
        img_right_resized = pad_and_resize_for_siglip(img_right)

        # Get latest arm and gripper state
        arm_right = self.eef_pose_deque[-1]
        gripper_state = self.gripper_state_deque[-1]
        gelsight_state = self.gelsight_state_deque[-1]
        # print(f"[Get frame] Gripper state: {gripper_state}")

        # Convert arm to 9D pose
        arm_right = ee_pose_7D_to_9D(arm_right)
        qpos = np.concatenate((arm_right, [gripper_state]))

        return (img_front_resized, img_right_resized, qpos, gelsight_state)


def model_inference(args, config, ros_operator):
    global lang_embeddings
    lstm_states = None

    import threading
    import sys
    import select
    import os
    import time

    # Flag to control execution
    paused = False
    should_exit = False

    # Terminal settings for Linux/Mac
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def keyboard_monitor():
        """Monitor keyboard input in a separate thread."""
        nonlocal paused, should_exit, old_settings, fd

        # Print instructions for the user
        print("\n--- Keyboard Controls ---")
        print("Press 'p' to pause execution and choose a new instruction")
        print("Press 'q' to quit")
        print("------------------------\n")

        try:
            tty.setraw(fd, termios.TCSANOW)

            while not should_exit:
                # Try to read a character if available (non-blocking)
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    key = sys.stdin.read(1).lower()
                    if key == 'p':
                        # Need to restore terminal settings before pausing
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        paused = True  # Only set to pause
                        print("\nPausing execution...")
                    elif key == 'q':
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        print("\nQuitting...")
                        should_exit = True

                # If we're paused or exiting, don't try to read more keys in raw mode
                if paused or should_exit:
                    break

                # Sleep to avoid high CPU usage
                time.sleep(0.1)

        except Exception as e:
            # Restore terminal settings before raising any exception
            if old_settings:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print(f"\nError in keyboard monitor: {e}")
            should_exit = True

    # Load the controller if specified
    controller = None
    if args.use_controller:
        if args.use_controller =='bridge':
            controller = load_bridge_controller(args.controller_path)
        elif args.use_controller =='lstm':
            controller = load_lstm_controller(args.controller_path)
        controller.load(args.controller_path)
        print(f"[LOADING CONTROLLER]: {args.controller_path}")

    policy = make_policy(args)

    lang_dict = torch.load(args.lang_embeddings_path)
    all_instructions_list = lang_dict['all_instructions']

    print("Select the instruction to perform:")
    for i in range(len(all_instructions_list)):
        print(f"{i}. {all_instructions_list[i]}")

    instruct_index = int(input("Enter the number of your choice: "))
    lang_embeddings = lang_dict[lang_dict['all_instructions'][instruct_index]]

    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    # Initial position of the franka arm
    # right1 = np.array([0.5190, 0.2467, 0.4810, -0.2464, 0.2491, -0.1436, -0.1400, 100]) # cup task init pose
    right1 = np.array([0.484, 0.445, 0.433, 0.654, -0.624, 0.309, 0.293, 100])  # watercup
    input("Press enter to initial position...")
    ros_operator.arm_publish(right1)
    input("Press enter to continue to init position")
    ros_operator.arm_publish(right1)
    input("Press enter to start inference")

    # # Start keyboard monitoring thread
    keyboard_thread = threading.Thread(target=keyboard_monitor, daemon=True)
    keyboard_thread.start()

    print("Execution started. Press 'p' to pause and select a new instruction, 'q' to quit.")

    with torch.inference_mode():
        while not rospy.is_shutdown() and not should_exit:
            t = 0
            rate = rospy.Rate(args.publish_rate)
            vla_action_buffer = np.zeros([chunk_size, config['state_dim']])
            vla_tensor = None

            # Reset LSTM states at the beginning of each episode
            if args.use_controller == 'lstm' and controller is not None:
                controller.reset_state(batch_size=1)

            while t < max_publish_step and not rospy.is_shutdown() and not should_exit:
                # Check if execution is paused
                if paused and not should_exit:
                    # Terminal is already restored in the keyboard monitor thread
                    print("\n\nExecution paused. Choose a new instruction to continue.")

                    # Allow user to select a new instruction
                    print("Select the instruction to perform:")
                    for i in range(len(all_instructions_list)):
                        print(f"{i}. {all_instructions_list[i]}")

                    # Input should work normally now since terminal settings are restored
                    instruct_index = int(input("Enter the number of your choice: "))
                    lang_embeddings = lang_dict[lang_dict['all_instructions'][instruct_index]]

                    print("Updating observation window...")
                    update_observation_window(args, config, ros_operator)

                    print(f"Resuming execution with new instruction: {all_instructions_list[instruct_index]}")
                    paused = False

                    # Get a new VLA action immediately
                    vla_action_buffer, vla_tensor = inference_fn(config, policy)

                    # Reset LSTM states when changing instructions
                    if args.use_controller == 'lstm' and controller is not None:
                        controller.reset_state(batch_size=1)

                    # Restart the keyboard monitor thread with raw mode
                    if not should_exit:  # Not Windows and not exiting
                        keyboard_thread.join(timeout=0.5)  # Wait for previous thread to end
                        keyboard_thread = threading.Thread(target=keyboard_monitor, daemon=True)
                        keyboard_thread.start()

                # Continue waiting if still paused
                while paused and not should_exit:
                    time.sleep(0.1)

                # If we should exit, break out
                if should_exit:
                    break

                update_observation_window(args, config, ros_operator)

                act_chunk_execute_step = 16
                if t % act_chunk_execute_step == 0:
                    print("################### Inferenece new VLA action chunk #######################")
                    vla_action_buffer, vla_tensor = inference_fn(config, policy)

                vla_act = vla_action_buffer[t % chunk_size]

                if args.controller:

                    # Apply residual controller at higher frequency if enabled
                    current_state = observation_window[-1]['qpos'].unsqueeze(0)
                    front_image = observation_window[-1]['images'][config['camera_names'][0]]
                    right_image = observation_window[-1]['images'][config['camera_names'][1]]

                    img_cam1 = torch.from_numpy(np.array(front_image)).unsqueeze(0).unsqueeze(0)
                    img_cam2 = torch.from_numpy(np.array(right_image)).unsqueeze(0).unsqueeze(0)
                    gelsight = observation_window[-1]['gelsight'].unsqueeze(0)

                    vla_tensor[:, :, -1] /= 255
                    if args.use_controller == 'bridge':
                        with torch.no_grad():

                            refined_trajectory = controller.predict(current_state,
                                                                    vla_tensor,
                                                                    img_cam1,
                                                                    img_cam2,
                                                                    gelsight)
                            refined_actions = refined_trajectory.squeeze(0).cpu().numpy()
                            refined_actions[:,-1] *=255

                            for k in range(0, act_chunk_execute_step):
                                # Check again if paused or should exit
                                if paused or should_exit:
                                    break

                                refined_act = refined_actions[t]
                                ros_operator.arm_publish(refined_act)
                                rate.sleep()
                                t += 1

                    elif args.use_controller=='lstm':

                        controller_execute_step = 16
                        with torch.no_grad():

                            obs_cond = controller.encode_observation(current_state, img_cam1, img_cam2)

                            vla_tensors_n = normalize_actions(vla_tensor, controller.stats, 'vla')

                            for k in range(0, controller_execute_step):
                                if paused or should_exit:
                                    break

                                update_observation_window(args, config, ros_operator)
                                gelsight = observation_window[-1]['gelsight'].unsqueeze(0)

                                refined_action = controller.predict(
                                    obs_cond=obs_cond,
                                    vla_action=vla_tensors_n[:,k],
                                    force=gelsight,
                                    initialize=(t == 0)  # Only initialize on first step
                                )
                                refined_action_np = refined_action.squeeze(0).cpu().numpy()

                                refined_action_np[-1] *=255
                                # Execute the refined action
                                ros_operator.arm_publish(refined_action_np)
                                rate.sleep()
                                t += 1

                else:
                    # Check if paused
                    if not paused:
                        print("Executing VLA action step:", t)
                        ros_operator.arm_publish(vla_act)
                        rate.sleep()
                        t += 1

    print("Execution finished")
    # Make sure we clean up the keyboard thread
    should_exit = True
    keyboard_thread.join(timeout=1.0)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int,
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int,
                        help='Random seed', default=None, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera1/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera2/image_raw', required=False)

    parser.add_argument('--arm_right_topic', action='store', type=str, help='arm_right_topic',
                        default='/desired_pose', required=False)
    parser.add_argument('--arm_eef_right_topic', action='store', type=str, help='arm_eef_right_topic',
                        default='/end_effector_pose', required=False)

    parser.add_argument('--gripper_state_topic', action='store', type=str, help='gripper_state_topic',
                        default='/Robotiq2FGripperRobotInput', required=False)
    parser.add_argument('--gripper_cmd_topic', action='store', type=str, help='gripper_cmd_topic',
                        default='/Robotiq2FGripperRobotOutput', required=False)

    parser.add_argument('--gelsight_state_topic', action='store', type=str, help='gelsight_state_topic',
                        default='/marker_tracker_node/marker_state', required=False)

    parser.add_argument('--publish_rate', action='store', type=int,
                        help='The rate at which to publish the actions',
                        default=6, required=False)
    parser.add_argument('--ctrl_freq', action='store', type=int,
                        help='The control frequency of the robot',
                        default=10, required=False)

    parser.add_argument('--chunk_size', action='store', type=int,
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float,
                        help='The maximum change allowed for eef pos per timestep',
                        default=[0.05, 0.05, 0.05], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true',
                        help='Whether to use depth images',
                        default=False, required=False)

    parser.add_argument('--disable_arm', action='store_true',
                        help='Whether to disable the arm. This is useful for safely debugging', default=False)

    parser.add_argument('--config_path', type=str, default="./configs/base.yaml",
                        help='Path to the config file')

    parser.add_argument('--pretrained_model_name_or_path', type=str, required=False, default=None,
                        help='Name or path to the pretrained model')

    parser.add_argument('--lang_embeddings_path', type=str, required=False, default="../outs/instruction_embeddings.pt",
                        help='Path to the pre-encoded language instruction embeddings')


    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    rospy.init_node('eef_publisher', anonymous=True)
    ros_operator = RosOperator(args)

    task_name = 'watercup'  # wipe, mango

    task_path = "./checkpoints/rdt-finetune-1b-watercup-rgb"
    ckpt_path = os.path.join(task_path, "checkpoint-20000")
    lang_embedding_path = os.path.join(task_path, "all_instruction_embeddings.pt")

    args.use_controller = None  # 'lstm', 'bridge'
    if args.use_controller:
        args.controller_path = f"./residual_controller/checkpoint/{args.use_controller}_controller/{task_name}/best_model"

    args.pretrained_model_name_or_path = ckpt_path
    args.lang_embeddings_path = lang_embedding_path

    if args.seed is not None:
        set_seed(args.seed)

    config = get_config(args)

    try:
        inference_thread = threading.Thread(target=model_inference, args=(args, config, ros_operator))
        inference_thread.start()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        inference_thread.join()


def test():
    args = get_arguments()
    rospy.init_node('joint_state_publisher', anonymous=True)
    ros_operator = RosOperator(args)

    task_path = "./checkpoints/rdt-finetune-1b-cup"
    lang_embedding_path = os.path.join(task_path, "all_instruction_embeddings.pt")
    lang_embedding = torch.load(lang_embedding_path)
    print(lang_embedding['all_instructions'])

    result = ros_operator.get_frame()
    for i in result:
        print(i.shape)

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
    # test()

