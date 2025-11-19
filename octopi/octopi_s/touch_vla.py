from openai import OpenAI
import json
import os
from datetime import datetime
import base64
import numpy as np
import re
import argparse


client = OpenAI(api_key="your api key")
# Set your OpenAI API key
MODEL = "gpt-4o"

# Experiment configurations
EXPERIMENTS = {
    "wipe": {
        "prompt": "There are two sponges in the image, one on the left and another on the right. Step by step, touch and identify the roughness of the sponges, and pick up the smoother one.",
        "dataset_base_path": "/users/kevinma/tactile/dataset/wipe",
        "tactile_analysis_prompt": "Analyze the images to figure out the smoothness of the object.",
        "output_file": "results/wipe_results.jsonl"
    },
    "mango": {
        "prompt": "There are two mangoes in the image, one on the left and another on the right. Step by step, touch and identify the properties of the mangoes, and pick up the riper one.",
        "dataset_base_path": "/users/kevinma/tactile/dataset/mango_touch",
        "tactile_analysis_prompt": "Analyze the images to figure out the hardness of the object.",
        "output_file": "results/mango_results.jsonl"
    },
    "cup": {
        "prompt": "There is a cup in the image. Step by step, identify whether the cup is full or empty. If it is not empty, put it onto the plate.",
        "dataset_base_path": "/users/kevinma/tactile/dataset/water_cup_new",
        "tactile_analysis_prompt": None,
        "force_reference": "For reference, the max force magnitude is around 0.55 for an empty cup and around 1.1 for a full cup.",
        "output_file": "results/cup_results.jsonl"
    }
}

Conditioning = ("You are a robot highly skilled in robotic task planning and interactive reasoning, adept at planning actions to retrieve unknown information for reasoning and decision making, "
                "and subsequently breaking down intricate and long-term tasks into distinct primitive actions. "
                "As a robot, you have one arm with a gripper. You have a tactile sensor mounted on the gripper, which can be used to classify physical properties (hardness, roughness, weight) of objects. "
                "Your task is to plan out steps of actions to take to retrieve information and complete the task. For each time, return "
                "1. a primitive action in the form of one sentence, it should contain one elemental robot action interacting with at most one object "
                "2. information needed to retrieve if applicable. "
                "After each action, feedback will be given back to you for information retrieval or action execution, you will then plan the next robot action based on the feedback. "
                "The feedback can be in the form of property classification results or raw tactile sensor images. Remember to refer to objects by their spatial locations (like left or right). "
                "Only give physical actions that the robot has to execute. Keep the action and information needed concise. Only give one action step in each response and wait for the user feedback.")

def find_first_last_imgs_octopi(directory):
    # print(directory)
    all_files = os.listdir(directory)
    all_files.sort()

    first_path = os.path.join(directory, all_files[0])
    last_path = os.path.join(directory, all_files[-1])

    return first_path, last_path

def find_first_last_imgs(directory):
    """
    Find gel image files with the smallest and largest numbers in their filenames.
    Returns paths to both files.
    """
    try:
        # Get all files in the directory
        all_files = os.listdir(directory)
        
        # Filter files that match the pattern gel_[number].jpg
        gel_files = [f for f in all_files if re.match(r'^gel_\d+\.jpg$', f)]
        # gel_files = [f for f in all_files if re.match(r'^\d+\.jpg', f)]
        
        if not gel_files:
            print("No files matching the pattern gel_[number].jpg were found.")
            return None
        
        # Extract numbers from filenames and create pairs of (number, filename)
        number_file_pairs = []
        for filename in gel_files:
            match = re.match(r'^gel_(\d+)\.jpg$', filename)
            # match = re.match(r'^\d+\.jpg', filename)
            number = int(match.group(1))
            number_file_pairs.append((number, filename))
        
        # Find files with smallest and largest numbers
        smallest_pair = min(number_file_pairs, key=lambda x: x[0])
        largest_pair = max(number_file_pairs, key=lambda x: x[0])
        
        # Get full paths
        first_path = os.path.join(directory, smallest_pair[1])
        last_path = os.path.join(directory, largest_pair[1])
        
        # print(f"Smallest number: {smallest_pair[0]}, Path: {smallest_path}")
        # print(f"Largest number: {largest_pair[0]}, Path: {largest_path}")
        
        return first_path, last_path
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def send_message(messages, image_path=None):
    """Send a list of messages to ChatGPT, optionally with an image."""
    response = client.chat.completions.create(model=MODEL,
    messages=messages,
    temperature=0.2,
    max_tokens=500)
    return response.choices[0].message.content

def get_user_feedback(experiment):
    """Get manual user feedback via terminal input."""
    exp_config = EXPERIMENTS[experiment]
    
    print("\n" + "="*60)
    print("Available inputs:")
    print("  'end' - End the session")
    
    if experiment == "cup":
        print("  'force:<episode_num>' - Get force magnitude data (e.g., 'force:5')")
        print("  '<image_path>' - Provide path to tactile images")
    else:
        print("  '<episode_num>' - Get tactile images for episode (e.g., '5')")
        print("  '<hardness>,<roughness>' - Manually input property values (e.g., '5.2,8.3')")
        print("  '<image_path>' - Provide path to tactile images")
    print("="*60)
    
    user_input = input("\nYour input: ").strip()
    return user_input

def run_interactive_session(initial_prompt, experiment):
    """Run one interactive session with manual input."""
    exp_config = EXPERIMENTS[experiment]
    
    # Get initial image path
    image_path = input(f"\nEnter initial image path: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found {image_path}")
        return
    
    base64_image = encode_image(image_path)
    messages = [
        {"role": "system", "content": Conditioning},
        {"role": "user", "content": [
            { "type": "text", "text": initial_prompt },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            },
        ],},
    ]
    
    session_log = {
        "experiment": experiment,
        "start_time": str(datetime.now()),
        "initial_image": image_path,
        "initial_prompt": initial_prompt,
        "steps": []
    }

    print(f"\n{'='*60}")
    print(f"Starting Interactive Session - {experiment.upper()}")
    print(f"Initial Image: {image_path}")
    print(f"{'='*60}")
    
    step = 0
    while True:
        response = send_message(messages)
        print(f"\n[Step {step + 1}] Assistant:", response)

        session_log["steps"].append({
            "step": step + 1,
            "assistant": response
        })

        # Get user input
        user_input = get_user_feedback(experiment)

        step += 1

        if user_input.lower() == "end":
            print(f"\nEnding session.")
            break
        
        # Handle episode number input
        elif user_input.isdigit():
            # User provided episode number
            tactile_episode = int(user_input)
            base_path = exp_config["dataset_base_path"]
            tactile_path = f"{base_path}/episode_{tactile_episode}/gelsight"
            
            if not os.path.exists(tactile_path):
                print(f"Warning: Tactile path {tactile_path} not found")
                continue
            
            first_tactile, last_tactile = find_first_last_imgs_octopi(tactile_path)
            base64_image_0 = encode_image(first_tactile)
            base64_image_1 = encode_image(last_tactile)
            
            # Build message with tactile images
            content = [
                {"type": "text", "text": "The first image is the gelsight tactile sensor image before touching the object"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_0}"}},
                {"type": "text", "text": "The second image is the gelsight tactile sensor image after touching the object"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_1}"}}
            ]
            
            if exp_config["tactile_analysis_prompt"]:
                content.append({"type": "text", "text": exp_config["tactile_analysis_prompt"]})
            
            message = {"role": "user", "content": content}
            session_log["steps"][-1]["user_feedback"] = f"Tactile images: {first_tactile}, {last_tactile}"
        
        elif user_input.lower().startswith("force"):
            # Handle force magnitude input for cup experiment
            # Format: "force:episode_num" or just "force"
            if ":" in user_input:
                force_episode = int(user_input.split(":")[1])
                base_path = exp_config["dataset_base_path"]
                force_path = f"{base_path}/episode_{force_episode}/gelsight_force.npy"
                force = np.load(force_path)
                last_force = force[-1][-1]
                user_feedback = f"The tactile shear force vector has magnitude: {last_force[0]}, xy-direction: [{last_force[1]}, {last_force[2]}]. {exp_config.get('force_reference', '')}"
            else:
                # Manual input
                magnitude = float(input("Enter force magnitude: ").strip())
                x_dir = float(input("Enter x-direction: ").strip())
                y_dir = float(input("Enter y-direction: ").strip())
                user_feedback = f"The tactile shear force vector has magnitude: {magnitude}, xy-direction: [{x_dir}, {y_dir}]. {exp_config.get('force_reference', '')}"
            
            print(user_feedback)
            message = {"role": "user", "content": user_feedback}
            session_log["steps"][-1]["user_feedback"] = user_feedback
        
        elif "," in user_input:
            # Manual property input (hardness, roughness)
            values = [float(val.strip()) for val in user_input.split(",")]
            if len(values) == 2:
                user_feedback = f"The hardness level is: {values[0]}, The roughness level is: {values[1]}"
            else:
                user_feedback = f"Property values: {values}"
            message = {"role": "user", "content": user_feedback}
            session_log["steps"][-1]["user_feedback"] = user_feedback
        
        elif os.path.exists(user_input):
            # User provided a path to tactile images
            tactile_path = user_input
            first_tactile, last_tactile = find_first_last_imgs_octopi(tactile_path)
            base64_image_0 = encode_image(first_tactile)
            base64_image_1 = encode_image(last_tactile)
            
            content = [
                {"type": "text", "text": "The first image is the gelsight tactile sensor image before touching the object"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_0}"}},
                {"type": "text", "text": "The second image is the gelsight tactile sensor image after touching the object"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_1}"}}
            ]
            
            if exp_config.get("tactile_analysis_prompt"):
                content.append({"type": "text", "text": exp_config["tactile_analysis_prompt"]})
            
            message = {"role": "user", "content": content}
            session_log["steps"][-1]["user_feedback"] = f"Tactile images: {first_tactile}, {last_tactile}"
        
        else:
            message = {"role": "user", "content": user_input}
            session_log["steps"][-1]["user_feedback"] = user_input

        messages.append({"role": "assistant", "content": response})
        messages.append(message)

    save_session(session_log, exp_config)


def save_session(session_log, exp_config):
    """Save session log to output file."""
    output_file = exp_config["output_file"]
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "a") as f:
        f.write(json.dumps(session_log) + "\n")
    print(f"\nSession saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Touch VLA - Run robotic tactile experiments')
    parser.add_argument('--experiment', type=str, choices=['wipe', 'mango', 'cup'], required=True,
                        help='Which experiment to run: wipe (smoothness), mango (hardness), or cup (force)')
    parser.add_argument('--api-key', type=str, default="your api key",
                        help='OpenAI API key')
    args = parser.parse_args()
    
    # Update API key
    global client
    client = OpenAI(api_key=args.api_key)
    
    exp_config = EXPERIMENTS[args.experiment]
    initial_prompt = exp_config["prompt"]
    
    print(f"\n{'='*60}")
    print(f"Touch VLA - {args.experiment.upper()} Experiment")
    print(f"Mode: Manual Interactive")
    print(f"{'='*60}\n")
    print(f"Task prompt: {initial_prompt}\n")
    
    # Run interactive session
    run_interactive_session(initial_prompt, args.experiment)

if __name__ == "__main__":
    main()