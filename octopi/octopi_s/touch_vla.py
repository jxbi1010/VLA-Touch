from openai import OpenAI
import json
import os
from datetime import datetime
import base64
import numpy as np
import re
import pickle


client = OpenAI(api_key="your api key")
# Set your OpenAI API key
MODEL = "gpt-4o"

# === Settings ===
# Path to a text file or list that stores image paths

num_episode = 10
num_success = 0
# TRIAL_IMAGE_PATHS = [f"/users/kevinma/tactile/dataset/wipe/pink/episode_{i}/camera1/rgb_0.jpg" for i in range(num_episode)]

pink = np.load('pink.npy')
brown = np.load('brown.npy')
# hard = np.load('hard.npy')
# soft = np.load('soft.npy')
# with open('hard.pkl', 'rb') as f:
#     hard = pickle.load(f)
# with open('soft.pkl', 'rb') as f:
#     soft = pickle.load(f)

OUTPUT_FILE = "results/wipe_octopi_results.jsonl"  # Output trial results
# OUTPUT_FILE = "results/mango_raw_results.jsonl"  # Output trial results
# OUTPUT_FILE = "results/cup_force_ref_results.jsonl"  # Output trial results

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

def run_trial(idx, initial_prompt):
    """Run one trial automatically with a numbered trial."""
    global num_success
    image_path = f"/users/kevinma/tactile/dataset/wipe/pink/episode_{idx}/camera1/rgb_0.jpg"
    # image_path = f"/users/kevinma/tactile/dataset/mango/medium/episode_{idx + 20}/camera1/rgb_0.jpg"
    # image_path = f"/users/kevinma/tactile/dataset/mango/soft/episode_{idx + 10}/camera1/rgb_0.jpg"
    # image_path = f"/users/kevinma/tactile/dataset/water_cup_new/empty/episode_{idx + 80}/camera1/rgb_0.jpg"
    # image_path = f"/users/kevinma/tactile/dataset/water_cup_new/full/episode_{2 * idx}/camera1/rgb_0.jpg"
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

    trial_log = {
        "trial_number": idx + 1,
        "start_time": str(datetime.now()),
        "image": image_path,
        "initial_prompt": initial_prompt,
        "steps": []
    }

    print(f"\n=== Starting Trial {idx + 1} ===")
    print(f"Image: {image_path}")
    i = 0
    while True:
        # response = send_message(messages, image_path=image_path if len(trial_log["steps"]) == 0 else None)
        response = send_message(messages)
        print("\nAssistant:", response)

        trial_log["steps"].append({
            "assistant": response
        })

        # user_feedback = input("\nEnter feedback (or type 'end' to finish this trial): ").strip()
        # user_input = input("\n Enter octopi classifier result in the format hardness, roughness: ").strip()

        # wipe auto testing
        # if i == 0 and ("left" in response or "pink" in response):
        #     user_input = f"{pink[idx, 0]}, {pink[idx, 1]}"
        #     # user_input = "pink"
        # elif i == 1 and ("right" in response or "brown" in response):
        #     user_input = f"{brown[idx, 0]}, {brown[idx, 1]}"
        #     # user_input = "brown"
        
        # mango auto testing
        if i < 2 and "left" in response:
            # user_input = f"{hard[idx][0][0]}, {hard[idx][0][1]}"
            # user_input = f"{hard[idx+10][0][0]}, {hard[idx+10][0][1]}"
            user_input = "hard"
        elif i < 2 and "right" in response:
            # user_input = f"{soft[idx][0][0]}, {soft[idx][0][1]}"
            # user_input = f"{soft[idx+10][0][0]}, {soft[idx+10][0][1]}"
            user_input = "soft"

        # cup auto testing
        # if i < 2 and "left" in response:
        #     user_input = f"{hard[idx+10, 0]}, {hard[idx+10, 1]}"
        #     # user_input = "hard"
        # elif i < 2 and "right" in response:
        #     user_input = f"{soft[idx+10, 0]}, {soft[idx+10, 1]}"
        #     # user_input = "soft"

        else:
            user_input = "end"

        i += 1
        print(user_input)

        if user_input.lower() == "f" or user_input.lower() == "s" or user_input.lower() == "end":
            print(f"Ending Trial {idx + 1}.")
            # wipe
            # if i == 3 and ("left" in response):
            # mango
            if i == 3 and ("right" in response):
            # cup
            # if user_input.lower() == "s":
                print("SUCCESS!!!")
                num_success += 1

            break
        elif user_input.lower() == "pink" or user_input.lower() == "brown":
            if user_input.lower() == "pink":
                tactile_path = f"/users/kevinma/tactile/dataset/wipe/pink/episode_{idx}/gelsight"
            elif user_input.lower() == "brown":
                tactile_path = f"/users/kevinma/tactile/dataset/wipe/brown/episode_{idx + 20}/gelsight"

            # first_tactile, last_tactile = find_first_last_imgs(tactile_path)
            first_tactile, last_tactile = find_first_last_imgs_octopi(tactile_path)
            base64_image_0 = encode_image(first_tactile)
            base64_image_1 = encode_image(last_tactile)
            message = {"role": "user", "content": [
                { "type": "text", "text": "The first image is the gelsight tactile sensor image before touching the object" },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_0}",
                    },
                },
                { "type": "text", "text": "The second image is the gelsight tactile sensor image after touching the object" },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_1}",
                    },
                },
                { "type": "text", "text": "Analyze the images to figure out the smoothness of the object." },
            ]}
            trial_log["steps"][-1]["user_feedback"] = f"The first image is the gelsight tactile sensor image before touching the object: {first_tactile}. The second image is the gelsight tactile sensor image after touching the object: {last_tactile}"
        elif user_input.lower() == "hard" or user_input.lower() == "soft":
            if user_input.lower() == "hard":
                # tactile_path = f"/users/kevinma/tactile/dataset/mango_touch/hard/episode_{idx + 10}/gelsight"
                # tactile_path = hard[idx][1]
                tactile_path = hard[idx+10][1]
            elif user_input.lower() == "soft":
                # tactile_path = f"/users/kevinma/tactile/dataset/mango_touch/soft/episode_{idx + 30}/gelsight"
                # tactile_path = soft[idx][1]
                tactile_path = soft[idx+10][1]
            tactile_path = str(tactile_path)
            print(tactile_path)
            # first_tactile, last_tactile = find_first_last_imgs(tactile_path)
            first_tactile, last_tactile = find_first_last_imgs_octopi(tactile_path)
            base64_image_0 = encode_image(first_tactile)
            base64_image_1 = encode_image(last_tactile)
            message = {"role": "user", "content": [
                { "type": "text", "text": "The first image is the gelsight tactile sensor image before touching the object" },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_0}",
                    },
                },
                { "type": "text", "text": "The second image is the gelsight tactile sensor image after touching the object" },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_1}",
                    },
                },
                # { "type": "text", "text": "Analyze the images to figure out the hardness of the object." },
            ]}
            trial_log["steps"][-1]["user_feedback"] = f"The first image is the gelsight tactile sensor image before touching the object: {first_tactile}. The second image is the gelsight tactile sensor image after touching the object: {last_tactile}"
        elif user_input.lower() == "cup":
            # tactile_path = f"/users/kevinma/tactile/dataset/water_cup_new/empty/episode_{idx + 80}/gelsight"
            tactile_path = f"/users/kevinma/tactile/dataset/water_cup_new/full/episode_{2 * idx}/gelsight"

            # first_tactile, last_tactile = find_first_last_imgs(tactile_path)
            first_tactile, last_tactile = find_first_last_imgs_octopi(tactile_path)
            base64_image_0 = encode_image(first_tactile)
            base64_image_1 = encode_image(last_tactile)
            message = {"role": "user", "content": [
                { "type": "text", "text": "The first image is the gelsight tactile sensor image before touching the object" },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_0}",
                    },
                },
                { "type": "text", "text": "The second image is the gelsight tactile sensor image after lifting the object" },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_1}",
                    },
                },
                # { "type": "text", "text": "Analyze the images to figure out the hardness of the object." },
            ]}
            trial_log["steps"][-1]["user_feedback"] = f"The first image is the gelsight tactile sensor image before touching the object: {first_tactile}. The second image is the gelsight tactile sensor image after lifting the object: {last_tactile}"
        elif user_input.lower() == "force":
            # force_path = f"/users/kevinma/tactile/dataset/water_cup_new/empty/episode_{idx + 80}/gelsight_force.npy"
            force_path = f"/users/kevinma/tactile/dataset/water_cup_new/full/episode_{2 * idx}/gelsight_force.npy"
            force = np.load(force_path)
            last_force = force[-1][-1]
            print(last_force)
            user_feedback = f"The tactile shear force vector has magnitude: {last_force[0]}, xy-direction: [{last_force[1]}, {last_force[2]}]. For reference, the max force magnitude is around 0.55 for an empty cup and around 1.1 for a full cup."
            message = {"role": "user", "content": user_feedback}
            trial_log["steps"][-1]["user_feedback"] = user_feedback

        elif "," in user_input:
            hardness, roughness = [float(res.strip()) for res in user_input.split(",")]
            user_feedback = f"The hardness level is: {hardness}, The roughness level is: {roughness}"
            message = {"role": "user", "content": user_feedback}
            trial_log["steps"][-1]["user_feedback"] = user_feedback
        else:
            message = {"role": "user", "content": user_input}
            trial_log["steps"][-1]["user_feedback"] = user_input

        messages.append({"role": "assistant", "content": response})
        messages.append(message)


    save_result(trial_log)


def save_result(trial_log):
    """Append one trial's result to the output file."""
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(trial_log) + "\n")
    print(f"Trial {trial_log['trial_number']} results saved to {OUTPUT_FILE}")

def main():
    print("=== Robotic Interactive Trials ===")
    # initial_prompt = input("\nEnter the initial prompt that will be used for all trials: ").strip()
    initial_prompt = "There are two sponges in the image, one on the left and another on the right. Step by step, touch and identify the roughness of the sponges, and pick up the smoother one."
    # initial_prompt = "There are two mangoes in the image, one on the left and another on the right. Step by step, touch and identify the properties of the mangoes, and pick up the riper one."
    # initial_prompt = "There is a cup in the image. Step by step, identify whether the cup is full or empty. If it is not empty, put it onto the plate."

    # for idx, episode_path in enumerate(TRIAL_IMAGE_PATHS, start=1):
    for idx in range(num_episode):
        # if not os.path.exists(episode_path):
        #     print(f"Warning: Image not found {episode_path}. Skipping trial {idx}.")
        #     continue
        run_trial(idx, initial_prompt)

    print("\nAll trials completed.")
    print("num_success:", num_success)

if __name__ == "__main__":
    main()