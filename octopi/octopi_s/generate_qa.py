import os
import random
from utils.physiclear_constants import *
import json
import yaml


def get_property_order(parts, part_indices_map, property, decreasing):
    order = ""
    part_property_pairs = []
    for i, part in enumerate(parts):
        part_idx = i
        part_property_pairs.append((part_idx, RATINGS[property][part]))
    part_property_pairs = sorted(
        part_property_pairs, 
        key=lambda x: x[1],
        reverse=decreasing
    )
    num_pairs = len(part_property_pairs)
    for i in range(num_pairs):
        part_property_pair = part_property_pairs[i]
        order += f"{part_indices_map[part_property_pair[0]]}"
        if i != num_pairs - 1:
            if part_property_pair[1] == part_property_pairs[i+1][1]:
                order += " >= "
            else:
                order += " > "
    return order


def generate_description_ranking_qa(json_path, data_dir, split, num_samples, open_set_texture, use_parts, qa_id):
    # Load samples
    for i in range(len(json_path)):
        if i == 0:
            with open(json_path[i]) as json_file:
                samples = json.load(json_file)
                json_file.close()
        else:
            with open(json_path[i]) as json_file:
                samples_temp = json.load(json_file)
                json_file.close()
            for k, v in samples_temp.items():
                if k in samples.keys():
                    samples[k] += v
                else:
                    samples[k] = v

    # Data
    all_data = []
    for i in range(num_samples):
        min_num_objects = 1
        max_num_objects = 5
        num_objects = random.randint(min_num_objects, max_num_objects)
        object_indices = random.sample(range(min_num_objects, max_num_objects+1), num_objects)
        if num_objects > 1:
            get_order = random.choice([True, False])
        else:
            get_order = False
        if not get_order:
            get_description = True
        else:
            get_description = random.choice([True, False])
        decreasing = True
        if num_objects == 1:
            question = ["Describe the object in the following tactile video(s).\n\n"]
        elif get_description and get_order:
            question = ["Describe the objects in the following tactile videos and rank them in decreasing hardness and roughness.\n\n"]
        elif get_description:
            question = ["Describe the objects in the following tactile videos.\n\n"]
        elif get_order:
            question = ["Rank the objects in the following tactile videos in decreasing hardness and roughness.\n\n"]
        exploratory_procedures = ["pressing", "sliding"]
        data = {
            "info": {
                "get_description": get_description,
                "get_order": get_order,
                "decreasing": decreasing,
                "num_objects": num_objects,
                "exploratory_procedures": exploratory_procedures
            },
            "chat": []
        }
        answer = []
        # Get object(s) and their frames
        objects = random.sample(list(samples.keys()), k=num_objects)
        object_indices = random.sample(range(min_num_objects, max_num_objects+1), num_objects)
        all_tactile = []
        all_objects_dict = {}
        all_parts = []
        part_indices_map = {}
        part_count = 0
        for i, obj in enumerate(objects):
            if use_parts:
                num_parts = random.randint(1, 2)
            else:
                num_parts = 1
            if num_parts == 1:
                object_idx = object_indices[i]
                tactile = random.choice(samples[obj]) + "/tactile"
                all_objects_dict[f"Object {object_idx}"] = obj
                all_parts.append(obj)
                all_tactile.append(tactile)
                question.append(f"Object {object_idx}: ")
                question.append("<tact_tokens>")
                part_indices_map[part_count] = f"{object_idx}"
                part_count += 1
            else:
                parts = [obj]
                for p in range(num_parts-1):
                    part = random.choice(list(samples.keys()))
                    parts.append(part)
                object_idx = object_indices[i]
                all_objects_dict[f"Object {object_idx}"] = {}
                question.append(f"Object {object_idx}\n")
                for p in range(len(parts)):
                    all_objects_dict[f"Object {object_idx}"][p+1] = parts[p]
                    tactile = random.choice(samples[parts[p]]) + "/tactile"
                    all_tactile.append(tactile)
                    question.append(f"Part {object_idx}.{p+1}: ")
                    question.append("<tact_tokens>")
                    all_parts.append(parts[p])
                    part_indices_map[part_count] = f"{object_idx}.{p+1}"
                    part_count += 1
                    if p != num_parts - 1:
                        question.append("\n")
            if i != len(objects) - 1:
                question.append("\n\n")
            if get_description:
                if num_parts == 1:
                    if open_set_texture:
                        random.shuffle(OPEN_SET_TEXTURES[obj])
                        description = ", ".join(OPEN_SET_TEXTURES[obj])
                    answer.append(f"Object {object_idx}: {description}.")
                else:
                    answer.append(f"Object {object_idx}\n")
                    for p in range(num_parts):
                        if open_set_texture:
                            random.shuffle(OPEN_SET_TEXTURES[parts[p]])
                            description = ", ".join(OPEN_SET_TEXTURES[parts[p]])
                        answer.append(f"Part {object_idx}.{p+1}: {description}.")
                        if p != num_parts - 1:
                            answer.append("\n")
                if i != len(objects) - 1:
                    answer.append("\n\n")
                if get_order and i == len(objects) - 1:
                    answer.append("\n\n")
        data["info"]["tactile"] = all_tactile
        data["info"]["objects"] = all_objects_dict
        if get_order:
            hardness_order = get_property_order(all_parts, part_indices_map, "hardness", decreasing)
            roughness_order = get_property_order(all_parts, part_indices_map, "roughness", decreasing)
            if decreasing:
                answer.append(f"Object parts ranked in decreasing hardness: {hardness_order}\nObject parts ranked in decreasing roughness: {roughness_order}")
            else:
                answer.append(f"Object parts ranked in increasing hardness: {hardness_order}\nObject parts ranked in increasing roughness: {roughness_order}")
        data["chat"].append({
                "role": "user",
                "content": "".join(question),
            })
        data["chat"].append({
                "role": "assistant",
                "content": "".join(answer),
            })
        all_data.append(data)
    # Save all data
    file_name = f"description_ranking_qa_{qa_id}"
    data_file = open(os.path.join(data_dir, f"{split}_{file_name}.json"), "w")
    json.dump(all_data, data_file, indent=4) 
    data_file.close()


def generate_scenario_qa(json_path, data_dir, num_samples, scenarios_to_use, open_set_texture, use_parts, qa_id):
    scenario_info = {
        "guess_touch_from_objects_balls": {
            "target_sample": ["physiclear_baseball_seams", "physiclear_tennis_ball", "physiclear_stress_ball"],
            "all_candidate": ["a new baseball's seams", "a tennis ball", "a stress ball"],
            "pre_instruction": "",
            "question": "Task: Determine which option the above object is likely to be: ",
            "post_instruction": "\nFollow the steps below: 1. Select the surface texture descriptions that help to distinguish between the given options. 2. Give a succinct case for each option using the selected descriptions. 3. Select the best option and format your answer in the format 'Answer: <letter>) <name> is the most likely option because <reason(s)>'.",
            "follow_up": "Are the surface tactile properties you mentioned accurate based on common physical characteristics of each object choice's surface? For example, consider its surface texture, material, and how it feels. Correct any inconsistencies or errors. Format your final answer in the format 'Answer: <letter>) <name> is the most likely option because <reason(s)>'."
        },
        "guess_touch_from_objects_fruits": {
            "target_sample": ["physiclear_good_apple", "physiclear_orange", "physiclear_spoilt_apple"],
            "all_candidate": ["a ripe apple", "a ripe orange", "a spoilt apple"],
            "pre_instruction": "",
            "question": "Task: Determine which option the above object is likely to be: ",
            "post_instruction": "\nFollow the steps below: 1. Select the surface texture descriptions that help to distinguish between the given options. 2. Give a succinct case for each option using the selected descriptions. 3. Select the best option and format your answer in the format 'Answer: <letter>) <name> is the most likely option because <reason(s)>'.",
            "follow_up": "Are the surface tactile properties you mentioned accurate based on common physical characteristics of each object choice's surface? For example, consider its surface texture, material, and how it feels. Correct any inconsistencies or errors. Format your final answer in the format 'Answer: <letter>) <name> is the most likely option because <reason(s)>'."
        },
    }

    # Load samples
    for i in range(len(json_path)):
        if i == 0:
            with open(json_path[i]) as json_file:
                samples = json.load(json_file)
                json_file.close()
        else:
            with open(json_path[i]) as json_file:
                samples_temp = json.load(json_file)
                json_file.close()
            for k, v in samples_temp.items():
                if k in samples.keys():
                    samples[k] += v
                else:
                    samples[k] = v

    # Data
    all_data = []
    existing = {
        k: [] for k in list(scenario_info.keys())
    }
    if scenarios_to_use is None:
        scenarios_to_use = {k: v for k, v in scenario_info.items()}
    else:
        scenarios_to_use = {k: v for k, v in scenario_info.items() if k in scenarios_to_use}
    for _ in range(num_samples):
        exploratory_procedures = ["pressing", "sliding"]
        exist = False
        scenario = random.choice(list(scenarios_to_use.keys()))
        # sample_key, sample_value = None, None
        target_key, target_value = None, None
        # candidate_key, candidate_value = None, None
        for k in scenario_info[scenario].keys():
            if "sample" in k:
                if "all" in k:
                    num_objects = len(scenario_info[scenario][k])
                else:
                    num_objects = 1
            if "target" in k:
                target_key = k
                target_value = scenario_info[scenario][k].copy()
            if "candidate" in k:
                candidate_key = k
                candidate_value = scenario_info[scenario][k].copy()
                num_candidates = len(candidate_value)
        get_description = True
        if num_objects > 1:
            get_order = True
        else:
            get_order = False
        if get_order:
            decreasing = True
        else:
            decreasing = False
        data = {
            "info": {
                "scenario": scenario,
                "get_description": get_description,
                "get_order": get_order,
                "decreasing": decreasing,
                "num_objects": num_objects,
                "num_candidates": num_candidates,
                "exploratory_procedures": exploratory_procedures,
            },
            "chat": []
        }

        # 1) Get target and candidate information
        random_idx = random.randint(0, len(target_value)-1)
        target = target_value[random_idx]
        data["info"]["target"] = target
        candidate = candidate_value[random_idx]
        candidate_samples = True if "samples" in candidate_key else False

        # 2) Get description / ranking and reasoning answers
        if num_objects > 1 and candidate_samples:
            candidates = candidate_value.copy()
            chosen_objects_shuffled_index = [i for i in range(num_objects)]
            random.shuffle(chosen_objects_shuffled_index)
            all_tactile = [random.choice(samples[candidates[i]]) + "/tactile" for i in chosen_objects_shuffled_index]
            all_objects = [candidates[i] for i in chosen_objects_shuffled_index]
            reasoning_answer = f"Object {chosen_objects_shuffled_index.index(random_idx)+1}"
        elif num_objects == 1:
            all_tactile = [random.choice(samples[target]) + "/tactile"]
            all_objects = [target]
            options = ["A)", "B)", "C)", "D)"]
            reasoning_answer = options[random_idx] + " " + candidate
        if tuple(all_tactile) in existing[scenario]:
            exist = True
            continue
        else:
            existing[scenario].append(tuple(all_tactile))
        if num_objects == 1:
            question = ["Describe the object in the following tactile video(s).\n\n"]
        else:
            question = ["Describe the objects in the following tactile videos and rank them in decreasing hardness and roughness.\n\n"]
        answer = []
        # Get object frames
        all_objects_dict = {}
        if num_objects > 1 and candidate_samples:
            for i in range(num_objects):
                obj = candidates[chosen_objects_shuffled_index[i]]
                question.append(f"Object {i+1}: ")
                question.append("<tact_tokens>")
                if i != num_objects - 1:
                    question.append("\n\n")
                if get_description:
                    if open_set_texture:
                        random.shuffle(OPEN_SET_TEXTURES[obj])
                        description = ", ".join(OPEN_SET_TEXTURES[obj])
                    answer.append(f"Object {i+1}: {description}.")
                    if i != len(candidates) - 1:
                        answer.append("\n")
                    if get_order and i == len(candidates) - 1:
                        answer.append("\n\n")
                all_objects_dict[f"Object {i+1}"] = all_objects[i]
        elif num_objects == 1:
            question.append(f"Object 1: ")
            question.append("<tact_tokens>")
            if get_description:
                if open_set_texture:
                    random.shuffle(OPEN_SET_TEXTURES[target])
                    description = ", ".join(OPEN_SET_TEXTURES[target])
            all_objects_dict[f"Object 1"] = target
        data["info"]["tactile"] = all_tactile
        data["info"]["objects"] = all_objects_dict
        if num_objects > 1 and candidate_samples and get_order:
            object_indices = [i+1 for i in range(len(all_objects))]
            hardness_order = get_property_order([candidates[i] for i in chosen_objects_shuffled_index], object_indices, "hardness", decreasing)
            roughness_order = get_property_order([candidates[i] for i in chosen_objects_shuffled_index], object_indices, "roughness", decreasing)
            if decreasing:
                answer.append(f"Objects ranked in decreasing hardness: {hardness_order}.\nObjects in decreasing roughness: {roughness_order}.")
            else:
                answer.append(f"Objects ranked in increasing hardness: {hardness_order}.\nObjects in increasing roughness: {roughness_order}.")
        elif num_objects == 1:
            answer = "Object 1: " + description + "."
        data["chat"].append({
            "role": "user",
            "content": "".join(question)
        })
        data["chat"].append({
            "role": "assistant",
            "content": "".join(answer)
        })
        scenario_question = scenario_info[scenario]["question"]
        if num_objects > 1 and candidate_samples:
            scenario_question += target + "?"
        elif num_objects == 1:
            for i in range(len(candidate_value)):
                if i != len(candidate_value) - 1:
                    scenario_question += f"{options[i]} {candidate_value[i]}, "
                else:
                    scenario_question += f"{options[i]} {candidate_value[i]}?"
        data["chat"].append({
            "role": "user",
            "content": scenario_info[scenario]["pre_instruction"] + scenario_question + scenario_info[scenario]["post_instruction"],
        })
        data["chat"].append({
            "question": "assistant",
            "content": reasoning_answer
        })
        if "follow_up" in scenario_info[scenario].keys():
            data["chat"].append({
                "role": "user",
                "content": scenario_info[scenario]["follow_up"],
            })
            data["chat"].append({
                "question": "assistant",
                "content": reasoning_answer
            })
        if not exist:
            all_data.append(data)
    # Save all data
    file_name = f"test_scenario_qa_{qa_id}"
    data_file = open(os.path.join(data_dir, f"{file_name}.json"), "w")
    json.dump(all_data, data_file, indent=4)
    data_file.close()


if __name__ == "__main__":
    run_type = "generate_qa"
    config_path = f'configs/{run_type}.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    os.makedirs(configs["output_data_dir"], exist_ok=True)
    qa_id = input("Enter QA ID: ")
    while qa_id == "":
        qa_id = input("Please enter QA ID: ")

    # 1) Description / ranking
    print("Generating description / ranking QA pairs...")
    for split in ["train", "test"]:
        json_path = [os.path.join(configs["output_data_dir"], f"{split}_samples.json")]
        num_samples = configs[f"description_qa_{split}_num"]
        generate_description_ranking_qa(json_path, configs["output_data_dir"], split, num_samples, configs["open_set_texture"], configs["use_parts"], qa_id)
    print("Done!")

    # 2) Scenario reasoning
    # print("Generating scenario reasoning QA pairs...")
    # num_samples = configs["scenario_qa_test_num"]
    # json_paths = [
    #     os.path.join(configs["output_data_dir"], f"train_samples.json"),
    #     os.path.join(configs["output_data_dir"], f"test_samples.json"),
    # ]
    # generate_scenario_qa(json_paths, configs["output_data_dir"], num_samples, configs["scenarios"], configs["open_set_texture"], configs["use_parts"], qa_id)
    # print("Done!")