import os
import numpy as np
import cv2
import random
from utils.physiclear_constants import *
import json
import argparse
import shutil
import natsort


def extract_all_frames(dataset, dataset_file_path, frames_output_path, obj_count):
    os.makedirs(os.path.join(frames_output_path, f'{dataset}_{obj_count}'), exist_ok=True)
    cap = cv2.VideoCapture(dataset_file_path)
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(os.path.join(frames_output_path, f'{dataset}_{obj_count}/tactile'), exist_ok=True)
    for i in range(1, frame_number):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_output_path, f'{dataset}_{obj_count}/tactile', str(i).rjust(10, '0') + '.jpg')
        cv2.imwrite(frame_path, frame)
    cap.release()


def get_physiclear_frames(dataset, tactile_path, frames_output_path):
    exploratory_procedures = os.listdir(tactile_path)
    obj_count = 0
    for i in range(len(exploratory_procedures)):
        exploratory_procedure = exploratory_procedures[i]
        exploratory_procedure_path = os.path.join(tactile_path, exploratory_procedure)
        dataset_files = os.listdir(exploratory_procedure_path)
        for j in range(len(dataset_files)):
            tactile_file_path = os.path.join(exploratory_procedure_path, dataset_files[j])
            try:
                extract_all_frames(dataset, tactile_file_path, frames_output_path, obj_count)
                object_id = f"physiclear_" + "_".join(tactile_file_path.split("/")[-1].split("_")[:-1]).strip()
                object_name = OBJECTS_PART_NAMES[f"physiclear_" + "_".join(tactile_file_path.split("/")[-1].split("_")[:-1]).strip()]
                if object_id in TRAIN_OBJECTS:
                    split = "train"
                elif object_id in VAL_OBJECTS:
                    split = "val"
                elif object_id in TEST_OBJECTS:
                    split = "test"
                sample_data = {
                    "object_id": object_id,
                    "object": object_name,
                    "properties": {
                        "hardness": RATINGS['hardness'][object_id],
                        "roughness": RATINGS['roughness'][object_id]
                    },
                    "tactile_format": "video",
                    "exploratory_procedure": exploratory_procedure,
                    "tactile_path": tactile_file_path,
                    # "rgb_path": rgb_file_path,
                    "split": split
                }
                data_file = open(os.path.join(frames_output_path, f'{dataset}_{obj_count}/data.json'), "w")
                json.dump(sample_data, data_file, indent=4)
                data_file.close()

                if j % 100 == 0:
                    print(f"{j} / {len(dataset_files)} samples extracted for {exploratory_procedure}.")
                obj_count += 1
            except KeyError:
                # print(object_id)
                continue


def get_hardness_frames(dataset, tactile_path, frames_output_path):
    collections = os.listdir(tactile_path)
    obj_count = 0
    for i in range(len(collections)):
        collection = collections[i]
        collection_path = os.path.join(tactile_path, collection)
        dataset_files = os.listdir(collection_path)
        for j in range(len(dataset_files)):
            tactile_file_path = os.path.join(collection_path, dataset_files[j])
            try:
                extract_all_frames(dataset, tactile_file_path, frames_output_path, obj_count)

                object_id = f"{dataset}_" + "_".join(tactile_file_path.split("/")[-1].split("_")[:2]).strip()
                sample_data = {
                    "object_id": object_id,
                    # "object": None,
                    # "properties": None,
                    "tactile_format": "video",
                    "tactile_path": tactile_file_path,
                    "split": "train"
                }
                data_file = open(os.path.join(frames_output_path, f'{dataset}_{obj_count}/data.json'), "w")
                json.dump(sample_data, data_file, indent=4)
                data_file.close()

                if j % 100 == 0:
                    print(f"{j} / {len(dataset_files)} samples extracted for {collection}.")
                obj_count += 1
            except KeyError:
                continue


def get_objectfolder_frames(dataset, dataset_path, frames_output_path):
    object_map = {
        1: "a ceramic soup spoon",
        2: "a ceramic bowl",
        3: "a ceramic salad plate",
        4: "a ceramic dinner plate",
        5: "a wooden hair comb",
        6: "a blue, glass bowl",
        7: "a glass decorative plate",
        8: "a ceramic mixing bowl",
        9: "a ceramic serving bowl",
        10: "a ceramic soup bowl",
        11: "a wooden strainer spoon",
        12: "a wooden soup ladle",
        13: "a wooden serving spoon",
        14: "a wooden salad fork",
        15: "a wooden mixing spoon",
        16: "a wooden frying spatula",
        17: "an 8-inch, iron skillet",
        18: "a 10.25-inch, iron skillet",
        19: "a 10.5-inch, iron griddle",
        20: "an iron Dutch oven",
        21: "an iron Dutch oven lid",
        22: "a glass rinsing cup",
        23: "a plastic hand scoop",
        24: "a large, red, plastic shovel toy",
        25: "a small, green, plastic shovel toy",
        26: "a polycarbonate handle spoon",
        27: "a round, wooden plate",
        28: "a square, wooden plate",
        29: "a large, wooden cutting board",
        30: "a medium, wooden cutting board",
        31: "a small, wooden cutting board",
        32: "a wooden wine glass",
        33: "a wooden drinking cup",
        34: "a wooden beer mug",
        35: "a brown, polycarbonate portion cup",
        36: "a white, polycarbonate portion cup",
        37: "a steel cake pan",
        38: "a steel loaf pan",
        39: "a small, steel wrench",
        40: "a medium, steel wrench",
        41: "a large, steel wrench",
        42: "an iron pestle",
        43: "an iron mortar",
        44: "an iron sculpture",
        45: "an iron ladle",
        46: "an iron spatula",
        47: "a decorative cast iron",
        48: "a large, plastic mixing bowl",
        49: "a medium, plastic mixing bowl",
        50: "a small, plastic mixing bowl",
        51: "a glass fruit bowl",
        52: "a small, steel fork",
        53: "a large, steel fork",
        54: "a small, steel spoon",
        55: "a large, steel spoon",
        56: "a large, plastic knife",
        57: "a medium, plastic knife",
        58: "a small, plastic knife",
        59: "a glass soap dish",
        60: "a beer glass",
        61: "a large, ceramic container",
        62: "a medium, ceramic container",
        63: "a small, ceramic container",
        64: "a ceramic mug",
        65: "a ceramic vase",
        66: "an iron plate handle",
        67: "an iron plate",
        68: "a wooden plate base",
        69: "an iron display stand",
        70: "a polycarbonate dropping funnel",
        71: "a polycarbonate container lid",
        72: "a polycarbonate food pan",
        73: "a large ceramic flowerpot",
        74: "a small ceramic flowerpot",
        75: "a green, ceramic vase",
        76: "a blue, ceramic vase",
        77: "an orange, ceramic vase",
        78: "a large, ceramic swan",
        79: "a small, ceramic swan",
        80: "a wooden spoon holder",
        81: "a wooden utensil container",
        82: "a glass can",
        83: "a steel potato masher",
        84: "a steel skimmer",
        85: "a steel pasta server",
        86: "a steel slotted spoon",
        87: "a steel solid turner",
        88: "a steel ladle",
        89: "a steel solid spoon",
        90: "a steel slotted turner",
        91: "a green glass",
        92: "a red glass",
        93: "a glass vase",
        94: "a glass salad bowl",
        95: "a polycarbonate scoop",
        96: "a polycarbonate box lid",
        97: "a plastic frisbee",
        98: "an iron kettlebell",
        99: "a plastic trim removal tool",
        100: "a plastic trim removal tool"
    }

    objects = os.listdir(dataset_path)
    object_count = 0
    sample_count = 0
    for i in range(len(objects)):
        object_id = objects[i]
        object_path = os.path.join(dataset_path, object_id)
        samples = os.listdir(os.path.join(object_path, "tactile_data"))
        object_count += 1
        for j in range(len(samples)):
            if not os.path.isdir(os.path.join(object_path, "tactile_data", samples[j])):
                continue
            sample_path = os.path.join(object_path, "tactile_data", samples[j], "0", "gelsight")
            if "backup" in sample_path:
                continue
            os.makedirs(os.path.join(frames_output_path, f'{dataset}_{sample_count}'), exist_ok=True)
            os.makedirs(os.path.join(frames_output_path, f'{dataset}_{sample_count}', 'tactile'), exist_ok=True)
            for frame in os.listdir(sample_path):
                shutil.copyfile(os.path.join(sample_path, frame), os.path.join(frames_output_path, f'{dataset}_{sample_count}', 'tactile', frame))
            sample_data = {
                "object_id": f"objectfolder_{object_id}",
                "object": object_map[int(object_id)],
                # "properties": None,
                "tactile_format": "video",
                "exploratory_procedure": "pressing",
                "tactile_path": sample_path,
                "split": "train"
            }
            try:
                data_file = open(os.path.join(frames_output_path, f'{dataset}_{sample_count}/data.json'), "w")
            except FileNotFoundError:
                continue
            json.dump(sample_data, data_file, indent=4)
            data_file.close()
            sample_count += 1
        if object_count % 10 == 0:
            print(f"{object_count} / {len(objects)} objects extracted for {dataset}.")


def extract_span(dataset, output_path, threshold, min_len, max_len, top_frame_num):
    def find_longest_spans(arr):
        # Find the maximum length by traversing the array
        max_count = 0
        second_max_count = 0
        span, indices, max_indices, second_max_indices = [], [], [], []
        count = 0
        arr_by_image = natsort.natsorted(arr, key=lambda t: t[0])
        arr_by_image = [i[0] for i in arr_by_image]
        for i in range(1, len(arr_by_image)):
            # Check if the current element is equal to previous element +1
            frame_id = int(arr_by_image[i].split("/")[-1].split(".")[0])
            prev_frame_id = int(arr_by_image[i-1].split("/")[-1].split(".")[0])
            if frame_id == prev_frame_id + 1:
                if count == 0:
                    span.append(arr_by_image[i-1])
                    count += 1
                    # indices.append(i-1)
                count += 1
                span.append(arr_by_image[i])
                # indices.append(i)
            # Reset the count
            else:
                # Update the maximum
                if count > max_count:
                    max_count = count
                    max_span = span
                    # max_indices = indices
                elif count > second_max_count:
                    second_max_count = count
                    second_max_span = span
                    # second_max_indices = indices
                span = []
                count = 0
                # indices = []
        try:
            return max_span, max_indices, second_max_span, second_max_indices
        except UnboundLocalError:
            pass
        try:
            return max_span, max_indices, None, None
        except UnboundLocalError:
            # If there is no continuous span, get frame with the biggest difference
            max_span = [arr[0][0]]
            max_indices = []
            return max_span, max_indices, None, None

    print(f"Extracting spans with a minimum length of {min_len} and maximum length of {max_len} for {dataset}...")
    sample_count = 0
    num_samples = len([i for i in os.listdir(output_path) if dataset in i])
    for sample in os.listdir(output_path):
        if dataset == sample.split("_")[0]:
            sample_count += 1
            sample_path = os.path.join(output_path, sample)
            sample_frames = natsort.natsorted(os.path.join(sample_path, "tactile", i) for i in os.listdir(os.path.join(sample_path, "tactile")))
            # 1) Get a certain number of frames above a certain change threshold
            prev_frame_img = cv2.imread(sample_frames[0])
            prev_frame_gray = cv2.cvtColor(prev_frame_img, cv2.COLOR_BGR2GRAY)
            all_diffs = []
            for frame in sample_frames[1:]:
                # frame_id = int(frame.split("/")[-1].split(".")[0])
                frame_img = cv2.imread(frame)
                frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
                try:
                    frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)
                except cv2.error:
                    break
                _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
                total_diff = np.sum(thresh)
                all_diffs.append((frame, total_diff))
                prev_frame_gray = frame_gray
            all_diffs = sorted(all_diffs, key=lambda t: t[1], reverse=True)[:top_frame_num]
            if sample_count % 100 == 0:
                print(f"{sample_count} / {num_samples} spans processed for {dataset}.")
            # Check for minimum length
            if len(all_diffs) < min_len:
                shutil.rmtree(sample_path)
                continue
            # 2) Get continuous spans
            max_span, max_indices, second_max_span, second_max_indices = find_longest_spans(all_diffs)
            if second_max_indices is not None:
                final_span = natsort.natsorted(max_span + second_max_span)
                final_indices = natsort.natsorted(max_indices + second_max_indices)
            else:
                final_span = max_span
                final_indices = max_indices
            # Check for minimum length
            if len(final_span) < min_len:
                shutil.rmtree(sample_path)
                continue
            if len(final_span) > max_len:
                final_span = final_span[:max_len]
            # 3) Remove frames that are not in the final span
            for frame in sample_frames:
                if frame not in final_span:
                    os.remove(frame)
    num_remaining_samples = len([i for i in os.listdir(output_path) if dataset in i])
    print(f"{num_remaining_samples} / {num_samples} spans saved for {dataset}.")


def extract_dataset_spans(dataset, dataset_path, output_path, threshold, min_len, max_len, top_frame_num):
    dataset_path = os.path.join(dataset_path, dataset)
    if dataset == "physiclear" or dataset == "physicleardotted":
        get_physiclear_frames(dataset, dataset_path, output_path)
    elif dataset == "hardness":
        get_hardness_frames(dataset, dataset_path, output_path)
    elif dataset == "objectfolder":
        get_objectfolder_frames(dataset, dataset_path, output_path)
    extract_span(dataset, output_path, threshold, min_len, max_len, top_frame_num)


def get_physiclear_samples(data_output_path, train_json_path, val_json_path, test_json_path):
    # Shuffle seen objects before train/val split
    random.shuffle(TRAIN_OBJECTS)
    samples = [i for i in os.listdir(data_output_path) if os.path.isdir(os.path.join(data_output_path, i))]
    train_sample_paths = {}
    val_sample_paths = {}
    test_sample_paths = {}
    for sample in samples:
        try:
            f = open(os.path.join(data_output_path, sample, "data.json"), "r")
        except FileNotFoundError:
            continue
        if not os.path.exists(os.path.join(data_output_path, sample, "tactile")):
            continue
        data = json.load(f)
        if not "object_id" in data.keys():
            continue
        sample_obj = data["object_id"]
        if len(VAL_OBJECTS) == 0:
            if sample_obj in TRAIN_OBJECTS:
                rand = random.random()
                if rand < 0.8:
                    if sample_obj not in train_sample_paths.keys():
                        train_sample_paths[sample_obj] = [os.path.join(data_output_path, sample)]
                    else:
                        train_sample_paths[sample_obj].append(os.path.join(data_output_path, sample))
                elif rand >= 0.8:
                    if sample_obj not in val_sample_paths.keys():
                        val_sample_paths[sample_obj] = [os.path.join(data_output_path, sample)]
                    else:
                        val_sample_paths[sample_obj].append(os.path.join(data_output_path, sample))
        else:
            if sample_obj in TRAIN_OBJECTS:
                if sample_obj not in train_sample_paths.keys():
                    train_sample_paths[sample_obj] = [os.path.join(data_output_path, sample)]
                else:
                    train_sample_paths[sample_obj].append(os.path.join(data_output_path, sample))
            if sample_obj in VAL_OBJECTS:
                if sample_obj not in val_sample_paths.keys():
                    val_sample_paths[sample_obj] = [os.path.join(data_output_path, sample)]
                else:
                    val_sample_paths[sample_obj].append(os.path.join(data_output_path, sample))
        if sample_obj in TEST_OBJECTS:
            if sample_obj not in test_sample_paths.keys():
                test_sample_paths[sample_obj] = [os.path.join(data_output_path, sample)]
            else:
                test_sample_paths[sample_obj].append(os.path.join(data_output_path, sample))
    with open(train_json_path, 'w') as f:
        json.dump(train_sample_paths, f)
        f.close()
    with open(val_json_path, 'w') as f:
        json.dump(val_sample_paths, f)
        f.close()
    with open(test_json_path, 'w') as f:
        json.dump(test_sample_paths, f)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="data/tactile_datasets", help='directory with tactile dataset files')
    parser.add_argument('--frame_output_path', default="data/samples", help='directory to save processed frames and sample files')
    parser.add_argument('--qa_output_path', default="data/llm_qa", help='directory to save LLM QA pairs')
    args = parser.parse_args()
    os.makedirs(args.frame_output_path, exist_ok=True)
    os.makedirs(args.qa_output_path, exist_ok=True)

    # 1) Get frames from datasets
    thresholds = {
        "physiclear": 0,
        "physicleardotted": 0,
        "hardness": 0,
        "objectfolder": 0,
    }
    min_len = 5
    max_len = 10
    top_frame_num = 50
    datasets = ["physiclear"] # physicleardotted
    for dataset in datasets:
        print(f"\nGetting frames from {dataset}...")
        threshold = thresholds[dataset]
        extract_dataset_spans(dataset, args.dataset_path, args.frame_output_path, threshold, min_len, max_len, top_frame_num)
    print("Done!")

    # 2) Store PhysiCLeAR sample paths
    print(f"\nGetting sample files...")
    train_json_path = os.path.join(args.qa_output_path, "train_samples.json")
    val_json_path = os.path.join(args.qa_output_path, "val_samples.json")
    test_json_path = os.path.join(args.qa_output_path, "test_samples.json")
    get_physiclear_samples(args.frame_output_path, train_json_path, val_json_path, test_json_path)
    print("Done!")