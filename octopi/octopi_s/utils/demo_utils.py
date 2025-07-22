from typing import Union
from fastapi import FastAPI
import torch
import natsort
import cv2 as cv
import yaml, os
import torch, os, yaml
from utils.encoder import *
from utils.llm import *
from utils.dataset import *
from transformers import CLIPImageProcessor, AutoProcessor
from transformers.utils import logging
import numpy as np
import shutil
import os
from datetime import datetime
from qwen_vl_utils import process_vision_info



def extract_span(sample_video_path, sample_frame_path, threshold, min_len, max_len, top_frame_num):
    # start = datetime.now()
    def extract_frames(sample_video_path, sample_frame_path):
        vidcap = cv.VideoCapture(sample_video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            cv.imwrite(os.path.join(sample_frame_path, f"{count}.jpg"), image) # save frame as JPEG file      
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1

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
            
    extract_frames(sample_video_path, sample_frame_path)
    sample_frames = natsort.natsorted(os.path.join(sample_frame_path, i) for i in os.listdir(sample_frame_path))
    # 1) Get a certain number of frames above a certain change threshold
    prev_frame_img = cv.imread(sample_frames[0])
    prev_frame_gray = cv.cvtColor(prev_frame_img, cv.COLOR_BGR2GRAY)
    all_diffs = []
    for frame in sample_frames[1:]:
        # frame_id = int(frame.split("/")[-1].split(".")[0])
        frame_img = cv.imread(frame)
        frame_gray = cv.cvtColor(frame_img, cv.COLOR_BGR2GRAY)
        try:
            frame_diff = cv.absdiff(prev_frame_gray, frame_gray)
        except cv.error:
            break
        _, thresh = cv.threshold(frame_diff, threshold, 255, cv.THRESH_BINARY)
        total_diff = np.sum(thresh)
        all_diffs.append((frame, total_diff))
        prev_frame_gray = frame_gray
    all_diffs = sorted(all_diffs, key=lambda t: t[1], reverse=True)[:top_frame_num]
    # 2) Get continuous spans
    max_span, max_indices, second_max_span, second_max_indices = find_longest_spans(all_diffs)
    if second_max_indices is not None:
        final_span = natsort.natsorted(max_span + second_max_span)
        final_indices = natsort.natsorted(max_indices + second_max_indices)
    else:
        final_span = max_span
        final_indices = max_indices
    if len(final_span) > max_len:
        final_span = final_span[:max_len]
    # 3) Remove frames that are not in the final span
    for frame in sample_frames:
        if frame not in final_span:
            os.remove(frame)
    # end = datetime.now()
    # elapsed = (end - start).total_seconds()
    # print(f"Span extracted in {elapsed} seconds.")


def get_tactile_videos(demo_path, object_ids, replace=True):
    tactile_videos = {}
    for sample in natsort.natsorted(os.listdir(demo_path)):
        sample_path = os.path.join(demo_path, sample)
        if os.path.isdir(sample_path):
            sample_int = int(sample)
            if sample_int in object_ids:
                num_parts = len([i for i in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, i)) and i != "frames"])
                if num_parts <= 1:
                    # One object part only
                    sample_video_path = os.path.join(sample_path, f"item.mov")
                    sample_frame_path = os.path.join(sample_path, "frames")
                    if os.path.exists(sample_frame_path) and replace:
                        shutil.rmtree(sample_frame_path)
                    if not os.path.exists(sample_frame_path):
                        os.makedirs(sample_frame_path, exist_ok=True)
                        extract_span(sample_video_path, sample_frame_path, threshold=0, min_len=5, max_len=10, top_frame_num=50)
                    tactile_videos[sample_int] = os.path.join(os.path.join(demo_path, sample), "frames")
                else:
                    # Multiple object parts
                    tactile_videos[sample_int] = []
                    for part in natsort.natsorted(os.listdir(sample_path)):
                        part_path = os.path.join(sample_path, part)
                        part_video_path = os.path.join(part_path, "item.mov")
                        part_frame_path = os.path.join(part_path, "frames")
                        if os.path.exists(part_frame_path) and replace:
                            shutil.rmtree(part_frame_path)
                            # print("replaced")
                        if not os.path.exists(part_frame_path):
                            os.makedirs(part_frame_path, exist_ok=True)
                            extract_span(part_video_path, part_frame_path, threshold=0, min_len=5, max_len=10, top_frame_num=50)
                        tactile_videos[sample_int].append(os.path.join(part_path, "frames"))
    return tactile_videos


def get_tactile_embeds(model, tactile_vificlip, demo_configs, load_exp_configs, object_ids, image_transforms, device, image_processor, new_tokens, saved_embeddings, sample_tactile_paths, rag_object_ids, describe, rank):
    tactile_videos = get_tactile_videos(demo_configs["demo_path"], object_ids)
    tactile_paths = [tactile_videos[i] for i in object_ids]
    tactile_paths_flattened = []
    num_objects = len(tactile_paths)
    task_prompt = [""]
    for i in range(num_objects):
        tactile_paths_i = tactile_paths[i]
        if type(tactile_paths_i) == str:
            # One object part only
            task_prompt.append(f"Object {object_ids[i]}: ")
            task_prompt.append("<tact_tokens>")
            tactile_paths_flattened.append(tactile_paths[i])
        else:
            # Multiple object parts
            task_prompt.append(f"Object {object_ids[i]}\n")
            num_parts = len(tactile_paths[i])
            for p in range(num_parts):
                task_prompt.append(f"Part {object_ids[i]}.{p+1}: ")
                task_prompt.append("<tact_tokens>")
                tactile_paths_flattened.append(tactile_paths[i][p])
                if p != num_parts - 1:
                    task_prompt.append("\n")
        if i != num_objects - 1:
            task_prompt.append("\n\n")
    if len(tactile_paths_flattened) <= 1:
        # Force setting for one object part
        describe = True
        rank = False
    if num_objects <= 1:
        if describe and rank:
            task_prompt[0] = "Describe the object in the following tactile video(s) and rank them in decreasing hardness and roughness.\n\n"
        elif describe:
            task_prompt[0] = "Describe the object in the following tactile video(s).\n\n"
        elif rank:
            task_prompt[0] = "Rank the object in the following tactile video(s) in decreasing hardness and roughness.\n\n"
    else:
        if describe and rank:
            task_prompt[0] = "Describe the objects in the following tactile videos and rank them in decreasing hardness and roughness.\n\n"
        elif describe:
            task_prompt[0] = "Describe the objects in the following tactile videos.\n\n"
        elif rank:
            task_prompt[0] = "Rank the objects in the following tactile videos in decreasing hardness and roughness.\n\n"
    with torch.no_grad():
        question = task_prompt.copy()
        tactile_count = 0
        if demo_configs["rag"] and describe:
            rag_outputs = []
        else:
            rag_outputs = None
        for q in range(len(question)):
            if question[q] == "<tact_tokens>":
                # NOTE: Assume only non-dotted PhysiCLeAR samples
                tactile_path = tactile_paths_flattened[tactile_count]
                if demo_configs["rag"] and describe:
                    tactile_frames, _ = get_frames(tactile_path, None, image_transforms, frame_size=load_exp_configs["frame_size"], train=False, return_indices=True)
                    obj_name_description_map = get_rag_tactile_paths(tactile_frames, tactile_vificlip, saved_embeddings, sample_tactile_paths, rag_object_ids, device, retrieval_object_num=demo_configs["retrieval_object_num"])
                    rag_outputs.append(obj_name_description_map)
                question[q] = f"[{tactile_path}]"
                tactile_count += 1
        joined_question = "".join(question)
        # print(f"Question: {joined_question}")
        messages = [
            {"role": "user", "content": joined_question}
        ]
        question_template = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        question_embeds = process_user_input(question_template, image_processor, model, model.tokenizer, device, new_tokens, load_exp_configs["frame_size"], image_transforms)
        return question_embeds, joined_question, tactile_paths, rag_outputs


def describe_rank(model, tactile_vificlip, demo_configs, load_exp_configs, object_ids, image_transforms, device, image_processor, new_tokens, saved_embeddings, sample_tactile_paths, rag_object_ids, prev_embeds, describe: bool, rank: bool):
    question_embeds, question, tactile_paths, rag_outputs = get_tactile_embeds(model, tactile_vificlip, demo_configs, load_exp_configs, object_ids, image_transforms, device, image_processor, new_tokens, saved_embeddings, sample_tactile_paths, rag_object_ids, describe=describe, rank=rank)
    generation, generation_embeds, question_embeds = generate(question_embeds, model, demo_configs["max_new_tokens"], prev_embeds)
    # print(question, generation)
    if demo_configs["rag"] and describe:
        generation = generation.replace(model.tokenizer.eos_token, "")
        descriptions = generation.split("Object parts ranked")[0].split("Object")[1:]
        part_count = 0
        for obj_count, description in enumerate(descriptions):
            description = description.strip().strip("\n")
            # print(description)
            if "Part" not in description:
                description += "\nMost similar objects (in order of decreasing similarity):"
                for obj_name, obj_descriptions in rag_outputs[part_count].items():
                    if demo_configs["rag_use_descriptions"]:
                        description += f" {obj_name} ({', '.join(sorted([i for i in obj_descriptions]))});"
                    else:
                        description += f" {obj_name};"
                part_count += 1
            else:
                obj_id = description.split("Part")[0]
                parts = description.split("Part")[1:]
                for p, part in enumerate(parts):
                    parts[p] = parts[p].strip().strip("\n")
                    parts[p] += "\nMost similar objects (in order of decreasing similarity):"
                    for obj_name, obj_descriptions in rag_outputs[part_count].items():
                        if demo_configs["rag_use_descriptions"]:
                            parts[p] += f" {obj_name} ({', '.join(sorted([i for i in obj_descriptions]))});"
                        else:
                            parts[p] += f" {obj_name};"
                    if p != len(parts) - 1:
                        parts[p] += "\n"
                    part_count += 1
                description = obj_id + "Part " + "Part ".join(parts)
            if obj_count != len(descriptions) - 1:
                description += "\n\n"
            descriptions[obj_count] = description
        new_generation = "Object " + "Object ".join(descriptions)
        if rank and len(rag_outputs) > 1:
            generation = new_generation + "\n\nObject parts ranked" + "Object parts ranked".join(generation.split("Object parts ranked")[1:])
        else:
            generation = new_generation
        generation += model.tokenizer.eos_token
        # Regenerate generation embeddings
        generation_tokens = encode_text(model.tokenizer, generation)
        generation_embeds = torch.unsqueeze(model.llm.get_input_embeddings()(generation_tokens), dim=0)
    all_embeds = torch.cat([question_embeds, generation_embeds], dim=1)
    return generation, all_embeds, question, tactile_paths



def generate(question_embeds, model, max_new_tokens, prev_embeds=None):
    if prev_embeds is not None:
        question_embeds = torch.cat([prev_embeds, question_embeds], dim=1)
    generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None)
    generation = model.tokenizer.decode(generation_tokens[0]) # https://huggingface.co/docs/transformers/main/llm_tutorial
    generation_embeds = model.llm.get_input_embeddings()(generation_tokens)
    return generation, generation_embeds, question_embeds