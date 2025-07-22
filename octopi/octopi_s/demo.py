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
from utils.demo_utils import *
from transformers import CLIPImageProcessor, AutoProcessor
from transformers.utils import logging
import numpy as np
import shutil
import os
from datetime import datetime
from qwen_vl_utils import process_vision_info


# API
app = FastAPI()
logging.set_verbosity_error()

# Run settings
run_type = f"demo"
demo_config_path = f'../configs/{run_type}.yaml'
demo_configs = yaml.safe_load(open(demo_config_path))
device = f'cuda:{demo_configs["cuda"]}'
load_exp_path = demo_configs["load_exp_path"]
f = open(demo_configs["gpu_config"])
gpu_config = json.load(f)
embedding_history_path = demo_configs["embedding_history_path"]
chat_path = demo_configs["chat_path"]
dataset = "physiclear" # NOTE: Assume the tactile inputs uses the non-dotted GelSight Mini

# RAG
tactile_vificlip, dotted_tactile_adapter, plain_tactile_adapter, property_classifier, load_exp_configs = load_encoder(demo_configs, device)
image_transforms = get_image_transforms(load_exp_configs["frame_size"], dataset, split_name="test", flip_p=0)
if demo_configs["rag"]:
    if demo_configs["rag_generate_embeddings"]:
        print("\nGenerating RAG embeddings...")
        generate_rag_embeddings(demo_configs, load_exp_configs, tactile_vificlip, device, demo_configs["rag_sample_dir"], demo_configs["embedding_dir"])
    del dotted_tactile_adapter
    del plain_tactile_adapter
    del property_classifier
    saved_embeddings, sample_tactile_paths, rag_object_ids = get_rag_embeddings(demo_configs, device)
else:
    tactile_vificlip = None
    saved_embeddings = None
    sample_tactile_paths = None
    rag_object_ids = None

# Load models
load_exp_configs = yaml.safe_load(open(os.path.join(load_exp_path, "run.yaml")))
peft = "peft" in demo_configs["load_exp_path"]
tokenizer_path, model_path, new_tokens, no_split_module_classes = get_model_details(load_exp_configs["model_type"])
load_exp_configs.update(demo_configs)
model = load_mllm(load_exp_configs, tokenizer_path, model_path, new_tokens, no_split_module_classes, peft, device, gpu_config, exp_id=None)
if load_exp_configs["use_clip"]:
    image_processor = CLIPImageProcessor.from_pretrained(load_exp_configs["use_clip"])


def save_chat_history(user_input, generation):
    qa = f"###### USER: {user_input}\n\n###### ASSISTANT: {generation}\n\n"
    if not os.path.exists(chat_path):
        write_type = "w"
    else:
        write_type = "a"
    with open(chat_path, write_type) as f:
        f.write(qa)
        f.close()


@app.post("/describe")
def describe_objects(object_ids: str):
    object_ids = [int(i.strip()) for i in object_ids.split(",")]
    if os.path.exists(embedding_history_path):
        prev_embeds = torch.load(embedding_history_path, map_location=device, weights_only=True)
    else:
        prev_embeds = None
    generation, all_embeds, question, tactile_paths_flattened = describe_rank(model, tactile_vificlip, demo_configs, load_exp_configs, object_ids, image_transforms, device, image_processor, new_tokens, saved_embeddings, sample_tactile_paths, rag_object_ids, prev_embeds, describe=True, rank=False)
    torch.save(all_embeds, embedding_history_path)
    save_chat_history(question, generation)
    return {"response": generation}


@app.post("/rank")
def rank_objects(object_ids: str):
    object_ids = [int(i.strip()) for i in object_ids.split(",")]
    if os.path.exists(embedding_history_path):
        prev_embeds = torch.load(embedding_history_path, map_location=device, weights_only=True)
    else:
        prev_embeds = None
    generation, all_embeds, question, tactile_paths_flattened = describe_rank(model, tactile_vificlip, demo_configs, load_exp_configs, object_ids, image_transforms, device, image_processor, new_tokens, saved_embeddings, sample_tactile_paths, rag_object_ids, prev_embeds, describe=False, rank=True)
    torch.save(all_embeds, embedding_history_path)
    save_chat_history(question, generation)
    response_json = {"response": generation}
    ranks = generation.split("Object parts ranked")[1:]
    characters_to_replace = [model.tokenizer.eos_token, "=", ">"]
    for rank in ranks:
        prop = rank.split(":")[0].split()[-1]
        rank = rank.split(":")[-1]
        for character in characters_to_replace:
            rank = rank.replace(character, "")
        rank = rank.split(",")
        response_json[prop] = [i.strip().strip(".") for i in rank]
    return response_json


@app.post("/describe_and_rank")
def describe_rank_objects(object_ids: str):
    object_ids = [int(i.strip()) for i in object_ids.split(",")]
    if os.path.exists(embedding_history_path):
        prev_embeds = torch.load(embedding_history_path, map_location=device, weights_only=True)
    else:
        prev_embeds = None
    generation, all_embeds, question, tactile_paths_flattened = describe_rank(model, tactile_vificlip, demo_configs, load_exp_configs, object_ids, image_transforms, device, image_processor, new_tokens, saved_embeddings, sample_tactile_paths, rag_object_ids, prev_embeds, describe=True, rank=True)
    torch.save(all_embeds, embedding_history_path)
    save_chat_history(question, generation)
    response_json = {"response": generation}
    ranks = generation.split("Object parts ranked")[1:]
    characters_to_replace = [model.tokenizer.eos_token, "=", ">"]
    for rank in ranks:
        prop = rank.split(":")[0].split()[-1]
        rank = rank.split(":")[-1]
        for character in characters_to_replace:
            rank = rank.replace(character, "")
        rank = rank.split(",")
        response_json[prop] = [i.strip().strip(".") for i in rank]
    return response_json


@app.post("/describe_rgb")
def describe_rgb(prompt: str):
    # NOTE: Does not save into chat history or embedding history, only for demo purposes on the UI
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": demo_configs["image_path"],
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    generated_ids = model.llm.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    generation = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = {
        "generation": generation,
    }
    objects = generation.split("Object 1")[-1].split("\n")
    final_objects = []
    for obj in objects:
        final_objects.append(obj.split(":")[-1].strip()[:-1].lower())
    response["objects"] = final_objects
    return {"response": response}


@app.post("/guess_from_objects")
def guess_touch_given_objects(object_candidates: str):
    object_candidates_options = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E"
    }
    object_candidates = [f"{object_candidates_options[i]}) {obj.strip()}" for i, obj in enumerate(object_candidates.split(','))]
    object_candidates_text = ', '.join(object_candidates)
    task_prompt = f"Determine which option the above object is likely to be: {object_candidates_text}?\nFollow the steps below: 1. Select the surface texture descriptions (note: each part of an object contains a different salient texture) that help to distinguish between the given options. 2. Give a succinct case for each option using the selected descriptions. 3. Select the best option and format your answer in the format 'Answer: <letter>) <name> is the most likely option because <reason(s)>'."
    messages = [
        {"role": "user", "content": task_prompt}
    ]
    question_template = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    question_embeds = process_user_input(question_template, image_processor, model, model.tokenizer, device, new_tokens, load_exp_configs["frame_size"], image_transforms)
    if os.path.exists(embedding_history_path):
        prev_embeds = torch.load(embedding_history_path, map_location=device, weights_only=True)
    else:
        prev_embeds = None
    generation, generation_embeds, question_embeds = generate(question_embeds, model, demo_configs["max_new_tokens"], prev_embeds)
    embeds = torch.cat([question_embeds, generation_embeds], dim=1)
    torch.save(embeds, embedding_history_path)
    save_chat_history(task_prompt, generation)
    return {"response": generation}


@app.post("/ask")
def ask(query: str):
    messages = [
        {"role": "user", "content": query}
    ]
    question_template = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    question_embeds = process_user_input(question_template, image_processor, model, model.tokenizer, device, new_tokens, load_exp_configs["frame_size"], image_transforms)
    if os.path.exists(embedding_history_path):
        prev_embeds = torch.load(embedding_history_path, map_location=device, weights_only=True)
    else:
        prev_embeds = None
    generation, generation_embeds, question_embeds = generate(question_embeds, model, demo_configs["max_new_tokens"], prev_embeds)
    embeds = torch.cat([question_embeds, generation_embeds], dim=1)
    torch.save(embeds, embedding_history_path)
    save_chat_history(query, generation)
    return {"response": generation}


@app.post("/reset")
def reset_llm_history():
    if os.path.exists(embedding_history_path):
        os.remove(embedding_history_path)
    if os.path.exists(chat_path):
        os.remove(chat_path)
    for sample in os.listdir(demo_configs["demo_path"]):
        sample_path = os.path.join(demo_configs["demo_path"], sample)
        if os.path.isdir(sample_path):
            num_parts = len([i for i in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, i))])
            if num_parts <= 1:
                # One object part only
                if "frames" in os.listdir(sample_path):
                    shutil.rmtree(os.path.join(sample_path, "frames"))
            else:
                # Multiple object parts
                for part in natsort.natsorted(os.listdir(sample_path)):
                    part_path = os.path.join(sample_path, part)
                    if "frames" in os.listdir(part_path):
                        shutil.rmtree(os.path.join(part_path, "frames"))
    return {"status": "done"}