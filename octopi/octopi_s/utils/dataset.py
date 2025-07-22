import pickle 
import torch 
from torch.utils.data import Dataset
import numpy as np
import os
import natsort
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import random
import json
from .physiclear_constants import *
from torchvision.transforms.functional import crop
from typing import TypeVar, Optional, Iterator
T_co = TypeVar('T_co', covariant=True)


def regression_collate_fn(data):
    # Images
    max_frame_length = 0
    frames = []
    properties = []
    datasets = []
    paths = []
    for k in data:
        frame_len = torch.squeeze(torch.stack(k[0]), dim=0).shape[0]
        if frame_len > max_frame_length:
            max_frame_length = frame_len
        properties.append(k[1])
        datasets.append(k[2])
        paths.append(k[3])
    for k in data:
        new_k = torch.squeeze(torch.stack(k[0]), dim=0)
        frame_len = new_k.shape[0]
        if frame_len < max_frame_length:
            new_k_0 = torch.stack([new_k[0]] * (max_frame_length - frame_len), dim=0)
            new_k = torch.cat([new_k_0, new_k], dim=0)
        frames.append(new_k)
    frames = torch.stack(frames)
    properties = torch.stack(properties)
    return frames, properties, datasets, paths


class TactilePropertyRegressionDataset(Dataset):
    def __init__(self, image_processor, tokenizer, data_path, split_name, datasets, frame_size, flip_p=0):
        super().__init__()
        self.split_name = split_name
        self.flip_p = flip_p
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.num_samples = 0
        self.tactile = {}
        self.properties = {}
        self.datasets = []
        self.frame_size = frame_size
        for sample in os.listdir(data_path):
            sample_dataset = sample.split("_")[0]
            try:
                data = json.load(open(os.path.join(data_path, sample + "/data.json"), "r"))
            except FileNotFoundError:
                continue
            if split_name != data['split']:
                continue
            if "tactile" not in os.listdir(os.path.join(data_path, sample)):
                continue
            if "properties" in data.keys():
                if sample.split("_")[0] not in datasets:
                    continue
                if sample_dataset not in self.datasets:
                    self.datasets.append(sample_dataset)
                    self.properties[sample_dataset] = []
                    self.tactile[sample_dataset] = []
                self.num_samples += 1
                self.tactile[sample_dataset].append(os.path.join(data_path, sample + "/tactile"))
                self.properties[sample_dataset].append([data['properties']['hardness'], data['properties']['roughness']])
    
    def __len__(self): 
        return self.num_samples

    def __getitem__(self, index):
        # 1) Choose dataset
        dataset = random.choice(self.datasets)
        index = index % len(self.tactile[dataset])
        # 2) Get tactile data
        transforms_list = [
            transforms.ToTensor(),
            transforms.Resize(self.frame_size, interpolation=3),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ]
        if self.split_name == "train":
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomHorizontalFlip(1))
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomVerticalFlip(1))
        else:
            transforms_list.append(transforms.CenterCrop(self.frame_size))
        image_transforms = transforms.Compose(transforms_list)
        tactile = self.tactile[dataset][index]
        all_tactile_frames = []
        if self.split_name == "train":
            tactile_frames, _ = get_frames(tactile, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
        else:
            tactile_frames, _ = get_frames(tactile, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
        all_tactile_frames.append(tactile_frames) # [(l, c, h, w)]
        # 3) Get property labels
        properties = torch.Tensor(self.properties[dataset][index])
        return all_tactile_frames, properties, dataset, tactile


class TactileLLMDataset(Dataset):
    def __init__(self, image_processor, files, split_name, tokenizer, frame_size, flip_p, model_type, rag=False, tactile_vificlip=None, saved_embeddings=None, sample_tactile_paths=None, object_ids=None, device=None, retrieval_object_num=1):
        super().__init__()
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.eos_token_number = self.tokenizer.encode(self.eos_token)
        self.frame_size = frame_size
        self.flip_p = flip_p
        self.image_processor = image_processor
        self.samples = None
        self.model_type = model_type
        self.rag = rag
        if self.rag:
            self.tactile_vificlip = tactile_vificlip
            self.saved_embeddings = saved_embeddings
            self.sample_tactile_paths = sample_tactile_paths
            self.object_ids = object_ids
            self.device = device
            self.retrieval_object_num = retrieval_object_num
        if "llama-3" in self.model_type:
            self.eot_token = "<|eot_id|>"
        for f in files:
            with open(f) as json_file:
                if self.samples is None:
                    self.samples = json.load(json_file)
                else:
                    self.samples += json.load(json_file)
                json_file.close()
    
    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, index):
        # 1) Get sample question, answer and tactile path(s)
        sample = self.samples[index]
        tactile = sample["info"]["tactile"]
        all_objects_dict = sample["info"]["objects"]
        rag_outputs = []
        # 3) Get frame tensors
        tactile_frames = []
        all_indices = []
        all_datasets = []
        for t in tactile:
            dataset = t.split("/")[-2].split("_")[0]
            image_transforms = get_image_transforms(self.frame_size, dataset, split_name=self.split_name, flip_p=self.flip_p)
            if self.split_name == "train":
                frames, indices = get_frames(t, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
            else:
                frames, indices = get_frames(t, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
                if self.rag:
                    obj_name_description_map = get_rag_tactile_paths(frames, self.tactile_vificlip, self.saved_embeddings, self.sample_tactile_paths, self.object_ids, self.device, retrieval_object_num=self.retrieval_object_num)
                    rag_outputs.append(obj_name_description_map)
            tactile_frames.append(frames)
            all_indices.append(indices)
            all_datasets.append(dataset)
        if "scenario" in sample["info"].keys():
            # Scenario reasoning
            scenario = sample["info"]["scenario"]
            scenario_steps = len(sample["chat"])
            target = sample["info"]["target"]
            num_candidates = sample["info"]["num_candidates"]
            return sample["chat"], tactile_frames, tactile, all_datasets, all_indices, all_objects_dict, scenario, scenario_steps, target, num_candidates, rag_outputs
        else:
            # Descriptions and/or rankings
            question = self.tokenizer.apply_chat_template(sample["chat"][:-1], tokenize=False, add_generation_prompt=True)
            answer = sample["chat"][-1]["content"]
            if "llama-3" in self.model_type:
                answer_tokens = encode_text(self.tokenizer, answer + self.eot_token)
            else:
                answer_tokens = encode_text(self.tokenizer, answer + self.eos_token)
            return question, sample["chat"], answer_tokens, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict
        

def get_rag_tactile_paths(original_tactile_frames, tactile_vificlip, saved_embeddings, sample_tactile_paths, object_ids, device, retrieval_object_num=1):
    cos = nn.CosineSimilarity(dim=1, eps=1e-08)
    original_tactile_frames = torch.unsqueeze(original_tactile_frames, dim=0)
    tactile_video_features, _, _, _ = tactile_vificlip(original_tactile_frames.to(device), None, None)
    similarities = cos(saved_embeddings, tactile_video_features)
    similarities_topk = torch.topk(similarities, k=retrieval_object_num)
    similar_objects = [object_ids[i] for i in similarities_topk.indices]
    obj_name_description_map = {}
    for obj in similar_objects:
        obj_name = OBJECTS_PART_NAMES[obj]
        obj_name_description_map[obj_name] = sorted(OPEN_SET_TEXTURES[obj])
    return obj_name_description_map
    

def encode_text(tokenizer, text):
    # Remove start and end tokens if they exist
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.int64)
    if tokenizer.bos_token is not None:
        tokens = tokens[1:]
    return tokens


def get_dataset_sensor_type(dataset):
    dataset_sensor_map = {
        "hardness": "dotted",
        "objectfolder": "plain",
        "physiclear": "plain",
        "physicleardotted": "dotted",
    }
    return dataset_sensor_map[dataset]


def get_image_transforms(frame_size, dataset, split_name, flip_p):
    transforms_list = [
        transforms.ToTensor(),
        transforms.Resize(frame_size, interpolation=3),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ]
    if split_name == "train":
        if random.random() < flip_p:
            transforms_list.append(transforms.RandomHorizontalFlip(1))
        if random.random() < flip_p:
            transforms_list.append(transforms.RandomVerticalFlip(1))
    else:
        transforms_list.append(transforms.CenterCrop(frame_size))
    image_transforms = transforms.Compose(transforms_list)
    return image_transforms


def get_frames(frames_path, image_processor, image_transforms, frame_size, train=True, return_indices=False):
    # Get relevant object(s) and their frames
    image_tensors = []
    all_obj_sample_frames = natsort.natsorted(os.path.join(frames_path, i) for i in os.listdir(frames_path))
    frame_indices = natsort.natsorted([int(i.split("/")[-1].split(".")[0]) for i in all_obj_sample_frames])
    # Process images
    image = Image.open(all_obj_sample_frames[0]).convert("RGB")
    image = image_transforms(image)
    if train:
        i, j, _, _ = transforms.RandomCrop.get_params(image, output_size=(frame_size, frame_size))
        image = crop(image, i, j, frame_size, frame_size)
    image_tensors.append(image)
    for frame in all_obj_sample_frames[1:]:
        image = Image.open(frame).convert("RGB")
        image = image_transforms(image)
        if train:
            image = crop(image, i, j, frame_size, frame_size)
        image_tensors.append(image)
    image_tensors = torch.stack(image_tensors, dim=0) # (l, c, h, w)
    if return_indices:
        frame_indices = [int(i.split("/")[-1].split(".")[0]) for i in all_obj_sample_frames]
        return image_tensors, frame_indices
    else:
        return image_tensors