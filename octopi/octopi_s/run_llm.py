import itertools
import os
import re
import shutil 
import torch.nn as nn 
import torch 
from torch.utils.data import DataLoader
from torch import optim
import tqdm
import json
import numpy as np
from utils.dataset import *
from utils.llm import *
from utils.encoder import *
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor
import random
import yaml
from datetime import datetime
from scipy.stats import kendalltau



def run_llm(configs, exp_id, g, device, peft):
    # Prepare RAG embeddings for scenario reasoning
    train = len(configs["train_files"]) > 0
    reason = len(configs["reasoning_files"]) > 0
    if reason:
        if configs["rag"]:
            tactile_vificlip, dotted_tactile_adapter, plain_tactile_adapter, property_classifier, load_exp_configs = load_encoder(configs, device)
            if configs["rag_generate_embeddings"]:
                print("\nGenerating RAG embeddings...")
                generate_rag_embeddings(configs, load_exp_configs, tactile_vificlip, device, configs["rag_sample_dir"], configs["embedding_dir"])
            del tactile_vificlip
            del dotted_tactile_adapter
            del plain_tactile_adapter
            del property_classifier
            saved_embeddings, sample_tactile_paths, object_ids = get_rag_embeddings(configs, device)
        else:
            tactile_vificlip = None
            saved_embeddings = None
            sample_tactile_paths = None
            object_ids = None

    # Load tokenizer and LLM weights
    tokenizer_path, model_path, new_tokens, no_split_module_classes = get_model_details(configs["model_type"])
    os.makedirs(configs["offload_dir"], exist_ok=True)
    f = open(configs["gpu_config"])
    gpu_config = json.load(f)
    model = load_mllm(configs, tokenizer_path, model_path, new_tokens, no_split_module_classes, peft, device, gpu_config, exp_id=exp_id)
    tokenizer = model.tokenizer

    # Load datasets
    if configs["use_clip"]:
        image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    test = len(configs["test_files"]) > 0
    if train:
        train_dataset = TactileLLMDataset(image_processor, configs["train_files"], split_name="train", tokenizer=tokenizer, frame_size=configs["frame_size"], flip_p=configs["flip_p"], model_type=configs["model_type"])
        train_loader = DataLoader(train_dataset, batch_size=configs["per_device_batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    if test:
        
        test_dataset = TactileLLMDataset(image_processor, configs["test_files"], split_name="test", tokenizer=tokenizer, frame_size=configs["frame_size"], flip_p=configs["flip_p"], model_type=configs["model_type"])
        test_loader = DataLoader(test_dataset, batch_size=configs["per_device_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)
    if reason:
        reasoning_dataset = TactileLLMDataset(image_processor, configs["reasoning_files"], split_name="test", tokenizer=tokenizer, frame_size=configs["frame_size"], flip_p=configs["flip_p"], model_type=configs["model_type"], rag=configs["rag"], tactile_vificlip=model.tactile_vificlip, saved_embeddings=saved_embeddings, sample_tactile_paths=sample_tactile_paths, object_ids=object_ids, device=device, retrieval_object_num=configs["retrieval_object_num"])
        reasoning_loader = DataLoader(reasoning_dataset, batch_size=configs["per_device_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)
    
    # Training parameters
    # 1) LLM
    if train:
        llm_params = []
        if not peft:
            for name, param in model.llm.named_parameters():
                # NOTE: no lm_head here since they are not tied to word embeddings in LLaMA and no new tokens for generation
                if "vicuna" in configs["model_type"]:
                    if "lora" in name or "embed_tokens" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "lora" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                if param.requires_grad:
                    llm_params.append(param)
        else:
            for name, param in model.llm.named_parameters():
                if param.requires_grad:
                    llm_params.append(param)
        if len(llm_params) > 0:
            optimizer_llm = torch.optim.AdamW(llm_params, lr=configs["llm_lr"])
            num_steps = int(len(train_loader) / configs["llm_gradient_accumulation_steps"])
            scheduler_llm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_llm, T_max=num_steps)

    # 2) Encoder
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False

    # 3) Projection
    for name, param in model.project.named_parameters():
        param.requires_grad = not configs["freeze_projection"]
    if not configs["freeze_projection"]:
        project_params = model.project.parameters()
        if train and len(configs["train_files"]) > 0:
            optimizer_project = torch.optim.AdamW(project_params, lr=configs["projection_lr"])
            scheduler_project = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_project, T_max=len(train_loader) / configs["llm_gradient_accumulation_steps"])

    # Training
    if train:
        model.train()
        model.encoder.eval()
        # get trainable/non-trainable model parameter stats
        trainable_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])
        all_params = sum([np.prod(p.size()) for p in model.parameters()])
        if peft:
            max_train_steps = configs["max_peft_train_steps"]
        else:
            max_train_steps = configs["max_train_steps"]
        if max_train_steps < len(train_loader):
            print(f"\nFinetuning LLM for {max_train_steps} samples and {int(max_train_steps / configs['llm_gradient_accumulation_steps'])} gradient updates...")
        else:
            print(f"\nFinetuning LLM for {len(train_loader)} samples and {int(len(train_loader) / configs['llm_gradient_accumulation_steps'])} gradient updates...")
        print('Trainable params: {} ({:.2f}%)'.format(trainable_params, trainable_params / all_params * 100,))
        # total_train_loss = 0
        for train_sample_step, batch in enumerate(t:=tqdm.tqdm(train_loader)):
            question, chat, answer_tokens, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict = batch
            answer_tokens = answer_tokens.to(device)
            outputs, _ = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_datasets=all_datasets, all_indices=all_indices)
            train_loss = outputs.loss.detach().float()
            t.set_description(f"Train loss: {train_loss}")
            loss = outputs.loss / configs["llm_gradient_accumulation_steps"]
            loss.backward()
            if (train_sample_step + 1) % configs["llm_gradient_accumulation_steps"] == 0:
                # optimizer updates
                if not configs["freeze_projection"]:
                    optimizer_project.step()
                    scheduler_project.step()
                    optimizer_project.zero_grad()
                if len(llm_params) > 0:
                    optimizer_llm.step()
                    scheduler_llm.step()
                    optimizer_llm.zero_grad()
            if (train_sample_step + 1) >= max_train_steps:
                break
        print("Saving tokenizer and models...")
        tokenizer.save_pretrained(f"{configs['exps_path']}/{exp_id}/tokenizer")
        if peft:
            model.llm.save_pretrained(f"{configs['exps_path']}/{exp_id}/llm_weights_peft")
        else:
            model.llm.save_pretrained(f"{configs['exps_path']}/{exp_id}/llm_weights")
        torch.save(model.encoder.state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_encoder.pt")
        torch.save(model.project.state_dict(), f"{configs['exps_path']}/{exp_id}/project.pt")
        torch.save(model.tactile_vificlip.state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_vificlip.pt")
        if os.path.exists(os.path.join(configs['load_exps_path'], "prompt_learning.yaml")):
            shutil.copy(os.path.join(configs['load_exps_path'], "prompt_learning.yaml"), f"{configs['exps_path']}/{exp_id}/prompt_learning.yaml")
        print(f"LLM finetuning done!")

    # Testing
    if test:
        print(f"\nTesting LLM on the test set for description / ranking...")
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()
        preds = []
        with torch.no_grad():
            for test_sample_step, batch in enumerate(tqdm.tqdm(test_loader)):
                all_objects, sample_paths = [], []
                # NOTE: hardcoded for batch size of 1
                question, chat, answer_tokens, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict = batch
                answer_tokens = answer_tokens.to(device)
                _, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_datasets=all_datasets, all_indices=all_indices, question_embeds_only=True)
                generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None)
                generation = tokenizer.decode(generation_tokens[0], skip_special_tokens=True).strip()
                answer_tokens = answer_tokens[0].cpu().numpy()
                answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                generation = generation.strip().split(tokenizer.eos_token)[0].strip()
                for i in tactile:
                    sample_paths.append(i[0])
                    data = json.load(open(os.path.join("/".join(i[0].split("/")[:-1]), "data.json"), "r"))
                    all_objects.append(data["object_id"])
                preds.append({
                    "sample_paths": sample_paths,
                    "all_objects": all_objects,
                    "question": question,
                    "final_true_answer": answer,
                    "final_generation": generation,
                })
            if peft:
                preds_json_path = f'{configs["exps_path"]}/{exp_id}/preds/llm_peft.json'
            else:
                preds_json_path = f'{configs["exps_path"]}/{exp_id}/preds/llm.json'
            with open(preds_json_path, 'w') as f:
                json.dump(preds, f, indent=4)
                f.close()
        print(f"LLM testing done!")

    # Reasoning
    if reason:
        print(f"\nRunning LLM on the reasoning set...")
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()
        all_reason = {}
        sample_no = {}
        with torch.no_grad():
            for reasoning_sample_step, batch in enumerate(tqdm.tqdm(reasoning_loader)):
                all_objects, sample_paths = [], []
                # NOTE: hardcoded for batch size of 1
                chat, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict, scenario, scenario_steps, target, num_candidates, rag_outputs = batch
                generated_chat = []
                scenario = f"{scenario[0]}_{target[0]}"
                check_scenario = False
                if scenario_steps.item() == 0:
                    continue
                if configs["scenarios"] is not None:
                    for scenario_to_check in configs["scenarios"]:
                        if scenario_to_check in scenario:
                            check_scenario = True
                            break
                    if not check_scenario:
                        continue
                if scenario not in all_reason.keys():
                    all_reason[scenario] = []
                    sample_no[scenario] = 1
                else:
                    sample_no[scenario] += 1
                if configs["answer_step_idx"] is not None:
                    chat = chat[:int(configs["answer_step_idx"]*2)]
                for c in range(len(chat)-1):
                    # NOTE: Only for batch size = 1
                    chat[c] = {k:v[0] for k,v in chat[c].items()}
                    if c % 2 == 0:
                        # Question
                        # NOTE: Only for batch size = 1
                        generated_chat.append(chat[c])
                    else:
                        # Answer
                        answer_idx = int((c-1)/2)
                        # RAG and change descriptions
                        if answer_idx in configs["generate_idx"]:
                            # Generate
                            question = [tokenizer.apply_chat_template(generated_chat, tokenize=False, add_generation_prompt=True)]
                            _, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=None, all_datasets=all_datasets, all_indices=all_indices, question_embeds_only=True)
                            # Descriptions / rankings
                            generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
                            generation = tokenizer.decode(generation_tokens.sequences[0], skip_special_tokens=True).strip()
                            generation = generation.strip().split(tokenizer.eos_token)[0].strip()
                            chat[c]["generate"] = True
                            chat[c]["true_answer"] = chat[c]["content"]
                            chat[c]["content"] = generation
                            generated_chat.append(chat[c])
                        else:
                            chat[c]["generate"] = False
                            generated_chat.append(chat[c])
                        # NOTE: Only for one touched object in the guessing game
                        if answer_idx == 0 and configs["rag"]: # NOTE: Assume generate_idx=0 is the description
                            chat[c]["content"] += "\nMost similar objects (in order of decreasing similarity):"
                            for obj_name, obj_descriptions in rag_outputs[0].items():
                                if configs["rag_use_descriptions"]:
                                    chat[c]["content"] += f" {obj_name} ({', '.join(sorted([i[0] for i in obj_descriptions]))});"
                                else:
                                    chat[c]["content"] += f" {obj_name};"
                final_question = [tokenizer.apply_chat_template(generated_chat, tokenize=False, add_generation_prompt=True)]
                final_true_answer = chat[-1]["content"][0]
                _, question_embeds = model(question=final_question, tactile_frames=tactile_frames, answer_tokens=None, all_datasets=all_datasets, all_indices=all_indices, question_embeds_only=True)
                if configs["reasoning_sampling_num"] == 1:
                    generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
                    final_generation = tokenizer.decode(generation_tokens.sequences[0], skip_special_tokens=True).strip()
                    final_generation = final_generation.strip().split(tokenizer.eos_token)[0].strip()
                else:
                    generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=True, temperature=configs["reasoning_temperature"], num_return_sequences=configs["reasoning_sampling_num"], top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
                    option_generations = {}
                    option_counts = {}
                    option_entropies = {}
                    if configs["reasoning_selection_type"] == "best_of_n":
                        entropies = get_sentence_entropy(generation_tokens, token_start_index=0)
                        max_avg_entropy_per_token = max([i["avg_entropy_per_token"] for i in entropies])
                    for seq_idx, seq in enumerate(generation_tokens.sequences):
                        generation = tokenizer.decode(seq, skip_special_tokens=True).strip()
                        generation = generation.strip().split(tokenizer.eos_token)[0].strip()
                        option = generation.replace("*", "").split("Answer: ")[-1][0]
                        if option not in ["A", "B", "C"]:
                            # print(generation) # For debugging
                            continue
                        if option not in option_generations.keys():
                            option_generations[option] = [generation]
                            option_counts[option] = 1
                            if configs["reasoning_selection_type"] == "best_of_n":
                                option_entropies[option] = [(max_avg_entropy_per_token - entropies[seq_idx]["avg_entropy_per_token"]) / max_avg_entropy_per_token]
                        else:
                            option_generations[option].append(generation)
                            option_counts[option] += 1
                            if configs["reasoning_selection_type"] == "best_of_n":
                                option_entropies[option].append((max_avg_entropy_per_token - entropies[seq_idx]["avg_entropy_per_token"]) / max_avg_entropy_per_token)
                    if configs["reasoning_selection_type"] == "majority_voting":
                        # Get random generation from best option
                        most_common_option = max(option_counts, key=option_counts.get)
                        final_generation = random.choice(option_generations[most_common_option])
                    elif configs["reasoning_selection_type"] == "best_of_n":
                        # Weigh with average entropy per token
                        best_option = max(option_entropies, key=lambda k: sum(option_entropies[k]))
                        final_generation = option_generations[best_option][option_entropies[best_option].index(max(option_entropies[best_option]))]
                all_reason[scenario].append({
                    "sample_no": sample_no[scenario],
                    "sample_paths": sample_paths,
                    "all_objects": all_objects,
                    "num_candidates": num_candidates.item(),
                    "chat": generated_chat,
                    "generate_idx": configs["generate_idx"],
                    "answer_step_idx": configs["answer_step_idx"],
                    "reasoning_sampling_num": configs["reasoning_sampling_num"],
                    "reasoning_selection_type": configs["reasoning_selection_type"],
                    "final_true_answer": final_true_answer,
                    "final_generation": final_generation,
                    "option_counts": option_counts,
                    "option_entropies": {k: sum(v) for k, v in option_entropies.items()},
                })
            # Prioritize PEFT LLM (if any)
            if os.path.exists(os.path.join(configs["load_exp_path"], "llm_weights_peft")) or os.path.exists(os.path.join(f'{configs["exps_path"]}/{exp_id}/', "llm_weights_peft")):
                peft = True
            else:
                peft = False
            # Save predictions by scenario
            for scenario in all_reason.keys():
                if peft:
                    reason_json_path = f'{configs["exps_path"]}/{exp_id}/reason/{scenario}_peft.json'
                else:
                    reason_json_path = f'{configs["exps_path"]}/{exp_id}/reason/{scenario}.json'
                with open(reason_json_path, 'w') as f:
                    json.dump(all_reason[scenario], f, indent=4)
                    f.close()
        print(f"LLM reasoning done!")
    
    # Clean up
    del model
    with torch.no_grad():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_type = f"run"
    config_path = f'configs/{run_type}.yaml'
    # get configs
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    exp_name = input("\nExperiment name: ")
    if len(exp_name) == 0:
        exp_name = "debug"
    exp_id = "_llm"
    if len(configs["train_files"]) > 0:
        exp_id += "_train"
    if len(configs["test_files"]) > 0:
        exp_id += "_test"
    if len(configs["reasoning_files"]) > 0:
        exp_id += "_reason"
    if len(exp_name) > 0:
        exp_id = exp_id + f"_{exp_name}"

    # Make experiment folder
    now = datetime.now()
    exp_date = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_id = exp_date + exp_id
    os.makedirs(f"{configs['exps_path']}", exist_ok=True)
    os.makedirs(f"{configs['exps_path']}/{exp_id}", exist_ok=True)
    if len(configs["train_files"]) > 0 or len(configs["test_files"]) > 0:
        os.makedirs(f"{configs['exps_path']}/{exp_id}/preds", exist_ok=True)
    if len(configs["reasoning_files"]) > 0:
        os.makedirs(f"{configs['exps_path']}/{exp_id}/reason", exist_ok=True)
    with open(f"{configs['exps_path']}/{exp_id}/{run_type}.yaml", 'w') as file:
        documents = yaml.dump(configs, file, sort_keys=False)
        file.close()

    # Seed
    torch.manual_seed(configs["seed"])
    torch.random.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    # torch.use_deterministic_algorithms(True)
    random.seed(configs["seed"])
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(configs["seed"])
    device = f'cuda:{configs["cuda"]}' # for non-LLM models

    # Training and/or testing
    if len(configs["train_files"]) > 0:
        run_llm(configs, exp_id, g, device, peft=configs["peft"])
    elif len(configs["test_files"]) > 0 or len(configs["reasoning_files"]) > 0:
        run_llm(configs, exp_id, g, device, peft=configs["peft"])