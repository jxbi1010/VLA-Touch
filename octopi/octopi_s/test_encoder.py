import os
import torch
import json
import numpy as np
import yaml
import tqdm
from torch.utils.data import DataLoader
from utils.dataset import TactilePropertyRegressionDataset, regression_collate_fn
from utils.encoder import CLIPVisionEncoder, ViFiCLIP, Adapter, PropertyClassifier, load_encoder
from transformers import CLIPImageProcessor, AutoTokenizer
import pickle


def load_model_configs(exp_path):
    """Load configuration from the experiment directory."""
    with open(os.path.join(exp_path, "run.yaml"), 'r') as file:
        configs = yaml.safe_load(file)
    return configs

# def load_models(configs, exp_path, device):
#     """Load trained models from experiment directory."""
#     # Load model components
#     tactile_encoder = CLIPVisionEncoder(clip_model=configs["use_clip"]).to(device)
    
#     # Load ViFiCLIP
#     if configs["prompt_learning"]:
#         # If prompt learning was used, you may need to load those configs
#         with open(os.path.join(exp_path, "prompt_learning.yaml"), 'r') as f:
#             prompt_learning_configs = yaml.safe_load(f)
#         # Initialize with prompt learning configs
#         from utils.llm import PromptLearningCLIPModel
#         clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], prompt_learning_configs)
#     else:
#         # Standard initialization
#         from transformers import CLIPModel
#         clip = CLIPModel.from_pretrained(configs["use_clip"])
    
#     # Initialize model components
#     tactile_vificlip = ViFiCLIP(clip, freeze_text_encoder=True, use_positional_embeds=True).to(device)
#     plain_tactile_adapter = Adapter(
#         input_size=configs["dim_context_vision"], 
#         output_size=configs["dim_context_vision"], 
#         residual_ratio=configs["residual_ratio"]
#     ).to(device)
#     property_classifier = PropertyClassifier(input_size=configs["dim_context_vision"]).to(device)
    
#     # Load saved weights
#     tactile_vificlip.load_state_dict(torch.load(os.path.join(exp_path, "tactile_vificlip.pt"), map_location=device))
#     plain_tactile_adapter.load_state_dict(torch.load(os.path.join(exp_path, "plain_tactile_adapter.pt"), map_location=device))
#     property_classifier.load_state_dict(torch.load(os.path.join(exp_path, "property_classifier.pt"), map_location=device))
    
#     models = {
#         "tactile_vificlip": tactile_vificlip,
#         "plain_tactile_adapter": plain_tactile_adapter,
#         "property_classifier": property_classifier
#     }
    
#     return models

def evaluate(models, dataloader, device):
    """Evaluate models on the given dataloader."""
    models["tactile_vificlip"].eval()
    models["property_classifier"].eval()
    models["dotted_tactile_adapter"].eval()
    
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    mse_loss_fn = torch.nn.MSELoss()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_paths = []
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            all_tactile_frames, properties, dataset, paths = batch
            all_labels.append(properties.cpu().numpy())
            all_paths.append(paths)
            batch_size = all_tactile_frames.shape[0]
            num_samples += batch_size
            
            # Forward pass through models
            all_tactile_frames = all_tactile_frames.to(device)
            tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames, None, None)
            plain_tactile_video_features = models["dotted_tactile_adapter"](tactile_video_features)
            prop_preds = models["property_classifier"](plain_tactile_video_features)
            
            # Calculate loss
            # loss = ce_loss_fn(prop_preds, properties.squeeze(1).to(device).long())
            loss = mse_loss_fn(prop_preds, properties.squeeze(1).to(device))
            total_loss += loss.item() * batch_size
            
            # Store predictions
            all_preds.append(prop_preds.cpu().numpy())
    
    # Process predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_paths = np.concatenate(all_paths, axis=0)
    
    # Get predicted classes
    # pred_classes = np.argmax(all_preds, axis=1)
    # true_classes = all_labels.squeeze(1)

    # wipe: 100% success rate
    pred_classes = all_preds[:, 1] > 7
    true_classes = all_labels[:, 1] > 7

    # mango
    # thres = 3.0
    # pred_classes = all_preds[:, 0] > thres
    # true_classes = all_labels[:, 0] > thres
    
    # Calculate accuracy
    accuracy = np.mean(pred_classes == true_classes)
    hard = []
    soft = []
    # print("all_labels", all_labels)
    # print("all_preds", all_preds)
    for i in range(num_samples):
        print("label:", all_labels[i], "preds:", all_preds[i])
        # print(true_classes[i], pred_classes[i])
        # if all_labels[i][0] == 1:
        #     pink.append(all_preds[i])
        # else:
        #     brown.append(all_preds[i])
        if all_labels[i][0] == 6:
            hard.append((all_preds[i], all_paths[i]))
        else:
            soft.append((all_preds[i], all_paths[i]))

    return {
        "total_loss": total_loss,
        "num_samples": num_samples,
        "avg_loss": total_loss / num_samples,
        "accuracy": accuracy
    }, hard, soft #, pink, brown

def main():
    # Path to your experiment directory containing the trained models
    # exp_path = input("Enter path to experiment directory: ")
    exp_path = "/home/allenbi/PycharmProjects24/octopi/octopi-s-main/octopi-s-main/configs"
    # Example: "/path/to/exps/2025_04_28_12_34_56_train_encoder_distributed_my_experiment"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configurations
    configs = load_model_configs(exp_path)
    print("Loaded configurations")
    
    # Create data loader for test set
    image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    tokenizer = AutoTokenizer.from_pretrained(configs["use_clip"])
    
    # Create test dataset and dataloader
    test_dataset = TactilePropertyRegressionDataset(
        image_processor=image_processor, 
        tokenizer=tokenizer, 
        # data_path=configs["data_dir"], 
        # data_path="/home/allenbi/PycharmProjects24/octopi/octopi-s-main/octopi-s-main/data/wipe_samples_span",
        data_path="/home/allenbi/PycharmProjects24/octopi/octopi-s-main/octopi-s-main/data/online_inference",
        split_name="test", 
        # datasets=configs["datasets"],
        datasets=["wipe"],
        # datasets=["mango"],
        frame_size=configs["frame_size"]
    )
    
    test_loader = DataLoader(
        test_dataset, 
        # batch_size=configs["batch_size"], 
        batch_size=16,
        shuffle=False, 
        collate_fn=regression_collate_fn
    )
    
    print(f"Created test dataloader with {len(test_dataset)} samples")
    
    # Load models
    # models = load_models(configs, exp_path, device)
    tactile_vificlip, dotted_tactile_adapter, plain_tactile_adapter, property_classifier, load_exp_configs = load_encoder(configs, device)
    models = {
        "tactile_vificlip": tactile_vificlip,
        "dotted_tactile_adapter": dotted_tactile_adapter,
        "property_classifier": property_classifier
    }
    print("Models loaded successfully")
    
    # Evaluate on test set
    print("Starting evaluation...")
    # results, pink, brown = evaluate(models, test_loader, device)
    results, hard, soft = evaluate(models, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Test Loss: {results['avg_loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # Save results to file
    results_file = os.path.join(exp_path, "test_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_file}")

    # np.save("pink.npy", np.asarray(pink))
    # np.save("brown.npy", np.asarray(brown))
    # for i in range(len(pink)):
    #     input("\nEnter to get new pair")
    #     print('pink:', pink[i], 'brown:', brown[i])
    
    # hard = np.asarray(hard)
    # soft = np.asarray(soft)
    # np.save("hard.npy", hard)
    # np.save("soft.npy", soft)
    # print(np.mean(hard[:, 0]), np.mean(soft[:, 0]))
    # with open('hard.pkl', 'wb') as f:
    #     pickle.dump(hard, f)
    # with open('soft.pkl', 'wb') as f:
    #     pickle.dump(soft, f)
    n_success = 0
    for i in range(len(hard)):
        # input("\nEnter to get new pair")
        print('hard:', hard[i][0], 'soft:', soft[i][0])
        if hard[i][0][0] > soft[i][0][0]:
            n_success += 1

    print('n_success', n_success)

if __name__ == "__main__":
    main()