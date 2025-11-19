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
import argparse


# Experiment configurations
EXPERIMENTS = {
    "mango": {
        "datasets": ["mango"],
        "data_path": "data/mango_new_samples_span",
        "threshold": 3.0,
        "property_idx": 0,  # First property dimension
        "label_names": ["soft", "hard"],
        "output_files": ["soft.npy", "hard.npy"]
    },
    "wipe": {
        "datasets": ["wipe"],
        "data_path": "data/wipe_samples_span",
        "threshold": 7.0,
        "property_idx": 1,  # Second property dimension
        "label_names": ["pink", "brown"],  # smooth, rough
        "output_files": ["pink.npy", "brown.npy"]
    }
}


def load_model_configs(exp_path):
    """Load configuration from the experiment directory."""
    with open(os.path.join(exp_path, "run.yaml"), 'r') as file:
        configs = yaml.safe_load(file)
    return configs

def evaluate(models, dataloader, device, exp_config):
    """Evaluate models on the given dataloader."""
    models["tactile_vificlip"].eval()
    models["property_classifier"].eval()
    models["dotted_tactile_adapter"].eval()
    
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
            loss = mse_loss_fn(prop_preds, properties.squeeze(1).to(device))
            total_loss += loss.item() * batch_size
            
            # Store predictions
            all_preds.append(prop_preds.cpu().numpy())
    
    # Process predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_paths = np.concatenate(all_paths, axis=0)
    
    # Get predicted classes based on experiment type
    property_idx = exp_config["property_idx"]
    threshold = exp_config["threshold"]
    pred_classes = all_preds[:, property_idx] > threshold
    true_classes = all_labels[:, property_idx] > threshold
    
    # Calculate accuracy
    accuracy = np.mean(pred_classes == true_classes)
    
    # Separate samples by class
    class_0_samples = []  # Below threshold (soft/pink)
    class_1_samples = []  # Above threshold (hard/brown)
    
    for i in range(num_samples):
        print(f"label: {all_labels[i]}, preds: {all_preds[i]}")
        
        if true_classes[i]:  # Above threshold
            class_1_samples.append((all_preds[i], all_paths[i]))
        else:  # Below threshold
            class_0_samples.append((all_preds[i], all_paths[i]))

    return {
        "total_loss": total_loss,
        "num_samples": num_samples,
        "avg_loss": total_loss / num_samples,
        "accuracy": accuracy
    }, class_0_samples, class_1_samples

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test encoder on mango or wipe experiments')
    parser.add_argument('--experiment', type=str, choices=['mango', 'wipe'], required=True,
                        help='Which experiment to run: mango (soft/hard) or wipe (pink/brown)')
    parser.add_argument('--exp-path', type=str, 
                        default="configs",
                        help='Path to experiment directory containing trained models')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default=".",
                        help='Directory to save output .npy files')
    args = parser.parse_args()
    
    # Get experiment configuration
    exp_config = EXPERIMENTS[args.experiment]
    print(f"\n{'='*60}")
    print(f"Running {args.experiment.upper()} experiment")
    print(f"Classes: {exp_config['label_names'][0]} vs {exp_config['label_names'][1]}")
    print(f"{'='*60}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configurations
    configs = load_model_configs(args.exp_path)
    print("Loaded configurations")
    
    # Create data loader for test set
    image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    tokenizer = AutoTokenizer.from_pretrained(configs["use_clip"])
    
    # Create test dataset and dataloader
    test_dataset = TactilePropertyRegressionDataset(
        image_processor=image_processor, 
        tokenizer=tokenizer, 
        data_path=exp_config["data_path"],
        split_name="test", 
        datasets=exp_config["datasets"],
        frame_size=configs["frame_size"]
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        collate_fn=regression_collate_fn
    )
    
    print(f"Created test dataloader with {len(test_dataset)} samples")
    
    # Load models
    tactile_vificlip, dotted_tactile_adapter, plain_tactile_adapter, property_classifier, load_exp_configs = load_encoder(configs, device)
    models = {
        "tactile_vificlip": tactile_vificlip,
        "dotted_tactile_adapter": dotted_tactile_adapter,
        "property_classifier": property_classifier
    }
    print("Models loaded successfully")
    
    # Evaluate on test set
    print("\nStarting evaluation...")
    results, class_0_samples, class_1_samples = evaluate(models, test_loader, device, exp_config)
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results:")
    print(f"Test Loss: {results['avg_loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Total samples: {results['num_samples']}")
    print(f"{exp_config['label_names'][0]} samples: {len(class_0_samples)}")
    print(f"{exp_config['label_names'][1]} samples: {len(class_1_samples)}")
    print(f"{'='*60}\n")
    
    # Save results to JSON file
    results_file = os.path.join(args.exp_path, f"test_evaluation_results_{args.experiment}.json")
    with open(results_file, 'w') as f:
        json.dump({
            **results,
            "experiment": args.experiment,
            "threshold": exp_config["threshold"],
            "property_idx": exp_config["property_idx"],
            "label_names": exp_config["label_names"]
        }, f, indent=4)
    print(f"Results saved to {results_file}")
    
    # Save predictions to .npy files
    class_0_array = np.array([sample[0] for sample in class_0_samples])
    class_1_array = np.array([sample[0] for sample in class_1_samples])
    
    class_0_file = os.path.join(args.output_dir, exp_config["output_files"][0])
    class_1_file = os.path.join(args.output_dir, exp_config["output_files"][1])
    
    np.save(class_0_file, class_0_array)
    np.save(class_1_file, class_1_array)
    print(f"\nSaved {exp_config['label_names'][0]} predictions to {class_0_file}")
    print(f"Saved {exp_config['label_names'][1]} predictions to {class_1_file}")
    
    # Calculate pairwise comparison success rate
    n_success = 0
    n_comparisons = min(len(class_0_samples), len(class_1_samples))
    
    print(f"\nPairwise Comparisons ({n_comparisons} pairs):")
    for i in range(n_comparisons):
        class_0_pred = class_0_samples[i][0]
        class_1_pred = class_1_samples[i][0]
        
        # Check if class_1 (hard/brown) has higher prediction than class_0 (soft/pink)
        if class_1_pred[exp_config["property_idx"]] > class_0_pred[exp_config["property_idx"]]:
            n_success += 1
        
        print(f"Pair {i+1}: {exp_config['label_names'][0]}={class_0_pred[exp_config['property_idx']]:.3f}, "
              f"{exp_config['label_names'][1]}={class_1_pred[exp_config['property_idx']]:.3f}")
    
    pairwise_accuracy = n_success / n_comparisons if n_comparisons > 0 else 0
    print(f"\nPairwise comparison success: {n_success}/{n_comparisons} ({pairwise_accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()