# Tactile VLA Pipeline - Complete Guide

This guide covers the complete workflow for running tactile-based robotic experiments, from generating encoder outputs to executing VLA (Vision-Language-Action) tasks.

## Overview

The pipeline consists of two main components:

1. **`test_encoder.py`**: Tests and generates encoder outputs from tactile datasets
2. **`touch_vla.py`**: Runs VLA experiments using encoder outputs or raw tactile images

## Supported Experiments

- **`wipe`**: Smoothness classification (pink/smooth vs brown/rough sponges)
- **`mango`**: Hardness classification (soft/ripe vs hard/unripe mangoes)
- **`cup`**: Force magnitude measurement (empty vs full cups)

---

## Part 1: Test Encoder (`test_encoder.py`)

### Purpose
Evaluates trained tactile encoders on test datasets and generates property classification outputs saved as `.npy` files.

### Usage

```bash
python test_encoder.py --experiment <EXPERIMENT> [OPTIONS]
```

### Arguments

- `--experiment` (required): Choose from `wipe` or `mango`
- `--exp-path`: Path to experiment directory with trained models (default: configs path)
- `--batch-size`: Batch size for evaluation (default: 16)
- `--output-dir`: Directory to save .npy files (default: current directory)

### Examples

```bash
# Test wipe encoder (smoothness)
python test_encoder.py --experiment wipe --batch-size 16

# Test mango encoder (hardness)
python test_encoder.py --experiment mango --output-dir ./encoder_outputs/

# Custom experiment path
python test_encoder.py --experiment wipe --exp-path /path/to/trained/models
```

### What It Does

1. Loads trained tactile encoder models
2. Processes test dataset through the encoder
3. Generates property predictions (hardness, roughness/smoothness)
4. Saves outputs to `.npy` files:
   - **Wipe**: `pink.npy`, `brown.npy`
   - **Mango**: `soft.npy`, `hard.npy`
5. Calculates accuracy and success rates
6. Saves evaluation results to JSON

### Output Files

After running, you'll get:
- `{category}.npy`: Encoder predictions for each category
- `test_evaluation_results_{experiment}.json`: Detailed evaluation metrics

### Example Output

```
Loaded configurations
Created test dataloader with 20 samples
Models loaded successfully

Starting evaluation...
label: [5.2 8.1], preds: [5.3 7.9]
label: [6.1 9.2], preds: [6.0 9.1]
...

============================================================
Evaluation Results:
Test Loss: 0.1234
Test Accuracy: 0.95 (95.00%)
Total samples: 20
pink samples: 10
brown samples: 10
============================================================

Saved pink predictions to pink.npy
Saved brown predictions to brown.npy
Pairwise comparison success: 9/10 (90.00%)
```

---

## Part 2: Touch VLA (`touch_vla.py`)

### Purpose
Runs robotic tactile experiments using VLA models with manual interactive control through a terminal interface.

### Usage

```bash
python touch_vla.py --experiment <EXPERIMENT> [--api-key YOUR_KEY]
```

### Required Arguments

- `--experiment`: Choose from `wipe`, `mango`, or `cup`

### Optional Arguments

- `--api-key STR`: OpenAI API key (default: uses key in source)

### Operation Mode

**Manual Interactive Mode**
- User provides initial scene image at startup
- User manually inputs tactile data during session:
  - Episode numbers to load tactile images
  - Force data for cup experiment
  - Manual property values
  - Custom image paths
- LLM analyzes tactile sensor images or force data
- Full control over experiment flow
- Single session per run

### Examples

#### 1. Wipe Experiment

```bash
# Run wipe experiment
python touch_vla.py --experiment wipe

# At startup, provide initial image
Enter initial image path: /users/kevinma/tactile/dataset/wipe/episode_0/camera1/rgb_0.jpg
```

**During session:**
```
[Step 1] Assistant: I will touch the left sponge to analyze its smoothness...

Available inputs:
  'end' - End the session
  '<episode_num>' - Get tactile images for episode (e.g., '5')
  '<hardness>,<roughness>' - Manually input property values (e.g., '5.2,8.3')
  '<image_path>' - Provide path to tactile images

Your input: 0
[Loads tactile images from episode_0/gelsight]

[Step 2] Assistant: The left sponge appears smoother. I will now touch the right sponge...

Your input: 1
[Loads tactile images from episode_1/gelsight]

[Step 3] Assistant: Based on the analysis, I will pick up the left sponge as it is smoother.

Your input: end
Session saved to results/wipe_results.jsonl
```

#### 2. Mango Experiment

```bash
python touch_vla.py --experiment mango
```

**Manual property input example:**
```
Your input: 5.2,8.3
[Provides: "The hardness level is: 5.2, The roughness level is: 8.3"]
```

#### 3. Cup Experiment

```bash
python touch_vla.py --experiment cup
```

**Force magnitude input example:**
```
Your input: force:5
[Loads force data from episode_5/gelsight_force.npy]
[Provides: "The tactile shear force vector has magnitude: 1.05, xy-direction: [0.12, -0.08]..."]

# Or manual force input:
Your input: force
Enter force magnitude: 1.05
Enter x-direction: 0.12
Enter y-direction: -0.08
```

---

## Complete Workflow

### Workflow 1: Using Encoder Outputs (Optional)

```bash
# Step 1: Generate encoder outputs (optional - for reference/validation)
python test_encoder.py --experiment wipe

# Step 2: Run VLA experiment with manual input
python touch_vla.py --experiment wipe

# Manually input episode numbers or property values during session
```

### Workflow 2: Direct VLA Execution

```bash
# Run VLA directly without encoder outputs
python touch_vla.py --experiment mango

# Provide initial image, then input episode numbers as prompted
```

### Workflow 3: Cup Experiment (Force-Based)

```bash
# Cup experiment requires force magnitude input
python touch_vla.py --experiment cup

# Use "force:N" to load force data or "force" for manual input
```

---

## Dataset Structure

Both scripts expect datasets in this structure:

```
/users/kevinma/tactile/dataset/
├── wipe/
│   └── episode_0/
│       ├── camera1/
│       │   └── rgb_0.jpg        # Initial scene image
│       └── gelsight/             # Tactile sensor images
│           ├── 0000.jpg          # Before touch
│           └── 0001.jpg          # After touch
├── mango_touch/
│   └── episode_0/
│       ├── camera1/rgb_0.jpg
│       └── gelsight/
└── water_cup_new/
    └── episode_0/
        ├── camera1/rgb_0.jpg
        ├── gelsight/
        └── gelsight_force.npy    # Force magnitude data
```

---

## Required Files

### For test_encoder.py
- Trained model files in experiment directory:
  - `tactile_vificlip.pt`
  - `dotted_tactile_adapter.pt`
  - `property_classifier.pt`
  - `run.yaml`

### For touch_vla.py
- Dataset with tactile images (episode-based structure)
- Initial scene images (camera1/rgb_0.jpg)
- OpenAI API key
- For cup experiment: gelsight_force.npy files

---

## Output Files

### test_encoder.py Outputs
```
pink.npy                              # Encoder predictions for pink/smooth
brown.npy                             # Encoder predictions for brown/rough
soft.npy                              # Encoder predictions for soft/ripe
hard.npy                              # Encoder predictions for hard/unripe
test_evaluation_results_wipe.json    # Evaluation metrics
test_evaluation_results_mango.json   # Evaluation metrics
```

### touch_vla.py Outputs
```
results/
├── wipe_results.jsonl               # Wipe experiment sessions
├── mango_results.jsonl              # Mango experiment sessions
└── cup_results.jsonl                # Cup experiment sessions
```

Each `.jsonl` file contains session logs:
```json
{
  "experiment": "wipe",
  "start_time": "2025-11-19 10:30:00.123456",
  "initial_image": "/users/kevinma/tactile/dataset/wipe/episode_0/camera1/rgb_0.jpg",
  "initial_prompt": "There are two sponges in the image...",
  "steps": [
    {
      "step": 1,
      "assistant": "I will touch the left sponge first to analyze its smoothness...",
      "user_feedback": "Tactile images: /path/to/first.jpg, /path/to/last.jpg"
    },
    {
      "step": 2,
      "assistant": "Based on the tactile feedback, I will pick up the left sponge.",
      "user_feedback": "end"
    }
  ]
}
```

---

## Terminal Interface Guide

### Interactive Input Options

During each VLA session, you can provide different types of input:

#### For Wipe/Mango Experiments
- **`<episode_num>`** - Load tactile images from episode (e.g., `5`)
  - Loads from: `{dataset_base_path}/episode_5/gelsight/`
  - Provides before/after tactile sensor images to LLM
- **`<val1>,<val2>`** - Manual property values (e.g., `5.2,8.3`)
  - Hardness: first value, Roughness: second value
- **`<image_path>`** - Custom tactile image directory path
  - Provide full path to gelsight directory
- **`end`** - Complete session and save results

#### For Cup Experiment
- **`force:<episode_num>`** - Load force data (e.g., `force:5`)
  - Loads from: `{dataset_base_path}/episode_5/gelsight_force.npy`
  - Provides force magnitude and xy-direction
- **`force`** - Manual force input
  - Prompts for: magnitude, x-direction, y-direction
- **`<image_path>`** - Custom tactile image directory path
- **`end`** - Complete session and save results

### Example Session Flow

```
Starting Interactive Session - WIPE
Initial Image: /users/kevinma/tactile/dataset/wipe/episode_0/camera1/rgb_0.jpg

[Step 1] Assistant: I will touch the left sponge to determine its smoothness...

Available inputs:
  'end' - End the session
  '<episode_num>' - Get tactile images for episode (e.g., '5')
  '<hardness>,<roughness>' - Manually input property values (e.g., '5.2,8.3')
  '<image_path>' - Provide path to tactile images

Your input: 0
[Loads and analyzes tactile images from episode_0]

[Step 2] Assistant: The left sponge has a smoothness of 8.1. I will now touch the right sponge...

Your input: 1

[Step 3] Assistant: The right sponge has a smoothness of 6.5. I will pick up the left sponge as it is smoother.

Your input: end

Session saved to results/wipe_results.jsonl
```

---

## Success Criteria (For Reference)

These are the intended task outcomes for each experiment:

- **Wipe**: Robot picks up **left** (pink/smoother) sponge
- **Mango**: Robot picks up **right** (soft/riper) mango  
- **Cup**: Robot puts full cup onto **plate**

*Note: The simplified version does not automatically track success. Review session logs manually to evaluate task completion.*

---

## Troubleshooting

### test_encoder.py Issues

**Error: "Encoder file not found"**
- Ensure model files exist in `--exp-path` directory
- Check that `run.yaml` is present

**Error: "No such file or directory"**
- Verify dataset paths in experiment configs
- Check episode numbers exist in dataset

### touch_vla.py Issues

**Error: "Image not found"**
- Verify the initial image path exists
- Check camera1/rgb_0.jpg is present in episode directory

**Error: "Tactile path not found"**
- Ensure gelsight directories exist for the episode
- Check episode number matches your dataset structure
- Verify dataset_base_path in EXPERIMENTS config is correct

**Error: "Force file not found"**
- Ensure gelsight_force.npy exists for the episode
- Only cup experiment has force files

**No LLM response / Timeout**
- Check OpenAI API key is valid
- Verify internet connection
- Check API usage limits

---

## Quick Reference

```bash
# Generate encoder outputs (optional - for validation)
python test_encoder.py --experiment wipe
python test_encoder.py --experiment mango  

# Run VLA experiments (manual interactive mode)
python touch_vla.py --experiment wipe
python touch_vla.py --experiment mango
python touch_vla.py --experiment cup

# With custom API key
python touch_vla.py --experiment wipe --api-key YOUR_API_KEY
```

### Input Quick Reference

**Episode number:** `5` → Loads tactile images from episode_5  
**Manual properties:** `5.2,8.3` → Hardness 5.2, Roughness 8.3  
**Force from file:** `force:5` → Loads force from episode_5  
**Force manual:** `force` → Prompts for magnitude/direction  
**Image path:** `/path/to/gelsight/` → Custom tactile images  
**End session:** `end` → Save and exit

---

## Configuration

### Experiment Settings

Both scripts use centralized configuration dictionaries. Modify these in the source files to customize:

**test_encoder.py:**
- Property thresholds
- Dataset paths
- Model file locations
- Success criteria

**touch_vla.py:**
- Task prompts (EXPERIMENTS["experiment"]["prompt"])
- Dataset base paths (EXPERIMENTS["experiment"]["dataset_base_path"])
- Tactile analysis prompts
- Force reference values (cup only)
- Output file locations

### Dataset Paths

Default paths (modify in EXPERIMENTS dict if needed):
- Wipe: `/users/kevinma/tactile/dataset/wipe`
- Mango: `/users/kevinma/tactile/dataset/mango_touch`
- Cup: `/users/kevinma/tactile/dataset/water_cup_new`

---

## Performance Tips

1. **Encoder validation**: Run `test_encoder.py` to validate encoder accuracy before VLA experiments
2. **Episode selection**: Start with known episodes to verify dataset structure
3. **Manual vs episode input**: Use episode numbers for consistency, manual input for edge cases
4. **Session management**: Each run creates one session - run multiple times for different scenarios
5. **API costs**: Be mindful of OpenAI API usage - each step sends images to GPT-4o
