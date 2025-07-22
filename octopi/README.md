# Octopi-S
## Setup
**For the steps below, ensure you are in the root directory `octopi-s/` unless otherwise stated.**

### Environment
1. In a conda environment with PyTorch / CUDA available, run `pip install -r requirements.txt` to install all dependencies.
2. Install Uvicorn for the API using `sudo apt-get install uvicorn`.
3. We recommend 17GiB max memory for each GPU for the two RTX 5000 Ada Generation GPUs in `configs/gpu_config.json`.

### Weights
1. Download Octopi-S [model weights](https://drive.google.com/file/d/1YMn6V5W-_qvDlCbVSdZiufe729BOEl-A/view?usp=sharing).
2. Unzip and put the weights in `octopi_s/data/` as `octopi_s/data/weights/`.


## Quickstart
1. Set configs in `configs/demo.yaml`.
    * Absolute paths are preferred.
    * If you want to use RAG and have not generated the embeddings yet, set `rag: True` and `rag_generate_embeddings: True`. Set `rag_generate_embeddings: False` after the embeddings have been generated unless you want to regenerate them.
2. Set `load_exp_path: octopi_s/data/weights` to use our model weights.
3. For a `demo_path: ../data/demo` and `image_path: ../data/demo/rgb.png`, structure your directory like:
```
├── configs
│   └── ...
├── data
│   ├── demo
│   │   ├── 1
│   │   │   └── item.mov
│   │   ├── 2
│   │   │   ├── 1
│   │   │   │   └── item.mov
│   │   │   └── 2
│   │   │       └── item.mov
│   │   ├── ...
│   │   └── rgb.png
|   ├── embeddings
│   │   ├── physiclear_0.pt
│   │   └── ...
│   ├── llm_qa
│   │   ├── test_description_comparison_qa_{ID}.json
│   │   ├── test_samples.json
│   │   ├── test_scenario_qa_{ID}.json
│   │   ├── train_description_comparison_qa_{ID}.json
│   │   ├── train_samples.json
│   │   └── val_samples.json
│   ├── samples
│   │   ├── physiclear_0
│   │   │   ├── tactile
│   │   │   │   ├── XXX.jpg
│   │   │   │   └── ...
│   │   │   └── data.json
│   │   └── ...
│   └── tactile_datasets
│       ├── physiclear
│       │   └── ...
│       └── physicleardotted
│           └── ...
├── octopi_s
│   └── ...
└── ...
```
where `../data/demo/1` contains the tactile video of an object with only one unique part (texture-wise) while `../data/demo/2` is an object with two unique parts.

### Notebook
1. Change directory into `octopi_s/`.
2. Load `quickstart.ipynb`.
3. Run all cells.
4. Query the LLM using the pop-up box.
    * `$d(1,2)` to describe objects (`../data/demo/1`, `../data/demo/2`).
    * `$r(1,3)` to rank objects (`../data/demo/1`, `../data/demo/3`).
    * `$dr(3,2)` to describe and rank objects (`../data/demo/3`, `../data/demo/2`).
    * `restart` to restart the conversation.
    * `exit` to exit the conversation.

### API
1. Change directory into `octopi_s/`.
2. Run `uvicorn demo:app --host=0.0.0.0 --port=8000 --log-level=debug --reload`.
3. Open an API platform like Postman.
4. Refer to the [API documentation](https://github.com/clear-nus/octopi-s/wiki/API) for more information on usage.


## Training / Testing
### Processing PhysiCLeAR Datasets Into Salient Frames
1. Create the directory `octopi_s/data/tactile_datasets`.
2. Download our [tactile datasets](https://drive.google.com/file/d/1ckSzE4DxSiq4U34gWBIUGreImryLw94c/view?usp=drive_link).
3. Unzip and put the tactile datasets in `octopi_s/data/` as `octopi_s/data/tactile_datasets/`.
4. Run `python octopi_s/process_datasets.py --dataset_path octopi_s/data/tactile_datasets` to extract salient frame spans and generate data files mapping objects to their sample folder(s).

### Generating Question-Answer (QA) Files
1. Make sure the previous step is fully completed before you proceed to this step.
2. Set configs in `configs/generate_qa.yaml`.
3. Run `python octopi_s/generate_qa.py`.
4. Enter the scenario QA ID you want when prompted to make the QA files easily identifiable.
5. Three QA files will be generated in `output_data_dir` as `train_description_ranking_qa_{ID}.json` (description / ranking training), `test_description_ranking_qa_{ID}.json` (description / ranking testing) and `test_scenario_qa_{ID}.json` (scenario testing).

### Testing Multimodal LLM
1. Set configs in `configs/run.yaml`.
    * Set `load_exp_path: octopi_s/data/weights` to use our model weights.
    * Put at least one QA file absolute path in `test_files` and / or `reasoning_files` for it to test / reason.
    * Set `train_files: []` to skip training.
    * If you want to use RAG and have not generated the embeddings yet, set `rag: True` and `rag_generate_embeddings: True`. Set `rag_generate_embeddings: False` after the embeddings have been generated unless you want to regenerate them.
2. Run `python octopi_s/run_llm.py`.
3. Enter the experiment ID you want when prompted to make the experiment directory easily identifiable.
4. After you have generated prediction JSON file(s) for ranking and/or scenario reasoning, run `python octopi_s/evaluate_llm.py --llm_preds_path {path/to/results.json}` to get prediction results in your terminal.

### Training Multimodal LLM
1. Set configs in `configs/run.yaml`.
    * Put at least one QA file absolute path in `train_files` for it to train.
    * Set `test_files` and / or `reasoning_files` if you want it to test / reason as well.
    * Set `load_exp_path` if you want to start from an encoder checkpoint (highly recommended), else set as `null`.
    * If you want to use RAG for testing / reasoning and have not generated the embeddings yet, set `rag: True` and `rag_generate_embeddings: True`. Set `rag_generate_embeddings: False` after the embeddings have been generated unless you want to regenerate them.
2. Run `python octopi_s/run_llm.py`.
3. Enter the experiment ID you want when prompted to make the experiment directory easily identifiable.
4. If you have set `test_files` and / or `reasoning_files`, run `python octopi_s/evaluate_llm.py --llm_preds_path {path/to/results.json}` on the generated prediction JSON file(s) to get prediction results in your terminal.