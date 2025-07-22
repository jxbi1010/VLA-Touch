import os
import torch
import yaml
from models.multimodal_encoder.t5_encoder import T5Embedder



class T5_Text_Encoder:

    def __init__(self,gpu=0, model_path="google/t5-v1_1-xxl", config_path="configs/base.yaml", offload_dir=None):

        with open(config_path, "r") as fp:
            config = yaml.safe_load(fp)

        # Setup device
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

        # Initialize text embedder
        text_embedder = T5Embedder(
            from_pretrained=model_path,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=self.device,
            use_offload_folder=offload_dir
        )
        self.tokenizer, self.text_encoder = text_embedder.tokenizer, text_embedder.model

    def compute_instruction_embedding(self, instruction):
        """
        Compute embeddings for a given instruction using T5 model.
        """

        tokens = self.tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(self.device)

        # Reshape tokens and compute embedding
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            embedding = self.text_encoder(tokens).last_hidden_state.detach().cpu()

        print(f'Instruction embedded with shape {embedding.shape}')

        return embedding


def process_episode_instructions(dataset_dir):
    """
    Process all instructions from episode folders, compute embeddings,
    and save them to corresponding episode folders.

    Args:
        dataset_dir (str): Directory containing episode folders
        output_format (str): Format to save embeddings ('npy' or 'json')
    """

    txt_encoder = T5_Text_Encoder()

    # Get all episode directories
    episode_dirs = [f for f in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, f))]

    # Sort episode directories numerically
    episode_dirs.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else float('inf'))

    if not episode_dirs:
        print(f"No episode directories found in {dataset_dir}")
        return

    print(f"Found {len(episode_dirs)} episode directories")

    # List to store all instructions
    all_instructions = []
    # List to store mapping of instruction to episode path
    instruction_episode_map = []

    # First pass: Collect all instructions
    print("Collecting instructions from all episodes...")
    for episode_dir in episode_dirs:
        episode_path = os.path.join(dataset_dir, episode_dir)
        instruction_file = os.path.join(episode_path, "instruction.txt")

        if os.path.exists(instruction_file):
            with open(instruction_file, 'r') as f:
                instruction = f.read().strip()

            if instruction:  # Only add non-empty instructions
                all_instructions.append(instruction)
                instruction_episode_map.append({
                    'instruction': instruction,
                    'episode_path': episode_path,
                    'episode_name': episode_dir
                })
        else:
            print(f"Warning: No instruction.txt found in {episode_dir}")

    # Print summary of collected instructions
    print(f"Collected {len(all_instructions)} instructions from {len(episode_dirs)} episodes")

    print("\nComputing embeddings and saving to episode folders...")

    # Create a dictionary to store unique instructions and their embeddings
    unique_instructions = {'all_instructions': []}
    instruction_counts = {}

    # Count occurrences of each instruction
    for item in instruction_episode_map:
        instruction = item['instruction']
        if instruction in instruction_counts:
            instruction_counts[instruction] += 1
        else:
            instruction_counts[instruction] = 1
    for key in instruction_counts.keys():
        unique_instructions['all_instructions'].append(key)

    print(f"Found {len(instruction_counts)} unique instructions out of {len(instruction_episode_map)} total")

    # First compute embeddings for all unique instructions
    print("\nComputing embeddings for unique instructions...")
    for idx, (instruction, count) in enumerate(instruction_counts.items()):
        print(f"\nComputing embedding {idx + 1}/{len(instruction_counts)}: (used in {count} episodes)")
        print(f"Instruction: {instruction[:50]}..." if len(instruction) > 50 else f"Instruction: {instruction}")

        # Compute the embedding only once for this unique instruction
        embedding = txt_encoder.compute_instruction_embedding(instruction)

        # Store in our dictionary for later use
        unique_instructions[instruction] = embedding

    # save unique_instructions
    dict_output_path = os.path.join(dataset_dir, "all_instruction_embeddings.pt")
    print(f"\nSaving dictionary of instructions and embeddings to {dict_output_path}...")
    torch.save(unique_instructions, dict_output_path)
    print(f"Successfully saved dictionary with {len(unique_instructions['all_instructions'])} unique instructions.")

    # Now save the pre-computed embeddings to each episode folder
    print("\nSaving embeddings to episode folders...")
    for idx, item in enumerate(instruction_episode_map):
        instruction = item['instruction']
        episode_path = item['episode_path']
        episode_name = item['episode_name']

        # Get the already-computed embedding
        embedding = unique_instructions[instruction]

        # Save the embedding to the episode folder in PyTorch format
        embedding_file = os.path.join(episode_path, "instruction_embedding.pt")
        torch.save(embedding, embedding_file)

        print(f"Saved embedding to {episode_name}/instruction_embedding.pt")

    print("\nAll embeddings computed and saved successfully!")

    # Return the instructions list in case it's needed
    return all_instructions


if __name__ == "__main__":
    # Define directory
    current_path = os.getcwd()
    dataset_directory = os.path.join(current_path, "data/datasets/water_cup_new")


    # Process all episodes
    instructions = process_episode_instructions(dataset_directory)

    print(f"\nProcessed {len(instructions)} instructions")
    print("Process completed!")


    #
    print("=" * 40)
    print(f"Dataset Directory: {dataset_directory}")
    print("=" * 40)
    embedding_path = os.path.join(dataset_directory,'all_instruction_embeddings.pt')
    embeddings = torch.load(embedding_path)
    print(embeddings.keys())