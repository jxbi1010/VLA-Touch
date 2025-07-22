import os
import torch
import yaml
from models.multimodal_encoder.t5_encoder import T5Embedder
from typing import List, Dict

GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
SAVE_DIR = "data/datasets/ball_pick/"
SAVE_NAME = "instruction_embeddings.pt"
OFFLOAD_DIR = None

# Define your instructions as a list
INSTRUCTIONS = [
    "Pick up the stress ball.",
    "Pick up the baseball."
]


class LanguageEncoder:
    def __init__(self, model_path: str, config_path: str, device: torch.device, offload_dir: str = None):
        with open(config_path, "r") as fp:
            self.config = yaml.safe_load(fp)

        self.text_embedder = T5Embedder(
            from_pretrained=model_path,
            model_max_length=self.config["dataset"]["tokenizer_max_length"],
            device=device,
            use_offload_folder=offload_dir
        )
        self.tokenizer = self.text_embedder.tokenizer
        self.text_encoder = self.text_embedder.model
        self.device = device

    def encode_single(self, instruction: str) -> torch.Tensor:
        """Encode a single instruction."""
        tokens = self.tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(self.device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_encoder(tokens).last_hidden_state.detach().cpu()
        return pred

    def encode_instructions(self, instructions: List[str], save_path: str) -> Dict[str, torch.Tensor]:
        """
        Encode multiple instructions and save them to a single file.
        The dictionary uses instructions as keys and embeddings as values.
        """
        embeddings_dict = {}

        for instruction in instructions:
            embedding = self.encode_single(instruction)
            embeddings_dict[instruction] = embedding
            print(f'Encoded: "{instruction}" -> shape {embedding.shape}')

        # Save all embeddings in one file
        torch.save(embeddings_dict, save_path)
        print(f'\nSaved {len(embeddings_dict)} embeddings to {save_path}')

        return embeddings_dict


def load_embeddings(save_path: str) -> Dict[str, torch.Tensor]:
    """Load all embeddings from the file."""
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"No embeddings found at: {save_path}")

    embeddings_dict = torch.load(save_path)
    print(f"Loaded {len(embeddings_dict)} embeddings from {save_path}")
    return embeddings_dict


def get_embedding(instruction: str, embeddings_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Get embedding for a specific instruction."""
    if instruction not in embeddings_dict:
        raise KeyError(f"Instruction not found in embeddings dictionary: {instruction}")
    return embeddings_dict[instruction]


def main():
    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)

    # Initialize encoder and encode instructions
    device = torch.device(f"cuda:{GPU}")
    encoder = LanguageEncoder(MODEL_PATH, CONFIG_PATH, device, OFFLOAD_DIR)
    embeddings_dict = encoder.encode_instructions(INSTRUCTIONS, save_path)
    torch.save(embeddings_dict, "outs/instruction_embeddings.pt")

    # Example: Load embeddings and retrieve specific one
    loaded_embeddings = load_embeddings(save_path)

    # Example: Get embedding for specific instruction
    test_instruction = INSTRUCTIONS[0]
    embedding = get_embedding(test_instruction, loaded_embeddings)
    print(f"\nRetrieved embedding for '{test_instruction}'")
    print(f"Embedding shape: {embedding.shape}")

    # Print all available instructions
    print("\nAvailable instructions:")
    for idx, instruction in enumerate(loaded_embeddings.keys(), 1):
        print(f"{idx}. {instruction}")


if __name__ == "__main__":
    main()