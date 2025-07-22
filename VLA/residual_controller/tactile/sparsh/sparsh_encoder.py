# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from sparsh.tactile_ssl.model.vision_transformer import VisionTransformer
from functools import partial
from sparsh.tactile_ssl.model.layers import MemEffAttention
from sparsh.tactile_ssl.model.layers import NestedTensorBlock as Block

class EncoderModule(nn.Module):
    def __init__(
            self,
            encoder: VisionTransformer,
    ):
        super().__init__()

        # Encoder
        self.encoder: VisionTransformer = encoder
        self.in_chans = self.encoder.patch_embed.in_chans
        if isinstance(self.encoder.patch_embed.patch_size, tuple):
            self.patch_size = self.encoder.patch_embed.patch_size[0]
        else:
            self.patch_size = self.encoder.patch_embed.patch_size
        self.num_patches = self.encoder.patch_embed.num_patches
        self.embed_dim = self.encoder.embed_dim
        self.img_size = self.encoder.img_size
        self.in_chans = self.encoder.in_chans


    def forward(self, x: torch.Tensor):

        return  self.encoder(x)

def load_pretrained_encoder(model_size='base', device='cuda'):
    """
    Load a pretrained encoder from an MAE checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        encoder_module: The loaded encoder module
    """
    # Load the checkpoint
    checkpoint_path = f"./residual_controller/tactile/sparsh/sparsh_ckpt/sparsh-mae-{model_size}/mae_vit{model_size}.ckpt"

    checkpoint = torch.load(checkpoint_path, map_location=device ,weights_only=False)

    # Filter for only encoder parameters
    encoder_state_dict = {}
    if 'model' in checkpoint:
        # Get weights from the 'model' key
        model_state_dict = checkpoint['model']
    else:
        model_state_dict = checkpoint

    # Extract encoder parameters
    for key, value in model_state_dict.items():
        # Check if key refers to encoder parameters
        if key.startswith('encoder.') or key.startswith('module.encoder.'):
            # Remove the 'encoder.' or 'module.encoder.' prefix
            if key.startswith('module.encoder.'):
                new_key = key.replace('module.encoder.', '')
            else:
                new_key = key.replace('encoder.', '')
            encoder_state_dict[new_key] = value
        # Sometimes the encoder might not have a prefix in the state_dict
        elif not key.startswith('decoder.') and not key.startswith('module.decoder.'):
            # This is likely an encoder parameter
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            else:
                new_key = key
            encoder_state_dict[new_key] = value

    if model_size=='small':
        embed_dim = 384
        num_heads = 6
    elif model_size=='base':
        embed_dim = 768
        num_heads = 12
    else:
        print("model_size not indicated")

    encoder = VisionTransformer(
        img_size=(320,240),  # Adjust based on your model
        patch_size=16,  # Adjust based on your model
        in_chans=6,    # Adjust based on your model
        embed_dim=embed_dim, # Adjust based on your model
        depth=12,      # Adjust based on your model
        num_heads=num_heads,  # Adjust based on your model
        mlp_ratio=4.0,
        qkv_bias=True,
        num_register_tokens=0,
        pos_embed_fn = 'sinusoidal',
        block_fn=partial(Block, attn_class=MemEffAttention),
    )

    # Load the encoder state dict
    encoder.load_state_dict(encoder_state_dict)

    # Wrap the encoder in our simplified module
    encoder_module = EncoderModule(encoder=encoder)
    encoder_module.to(device)
    encoder_module.eval()  # Set to evaluation mode

    print(f"Loaded encoder with {sum(p.numel() for p in encoder_module.parameters())} parameters")

    return encoder_module


# Example usage
if __name__ == "__main__":
    import time
    # Load the encoder
    encoder = load_pretrained_encoder(model_size="base")

    # Example forward pass
    with torch.no_grad():
        # Create a dummy input tensor
        batch_size = 1
        channels = 6  # Adjust based on your model's input channels
        height = 320  # Adjust based on your model's input size
        width = 240  # Adjust based on your model's input size

        dummy_input = torch.randn(batch_size, channels, height, width).cuda()

        # Forward pass
        start = time.time()
        embeddings = encoder.forward(dummy_input)
        end = time.time()

        print(f"Embedding shape: {embeddings.shape}, inference time:{end-start}")