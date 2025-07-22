import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from transformers import Dinov2Model, Dinov2Config, AutoImageProcessor
# from models.multimodal_encoder.siglip_encoder import SiglipVisionTower

class DINOv2Encoder:
    """
    DINOv2 vision encoder that processes images for the controller.
    """

    def __init__(self, model_name="facebook/dinov2-small", device="cuda"):
        """
        Initialize the DINOv2 encoder.

        Args:
            model_name: Name or path of the DINOv2 model to use
                Options: 'facebook/dinov2-small', 'facebook/dinov2-base',
                         'facebook/dinov2-large', 'facebook/dinov2-giant'
            device: Device to run the model on
        """
        self.device = device

        # Initialize the DINOv2 model
        self.model = Dinov2Model.from_pretrained(model_name)
        # Use the proper image processor from transformers

        # Set image processing parameters
        if 'small' in model_name:
            self.patch_size = 14
            self.hidden_size = 384
        elif 'base' in model_name:
            self.patch_size = 14
            self.hidden_size = 768
        elif 'large' in model_name:
            self.patch_size = 14
            self.hidden_size = 1024
        elif 'giant' in model_name:
            self.patch_size = 14
            self.hidden_size = 1536
        else:
            # Default to small model parameters
            self.patch_size = 14
            self.hidden_size = 384

        # Set model to evaluation mode and move to specified device
        self.model.eval()
        self.model.to(device)

        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        """
        Process images into embeddings.
        Args:
            images: Input images as tensor or numpy array
                Expected format after processing: [batch_size, 3, height, width]
        Returns:
            Embeddings: Tensor of shape [batch_size, hidden_size]
        """
        # Handle numpy arrays
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float() / 255.0

        # Format conversion for different input shapes
        if len(images.shape) == 5:  # If images are in [B, T, H, W, C] format
            B, T, H, W, C = images.shape
            images = images.reshape(B * T, H, W, C)  # Flatten time dimension
            images = images.permute(0, 3, 1, 2)  # Change to (B*T, C, H, W)
        elif len(images.shape) == 4 and images.shape[-1] == 3:  # [B, H, W, C]
            images = images.permute(0, 3, 1, 2)  # Change to (B, C, H, W)

        # Ensure proper normalization
        if images.max() > 1.0:
            images = images / 255.0

        images = self._normalize_images(images)

        # Create inputs dict in the format expected by the model
        inputs = {'pixel_values': images}

        # Extract features with DINOv2
        with torch.no_grad():
            outputs = self.model(**inputs)

            # The CLS token embedding is the first token in the last_hidden_state
            cls_embeddings = outputs.pooler_output

            return cls_embeddings

    def _normalize_images(self, images):
        """
        Apply normalization to images if not already normalized
        """
        # Check if images appear to be already normalized
        if images.mean() < 0.5:
            return images

        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std


# class SiglipEncoder:
#     """
#     SigLIP vision encoder that processes images for the controller.
#     """
#
#     def __init__(self, model_path="google/siglip-so400m-patch14-384", device="cuda"):
#         self.device = device
#         self.vision_encoder = SiglipVisionTower(vision_tower=model_path, args=None)
#         self.image_processor = transforms.Compose([
#             transforms.Resize((384, 384)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
#         self.hidden_size = self.vision_encoder.hidden_size
#         self.vision_encoder.eval()
#         self.vision_encoder.to(device)
#
#     def forward(self, images):
#         """
#         Process images into embeddings.
#         """
#         if isinstance(images, np.ndarray):
#             images = torch.from_numpy(images).float() / 255.0  # Normalize pixel values
#         if images.dtype != torch.float32:
#             images = images.float() / 255.0  # Ensure float conversion
#         if images.max() > 1.0:
#             images = images / 255.0  # Normalize pixel values
#
#         # Ensure image format is correct (batch, C, H, W)
#         if len(images.shape) == 5:  # If images are in [B, T, H, W, C] format
#             B, T, H, W, C = images.shape
#             images = images.reshape(B * T, H, W, C)  # Flatten time dimension
#             images = images.permute(0, 3, 1, 2)  # Change to (B*T, C, H, W)
#         elif len(images.shape) == 4 and images.shape[-1] == 3:  # [B, H, W, C]
#             images = images.permute(0, 3, 1, 2)
#
#         images = images.to(self.device)
#         with torch.no_grad():
#             patch_embeddings = self.vision_encoder(images)
#
#         image_embeds = torch.mean(patch_embeddings, dim=1)
#         image_embeds = image_embeds.reshape(-1, self.hidden_size)
#         return image_embeds