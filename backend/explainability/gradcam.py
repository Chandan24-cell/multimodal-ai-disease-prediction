# backend/explainability/gradcam.py
import torch
import numpy as np
import cv2
import base64
import io
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import logging

from models.vit_model import MedicalViTModel

logger = logging.getLogger(__name__)

class LogitsModel(torch.nn.Module):
    def __init__(self, model: MedicalViTModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        return self.model(pixel_values).logits


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    patch_tokens = tensor[:, 1:, :]
    patch_count = patch_tokens.shape[1]
    grid_size = int(patch_count ** 0.5)
    return patch_tokens.reshape(tensor.shape[0], grid_size, grid_size, tensor.shape[2]).permute(0, 3, 1, 2)


class ViTGradCAM:
    """
    Generates Grad-CAM heatmaps for Vision Transformer models.
    """
    def __init__(self, model: MedicalViTModel, device: torch.device):
        self.device = device
        # Target the last layer of the ViT encoder
        # HuggingFace ViT structure: model.vit.vit.encoder.layer[-1]
        target_layer = model.vit.vit.encoder.layer[-1].output
        self.cam = GradCAM(
            model=LogitsModel(model),
            target_layers=[target_layer],
            reshape_transform=vit_reshape_transform
        )

    def generate_heatmap(self, image_tensor: torch.Tensor, target_class_idx: int, original_image_bytes: bytes) -> str:
        """
        Generates a Grad-CAM heatmap and overlays it on the original image.
        Returns a Base64 encoded PNG string for JSON serialization.
        """
        # 1. Generate the CAM
        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam = self.cam(input_tensor=image_tensor, targets=targets)[0, :]
        
        # 2. Resize CAM to match original image dimensions
        # Note: ViT expects 224x224, so we resize the original image to 224x224 for the overlay
        original_image = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
        original_image = original_image.resize((224, 224))
        rgb_img = np.float32(original_image) / 255.0
        
        # 3. Overlay CAM on image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # 4. Convert to Base64 string
        pil_img = Image.fromarray(visualization)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{img_base64}"