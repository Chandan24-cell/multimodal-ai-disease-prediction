# backend/models/vit_model.py
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
import logging

logger = logging.getLogger(__name__)

class MedicalViTModel(nn.Module):
    """
    Vision Transformer model for medical image (MRI/X-Ray) classification.
    
    Input Shape: (batch_size, num_channels, height, width) -> e.g., (B, 3, 224, 224)
    Output Shape: HuggingFace SequenceClassifierOutput (contains logits and loss)
    
    Intended Dataset: NIH ChestX-ray14, CheXpert, or custom MRI datasets.
    """
    def __init__(self, num_labels: int = 14, model_name: str = "google/vit-base-patch16-224"):
        super(MedicalViTModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        
        logger.info(f"Loading ViT backbone from: {model_name}")
        config = ViTConfig.from_pretrained(model_name, num_labels=num_labels)
        
        # Load weights (works for both HuggingFace IDs and local directories)
        self.vit = ViTForImageClassification.from_pretrained(model_name, config=config)
        
        # Optional: Freeze base layers for initial fine-tuning to prevent overfitting
        # self._freeze_base_layers()

    def _freeze_base_layers(self):
        """Freeze the ViT backbone, only train the classification head."""
        for name, param in self.vit.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        logger.info("Frozen ViT base layers. Only classifier head is trainable.")

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass.
        Args:
            pixel_values: Tensor of shape (B, 3, 224, 224)
            labels: Optional tensor of shape (B, num_labels) for loss calculation
        Returns:
            HuggingFace SequenceClassifierOutput containing logits and loss (if labels provided)
        """
        outputs = self.vit(pixel_values=pixel_values, labels=labels)
        return outputs

    def get_last_hidden_state(self, pixel_values: torch.Tensor):
        """
        Extract features for Explainability (Grad-CAM).
        Returns the output of the last transformer block.
        """
        with torch.no_grad():
            outputs = self.vit.vit(pixel_values=pixel_values, output_hidden_states=True)
            # Shape: (B, seq_len, hidden_dim) where seq_len = 1 (CLS) + 196 (patches)
            return outputs.last_hidden_state
