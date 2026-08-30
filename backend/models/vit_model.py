import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MedicalViTModel(nn.Module):
    """
    Medical Vision Transformer for multi-label disease classification.
    
    Architecture:
    - ViT backbone: Outputs 768-dimensional embeddings (CLS token)
    - 14-label classification head (multi-label sigmoid)
    - Supports Grad-CAM explainability via self.vit attribute
    """

    def __init__(self, num_labels=14, model_name=None, checkpoint_path=None):
        super().__init__()

        self.num_labels = num_labels
        self.embedding_dim = 768
        
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).resolve().parent / "vit" / "medical_finetuned" / "model.safetensors"
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        
        # Try to load pretrained ViT with safetensors checkpoint
        try:
            logger.info(f"Loading ViT from checkpoint: {checkpoint_path}")
            # ViT-base from transformers (hidden_size=768)
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_labels=num_labels,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                problem_type="multi_label_classification",
            )
            self.vit = ViTModel(config)
            
            # Load safetensors checkpoint if it exists
            if Path(checkpoint_path).exists():
                logger.info(f"Loading weights from {checkpoint_path}")
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
                # Filter to only vit keys if needed
                vit_state_dict = {
                    k.replace("vit.", ""): v for k, v in state_dict.items() 
                    if k.startswith("vit.")
                } or state_dict
                self.vit.load_state_dict(vit_state_dict, strict=False)
                logger.info("Successfully loaded checkpoint weights")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}. Using random initialization.")
        except Exception as e:
            logger.error(f"Error loading ViT checkpoint: {e}. Using default ViT initialization.")
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.vit = ViTModel(config)

        # Classification head for multi-label
        self.classifier = nn.Linear(self.embedding_dim, num_labels)

    def forward(self, pixel_values, labels=None):
        """
        Forward pass.
        
        Args:
            pixel_values: (batch_size, 3, 224, 224)
            labels: Optional, for training with BCE loss
            
        Returns:
            Object with logits attribute
        """
        # Get ViT output (last_hidden_state includes CLS token at position 0)
        vit_output = self.vit(pixel_values, return_dict=True)
        
        # CLS token embedding (batch_size, 768)
        pooled_output = vit_output.last_hidden_state[:, 0, :]
        
        # Classification logits (batch_size, num_labels)
        logits = self.classifier(pooled_output)

        return type(
            "Output",
            (),
            {"logits": logits}
        )()

    def get_last_hidden_state(self, pixel_values):
        """
        Returns the full last hidden state for explainability.
        
        Args:
            pixel_values: (batch_size, 3, 224, 224)
            
        Returns:
            (batch_size, num_patches+1, 768) - includes CLS token and patch embeddings
        """
        with torch.no_grad():
            vit_output = self.vit(pixel_values, return_dict=True)
        
        return vit_output.last_hidden_state
