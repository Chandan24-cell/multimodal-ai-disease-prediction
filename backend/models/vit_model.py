import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel
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
        
        # The local checkpoint was exported by Hugging Face
        # ``ViTForImageClassification`` and contains both ``vit.*`` and
        # ``classifier.*`` tensors. Loading only the backbone makes disease
        # scores come from a newly random classifier head.
        # Do not provide a random-initialized fallback for clinical inference.
        try:
            logger.info(f"Loading ViT from checkpoint: {checkpoint_path}")
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"ViT checkpoint does not exist: {checkpoint_path}")

            config_dir = checkpoint_path.parent
            config = ViTConfig.from_pretrained(config_dir)
            if config.num_labels != num_labels:
                raise ValueError(
                    f"Checkpoint config declares {config.num_labels} labels; expected {num_labels}."
                )

            # ViTForImageClassification has no pooler tensors. Mirroring that
            # layout lets strict loading verify every checkpoint tensor.
            self.vit = ViTModel(config, add_pooling_layer=False)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path, device="cpu")
            load_result = self.load_state_dict(state_dict, strict=True)
            if load_result.missing_keys or load_result.unexpected_keys:
                raise RuntimeError(
                    "Checkpoint key mismatch: "
                    f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
                )
            self.eval()
            logger.info(
                "Loaded complete ViT checkpoint strictly: %d tensors, classifier=%s",
                len(state_dict),
                tuple(self.classifier.weight.shape),
            )
        except Exception as e:
            logger.exception("Failed to load the complete ViT checkpoint")
            raise RuntimeError(
                f"Cannot load trained ViT checkpoint at {checkpoint_path}: {e}"
            ) from e

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
