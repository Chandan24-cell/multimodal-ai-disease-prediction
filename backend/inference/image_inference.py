# backend/inference/image_inference.py
import torch
from PIL import Image
import io
import numpy as np
from typing import Dict, Tuple
import logging
from pathlib import Path
from models.vit_model import MedicalViTModel

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "vit" / "medical_finetuned"

class ImageInferencePipeline:
    """
    Singleton pipeline for medical image inference.
    Handles ViT model loading, preprocessing, and prediction.
    Outputs 768-dimensional embeddings for multimodal fusion.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageInferencePipeline, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing Image Inference Pipeline...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.num_labels = 14 
        
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Initialize ViT model - strictly uses Vision Transformer
        # No MobileNet fallback; checkpoint loading is built into MedicalViTModel
        try:
            self.model = MedicalViTModel(
                num_labels=self.num_labels,
                checkpoint_path=str(MODEL_DIR / "model.safetensors"),
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Successfully loaded MedicalViTModel (768-dim embeddings, 14 labels)")
        except Exception as e:
            logger.error(f"Failed to initialize ViT model: {e}")
            raise RuntimeError(
                f"Cannot initialize ViT model. Ensure checkpoint exists at "
                f"{MODEL_DIR / 'model.safetensors'}. Error: {e}"
            )

        # 14 medical conditions for multi-label classification
        self.class_names = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
            "Consolidation", "Edema", "Emphysema", "Fibrosis", 
            "Pleural_Thickening", "Hernia"
        ]
        logger.info("Image Inference Pipeline initialized successfully (ViT-only).")

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Convert raw image bytes to preprocessed tensor."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        tensor = (tensor - self.image_mean) / self.image_std
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image_bytes: bytes) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Run inference on a single image using ViT backbone.
        
        Returns:
            - predictions: Dictionary of class names and probabilities.
            - embedding: The 768-dim tensor (CLS token) for multimodal fusion.
        """
        pixel_values = self.preprocess_image(image_bytes)
        
        with torch.inference_mode():
            # Get full hidden state for explainability
            last_hidden_state = self.model.get_last_hidden_state(pixel_values)
            # CLS token embedding: (batch, 768) -> squeeze to (768,)
            image_embedding = last_hidden_state[:, 0, :].squeeze(0).cpu()  # Shape: (768,)
            assert image_embedding.shape == torch.Size([768]), \
                f"Expected 768-dim embedding, got {image_embedding.shape}"

            # Forward pass for predictions
            outputs = self.model(pixel_values)
            logits = outputs.logits  # (batch, 14)
            if logits.shape != (1, self.num_labels):
                raise RuntimeError(
                    f"Unexpected ViT logits shape {tuple(logits.shape)}; "
                    f"expected (1, {self.num_labels})."
                )
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        predictions = {cls: float(prob) for cls, prob in zip(self.class_names, probs)}
        return predictions, image_embedding

# Global instance to be imported by the API layer
image_inference = ImageInferencePipeline()
