# backend/inference/image_inference.py
import torch
from transformers import ViTImageProcessor
from PIL import Image
import io
from typing import Dict, Tuple
import logging

from models.vit_model import MedicalViTModel

logger = logging.getLogger(__name__)

class ImageInferencePipeline:
    """
    Singleton pipeline for medical image inference.
    Handles model loading, preprocessing, and prediction.
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

        # POINT TO YOUR ACTUALLY TRAINED MEDICAL MODEL
        # Use an absolute path or ensure the relative path is correct from the backend dir
        import os
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(base_dir, "models", "vit", "medical_finetuned")
        
        self.num_labels = 14 
        
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        # Load the custom MedicalViTModel with the local path
        self.model = MedicalViTModel(num_labels=self.num_labels, model_name=model_path)

        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Successfully loaded FINETUNED medical model from {model_path}")

        # TODO: UPDATE THESE CLASS NAMES TO MATCH YOUR SPECIFIC DATASET
        self.class_names = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
            "Consolidation", "Edema", "Emphysema", "Fibrosis", 
            "Pleural_Thickening", "Hernia"
        ]
        logger.info("Image Inference Pipeline initialized successfully.")

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Convert raw image bytes to preprocessed tensor."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def predict(self, image_bytes: bytes) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Run inference on a single image.
        Returns:
            - predictions: Dictionary of class names and probabilities.
            - embedding: The 768-dim tensor representing the image, for multimodal fusion.
        """
        pixel_values = self.preprocess_image(image_bytes)
        
        with torch.no_grad():
            # Get the last hidden state for the embedding (CLS token)
            last_hidden_state = self.model.get_last_hidden_state(pixel_values)
            image_embedding = last_hidden_state[:, 0, :].squeeze(0).cpu() # Shape: (768,)

            outputs = self.model(pixel_values)
            logits = outputs.logits
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        predictions = {cls: float(prob) for cls, prob in zip(self.class_names, probs)}
        return predictions, image_embedding

# Global instance to be imported by the API layer later
image_inference = ImageInferencePipeline()
