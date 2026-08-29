# backend/inference/image_inference.py
import torch
from torchvision import transforms
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

        self.num_labels = 14 
        
        self.processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.model = MedicalViTModel(num_labels=self.num_labels)

        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Successfully loaded medical image model")

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
        return self.processor(image).unsqueeze(0).to(self.device)

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
