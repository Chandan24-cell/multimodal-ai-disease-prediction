# backend/inference/fusion_inference.py
import torch
from typing import Dict, Tuple
import logging

from backend.models.fusion_model import MultimodalFusionModel
from models.classifier import DiseaseClassifier

logger = logging.getLogger(__name__)

class FusionInferencePipeline:
    """
    Singleton pipeline for multimodal fusion and final disease prediction.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FusionInferencePipeline, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing Fusion Inference Pipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = 14 # Must match all other models
        
        self.fusion_model = MultimodalFusionModel(modality_dim=768, output_dim=512)
        self.classifier = DiseaseClassifier(input_dim=512, num_labels=self.num_labels)
        
        # TODO: Load fine-tuned weights for both models
        # self.fusion_model.load_state_dict(torch.load("../models/fusion/fusion_weights.pth", map_location=self.device))
        # self.classifier.load_state_dict(torch.load("../models/fusion/classifier_weights.pth", map_location=self.device))
        
        self.fusion_model.to(self.device).eval()
        self.classifier.to(self.device).eval()
        
        # TODO: UPDATE THESE CLASS NAMES TO MATCH YOUR SPECIFIC DATASET
        self.class_names = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
            "Consolidation", "Edema", "Emphysema", "Fibrosis", 
            "Pleural_Thickening", "Hernia"
        ]
        logger.info("Fusion Inference Pipeline initialized successfully.")

    def predict(self, image_emb: torch.Tensor, text_emb: torch.Tensor, history_emb: torch.Tensor) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Run fusion and classification.
        Returns:
            - predictions: Dictionary of class names and final fused probabilities.
            - fused_features: The 512-dim tensor (useful for SHAP explainability later).
        """
        # Move embeddings to device and add batch dimension if missing
        img = image_emb.unsqueeze(0).to(self.device) if image_emb.dim() == 1 else image_emb.to(self.device)
        txt = text_emb.unsqueeze(0).to(self.device) if text_emb.dim() == 1 else text_emb.to(self.device)
        hist = history_emb.unsqueeze(0).to(self.device) if history_emb.dim() == 1 else history_emb.to(self.device)
        
        with torch.no_grad():
            fused_features = self.fusion_model(img, txt, hist)
            outputs = self.classifier(fused_features)
            logits = outputs["logits"]
            
            # Sigmoid for multi-label probabilities
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        predictions = {cls: float(prob) for cls, prob in zip(self.class_names, probs)}
        return predictions, fused_features.squeeze(0).cpu()

# Global instance
fusion_inference = FusionInferencePipeline()
