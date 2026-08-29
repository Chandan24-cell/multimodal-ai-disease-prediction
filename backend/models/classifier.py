# backend/models/classifier.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class DiseaseClassifier(nn.Module):
    """
    Final multi-class/multi-label disease classification head.
    
    Input Shape: (batch_size, 512) - The fused multimodal features.
    Output Shape: (batch_size, num_labels) - Raw logits for each disease.
    
    Intended Dataset: Trained on the fused outputs of the MultimodalFusionModel.
    """
    def __init__(self, input_dim: int = 512, num_labels: int = 14):
        super(DiseaseClassifier, self).__init__()
        self.num_labels = num_labels
        logger.info(f"Initializing Disease Classifier for {num_labels} classes.")
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(input_dim, num_labels)
        )

    def forward(self, fused_features: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass.
        Args:
            fused_features: (batch_size, 512)
            labels: Optional (batch_size, num_labels) for loss calculation.
        Returns:
            Dictionary with 'logits' and optional 'loss'.
        """
        logits = self.classifier(fused_features)
        
        loss = None
        if labels is not None:
            # Using BCEWithLogitsLoss for multi-label (e.g., ChestX-ray). 
            # Change to CrossEntropyLoss if strictly single-label.
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            
        return {
            "loss": loss,
            "logits": logits
        }