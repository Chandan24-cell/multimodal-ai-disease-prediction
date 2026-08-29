# backend/models/history_model.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MedicalHistoryModel(nn.Module):
    """
    Neural network for structured/tabular medical history data.
    
    Input Shape: 
        - x: (batch_size, input_dim) 
        - Example input_dim: 1 (age) + 2 (gender one-hot) + 4 (vitals) + 14 (prior conditions multi-hot) = 21
    Output Shape: 
        - embeddings: (batch_size, 768) 
        - logits: (batch_size, num_labels) for auxiliary disease prediction
    
    Intended Dataset: MIMIC-III structured tables, EHR datasets, or custom tabular patient history CSVs.
    """
    def __init__(self, input_dim: int, num_labels: int = 14, hidden_dim: int = 768):
        super(MedicalHistoryModel, self).__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        
        logger.info(f"Initializing Medical History MLP with input_dim={input_dim}, hidden_dim={hidden_dim}")
        
        # Feature embedding block
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, hidden_dim), # Outputs 768-dim vector for multimodal fusion
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Classification head (optional, but useful for multi-task learning during training)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, input_dim) containing normalized tabular features.
            labels: Optional tensor of shape (batch_size, num_labels) for loss calculation.
        Returns:
            Dictionary containing 'embeddings' (for fusion) and 'logits' (for classification).
        """
        embeddings = self.feature_extractor(x)
        logits = self.classifier(embeddings)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() # Or BCEWithLogitsLoss for multi-label
            loss = loss_fct(logits, labels)
            
        return {
            "loss": loss,
            "logits": logits,
            "embeddings": embeddings # Crucial for Phase 5 (Multimodal Fusion)
        }