# backend/models/fusion_model.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MultimodalFusionModel(nn.Module):
    """
    Transformer-based multimodal fusion model.
    Combines image, text, and history embeddings using cross-modal self-attention.
    
    Input Shape: 
        - Three tensors of shape (batch_size, 768) representing the modalities.
    Output Shape: 
        - fused_features: (batch_size, 512) ready for the final classifier.
    
    Intended Dataset: Fuses outputs from ViT (Images), ClinicalBERT (Text), and History MLP (Tabular).
    """
    def __init__(self, modality_dim: int = 768, num_modalities: int = 3, output_dim: int = 512):
        super(MultimodalFusionModel, self).__init__()
        self.modality_dim = modality_dim
        self.num_modalities = num_modalities
        
        logger.info(f"Initializing Multimodal Fusion Transformer (dim={modality_dim}, modalities={num_modalities})")
        
        # Project modalities to ensure consistent dimensionality (though they already are)
        self.modality_projection = nn.Linear(modality_dim, modality_dim)
        
        # Transformer encoder for cross-modal attention
        # Treats the 3 modalities as a sequence of length 3
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=modality_dim, 
            nhead=8, 
            dim_feedforward=modality_dim * 2, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Final MLP to compress the attended sequence into a single fused vector
        self.fusion_mlp = nn.Sequential(
            nn.Linear(modality_dim * num_modalities, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor, history_emb: torch.Tensor):
        """
        Forward pass.
        Args:
            image_emb: (batch_size, 768)
            text_emb: (batch_size, 768)
            history_emb: (batch_size, 768)
        Returns:
            fused_features: (batch_size, 512)
        """
        # Stack modalities to form a sequence: (batch_size, 3, 768)
        x = torch.stack([image_emb, text_emb, history_emb], dim=1)
        
        # Project and apply self-attention across the 3 modalities
        x = self.modality_projection(x)
        x = self.transformer_encoder(x) # Shape remains (batch_size, 3, 768)
        
        # Flatten the sequence dimension: (batch_size, 3 * 768)
        x = x.view(x.size(0), -1)
        
        # Pass through the fusion MLP
        fused_features = self.fusion_mlp(x)
        
        return fused_features