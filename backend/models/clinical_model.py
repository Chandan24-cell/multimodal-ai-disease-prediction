# backend/models/clinical_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import logging

logger = logging.getLogger(__name__)

class ClinicalTransformerModel(nn.Module):
    """
    Clinical BERT-based model for symptom text analysis.
    
    Input Shape: 
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
    Output Shape: 
        - A dictionary containing:
            - 'logits': (batch_size, num_labels) for disease classification
            - 'embeddings': (batch_size, hidden_size) for multimodal fusion
    
    Intended Dataset: MIMIC-III, custom symptom-to-diagnosis datasets, or Kaggle medical text datasets.
    """
    def __init__(self, num_labels: int = 14, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super(ClinicalTransformerModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.hidden_size = 768 # Standard BERT hidden size
        
        logger.info(f"Loading Clinical Transformer backbone: {model_name}")
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        
        # Custom classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Optional: Freeze base layers for initial fine-tuning
        # self._freeze_base_layers()

    def _freeze_base_layers(self):
        """Freeze the BERT backbone, only train the classification head."""
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        logger.info("Frozen Clinical Transformer base layers. Only classifier head is trainable.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass.
        Returns both classification logits and pooled embeddings for fusion.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        # Pooled output is the [CLS] token representation, shape: (batch_size, hidden_size)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Assuming multi-class classification for symptoms. 
            # Change to BCEWithLogitsLoss if multi-label.
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return {
            "loss": loss,
            "logits": logits,
            "embeddings": pooled_output # Crucial for Phase 5 (Multimodal Fusion)
        }