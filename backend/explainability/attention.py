# backend/explainability/attention.py
import torch
from typing import List, Dict
import logging

from models.clinical_model import ClinicalTransformerModel

logger = logging.getLogger(__name__)

class TextAttentionExplainer:
    """
    Extracts attention weights from the Clinical Transformer to highlight important words.
    """
    def __init__(self, model: ClinicalTransformerModel, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_token_attention(self, text: str, target_class_idx: int) -> List[Dict[str, float]]:
        """
        Returns a list of tokens and their average attention weight from the last layer.
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512,
            return_token_type_ids=False
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Forward pass with output_attentions=True
        with torch.no_grad():
            # We bypass the wrapper and call the base model directly to get attentions
            outputs = self.model.bert(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_attentions=True
            )
            
        # Get attention from the last layer: shape (batch_size, num_heads, seq_len, seq_len)
        last_layer_attentions = outputs.attentions[-1]
        
        # Average across all attention heads: shape (batch_size, seq_len, seq_len)
        avg_attentions = last_layer_attentions.mean(dim=1).squeeze(0).cpu().numpy()
        
        # We care about the attention *to* each token *from* the [CLS] token (index 0)
        # Or, more commonly for classification, the attention each token receives from the whole sequence.
        # Let's use the mean attention received by each token across the sequence.
        token_importance = avg_attentions.mean(axis=0) # shape: (seq_len,)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
        
        # Filter out special tokens and pad tokens, and normalize scores
        result = []
        max_score = max(token_importance) if max(token_importance) > 0 else 1.0
        
        for token, score in zip(tokens, token_importance):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            
            # Clean up subword tokens (e.g., "##pain" -> "pain")
            clean_token = token.replace("##", "")
            normalized_score = float(score / max_score)
            
            result.append({
                "token": clean_token,
                "attention_weight": round(normalized_score, 4)
            })
            
        return result