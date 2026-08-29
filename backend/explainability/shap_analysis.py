# backend/explainability/shap_analysis.py
import torch
import numpy as np
import shap
import logging
from typing import List, Dict

from backend.models.history_model import MedicalHistoryModel

logger = logging.getLogger(__name__)

class HistorySHAPExplainer:
    """
    Generates SHAP values for the Medical History (tabular) model.
    """
    def __init__(self, model: MedicalHistoryModel, background_data: np.ndarray, feature_names: List[str], device: torch.device):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        
        # SHAP DeepExplainer requires the model to be in eval mode and on the same device
        self.model.eval()
        
        # Convert background data to torch tensor for DeepExplainer
        background_tensor = torch.tensor(background_data, dtype=torch.float32).to(device)
        
        # SHAP expects the model to return a tensor rather than the model's output dictionary.
        class LogitsModel(torch.nn.Module):
            def __init__(self, wrapped_model):
                super().__init__()
                self.wrapped_model = wrapped_model

            def forward(self, x):
                return self.wrapped_model(x)["logits"]

        self.explainer_model = LogitsModel(self.model).to(device)
        self.explainer = shap.DeepExplainer(self.explainer_model, background_tensor)
        logger.info("SHAP DeepExplainer initialized successfully.")

    def explain_instance(self, instance_features: np.ndarray, target_class_idx: int) -> Dict[str, float]:
        """
        Calculates SHAP values for a single patient instance.
        Returns a dictionary mapping feature names to their SHAP values for the target class.
        """
        # Reshape to (1, num_features)
        instance_tensor = torch.tensor(instance_features.reshape(1, -1), dtype=torch.float32)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance_tensor, check_additivity=False)
        
        # shap_values is a list of arrays (one per class). We want the target class.
        # Note: Depending on SHAP version, it might be a single array for multi-label. 
        # We'll handle the multi-label case by taking the specific class index.
        if isinstance(shap_values, list):
            target_shap = shap_values[target_class_idx][0]
        else:
            # If it returns a single array of shape (1, num_features, num_classes)
            target_shap = shap_values[0, :, target_class_idx]
            
        # Map to feature names
        explanation = {}
        for i, feature_name in enumerate(self.feature_names):
            explanation[feature_name] = round(float(target_shap[i]), 4)
            
        return explanation
