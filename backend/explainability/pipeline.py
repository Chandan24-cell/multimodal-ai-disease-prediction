# backend/explainability/pipeline.py
import torch
import logging
import numpy as np

from inference.image_inference import image_inference
from inference.text_inference import text_inference
from inference.history_inference import history_inference

from explainability.gradcam import ViTGradCAM
from explainability.attention import TextAttentionExplainer
from explainability.shap_analysis import HistorySHAPExplainer

logger = logging.getLogger(__name__)

class ExplainabilityPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize explainers
        self.gradcam = ViTGradCAM(image_inference.model, self.device)
        self.text_explainer = TextAttentionExplainer(text_inference.model, text_inference.tokenizer, self.device)
        
        # TODO: Replace with actual background data from your training set (e.g., 100 random patient histories)
        # For now, we use a mock background of 10 zero-mean, unit-variance samples
        mock_background = np.random.randn(10, history_inference.input_dim).astype(np.float32)
        mock_feature_names = ["age", "is_male", "is_female", "hr", "bp_sys", "bp_dia", "temp", "spo2"] + \
                             [f"cond_{i}" for i in range(14)]
                             
        self.shap_explainer = HistorySHAPExplainer(
            history_inference.model, 
            background_data=mock_background, 
            feature_names=mock_feature_names, 
            device=self.device
        )

    def generate_full_explanation(
        self, 
        image_bytes: bytes, 
        symptom_text: str, 
        patient_data: dict, 
        target_class_idx: int
    ) -> dict:
        """
        Generates all explainability artifacts for a given prediction.
        """
        logger.info(f"Generating explanations for target class index: {target_class_idx}")
        
        # 1. Image Grad-CAM
        img_tensor = image_inference.preprocess_image(image_bytes)
        heatmap_base64 = self.gradcam.generate_heatmap(img_tensor, target_class_idx, image_bytes)
        
        # 2. Text Attention
        token_attentions = self.text_explainer.get_token_attention(symptom_text, target_class_idx)
        
        # 3. Tabular SHAP
        features_np = history_inference.preprocess_features(patient_data)
        shap_values = self.shap_explainer.explain_instance(features_np, target_class_idx)
        
        return {
            "image_heatmap": heatmap_base64,
            "text_attention": token_attentions,
            "tabular_shap": shap_values
        }

# Global instance
explainability_pipeline = ExplainabilityPipeline()