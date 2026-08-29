# backend/inference/pipeline.py
from typing import Dict, Any, Tuple
import logging

from inference.image_inference import image_inference
from inference.text_inference import text_inference
from inference.history_inference import history_inference
from inference.fusion_inference import fusion_inference

logger = logging.getLogger(__name__)

class MasterInferencePipeline:
    """
    Master orchestrator for the entire multimodal inference process.
    Takes raw inputs (image bytes, text, patient data) and returns final predictions.
    """
    def __init__(self):
        self.image_pipe = image_inference
        self.text_pipe = text_inference
        self.history_pipe = history_inference
        self.fusion_pipe = fusion_inference

    def run(
        self, 
        image_bytes: bytes, 
        symptom_text: str, 
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the full multimodal pipeline.
        
        Returns a dictionary containing:
            - individual predictions (image, text, history)
            - final fused prediction
            - confidence score (max probability)
            - predicted disease(s)
        """
        logger.info("Starting Master Inference Pipeline...")
        
        # 1. Modality-specific inferences
        logger.info("Processing Image Modality...")
        img_preds, img_emb = self.image_pipe.predict(image_bytes)
        
        logger.info("Processing Text Modality...")
        txt_preds, txt_emb = self.text_pipe.predict(symptom_text)
        
        logger.info("Processing History Modality...")
        hist_emb = self.history_pipe.get_embedding(patient_data)
        # Note: History model doesn't strictly need to return individual preds for the final API, 
        # but we can add it if needed. For now, we just use its embedding.
        hist_preds = {} # Placeholder if needed
        
        # 2. Multimodal Fusion & Final Prediction
        logger.info("Running Multimodal Fusion...")
        fused_preds, fused_features = self.fusion_pipe.predict(img_emb, txt_emb, hist_emb)
        
        # 3. Extract final diagnosis and confidence
        # For multi-label, we threshold at 0.5. For multi-class, we take the argmax.
        # Assuming multi-label here:
        predicted_diseases = [disease for disease, prob in fused_preds.items() if prob >= 0.5]
        if not predicted_diseases:
            # Fallback to highest probability if none cross the threshold
            top_disease = max(fused_preds, key=fused_preds.get)
            predicted_diseases = [top_disease]
            
        max_confidence = max(fused_preds.values())
        
        logger.info(f"Pipeline complete. Predicted: {predicted_diseases} (Confidence: {max_confidence:.2f})")
        
        return {
            "image_prediction": img_preds,
            "text_prediction": txt_preds,
            "history_prediction": hist_preds,
            "fused_prediction": fused_preds,
            "final_diseases": predicted_diseases,
            "confidence_score": max_confidence,
            "fused_features": fused_features # Passed to SHAP later
        }

# Global instance
master_pipeline = MasterInferencePipeline()