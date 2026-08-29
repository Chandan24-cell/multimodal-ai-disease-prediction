# backend/inference/text_inference.py
import torch
from transformers import AutoTokenizer
from typing import Dict, Tuple
import logging

from models.clinical_model import ClinicalTransformerModel

logger = logging.getLogger(__name__)

class TextInferencePipeline:
    """
    Singleton pipeline for clinical text (symptom) inference.
    Handles tokenization, model inference, and returns both predictions and embeddings.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextInferencePipeline, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing Text Inference Pipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # TODO: Replace with your fine-tuned model path, e.g., "./models/clinical/fine_tuned_symptoms"
        model_path = "emilyalsentzer/Bio_ClinicalBERT"
        self.num_labels = 14 # Must match the ViT num_labels for consistent disease taxonomy
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ClinicalTransformerModel(num_labels=self.num_labels, model_name=model_path)
        
        # TODO: Load fine-tuned weights here once you have them
        # self.model.load_state_dict(torch.load("path/to/clinical_weights.pth", map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        # TODO: UPDATE THESE CLASS NAMES TO MATCH YOUR SPECIFIC DATASET/TAXONOMY
        self.class_names = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
            "Consolidation", "Edema", "Emphysema", "Fibrosis", 
            "Pleural_Thickening", "Hernia"
        ]
        logger.info("Text Inference Pipeline initialized successfully.")

    def predict(self, symptom_text: str) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Run inference on symptom text.
        Returns:
            - predictions: Dictionary of class names and their probabilities.
            - embedding: The 768-dim tensor representing the text, for multimodal fusion.
        """
        # Tokenize
        inputs = self.tokenizer(
            symptom_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            embeddings = outputs["embeddings"]
            
            # Apply softmax for multi-class probability distribution
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        predictions = {cls: float(prob) for cls, prob in zip(self.class_names, probs)}
        
        # Return embedding on CPU for easier handling in the fusion pipeline later
        return predictions, embeddings.squeeze(0).cpu()

# Global instance to be imported by the API layer later
text_inference = TextInferencePipeline()