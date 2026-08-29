# backend/inference/history_inference.py
import torch
import numpy as np
from typing import Dict, Any, Tuple
import logging

from models.history_model import MedicalHistoryModel

logger = logging.getLogger(__name__)

class HistoryInferencePipeline:
    """
    Singleton pipeline for structured medical history inference.
    Handles feature normalization, tensor conversion, and embedding extraction.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HistoryInferencePipeline, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing History Inference Pipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TODO: Set this to the exact number of features in your preprocessed tabular dataset
        # Example: Age(1) + Gender(2) + HR, BP_sys, BP_dia, Temp, SpO2(5) + 14 binary conditions = 22
        self.input_dim = 22 
        self.num_labels = 14 # Must match ViT and Clinical models
        
        self.model = MedicalHistoryModel(input_dim=self.input_dim, num_labels=self.num_labels)
        
        # TODO: Load fine-tuned weights here
        # self.model.load_state_dict(torch.load("../models/history/history_weights.pth", map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        # TODO: Load your preprocessing statistics (e.g., from a saved sklearn StandardScaler)
        # self.age_mean, self.age_std = 55.0, 15.0
        # self.vitals_mean = [80, 120, 80, 98.6, 98.0]
        # self.vitals_std = [15, 15, 10, 1.0, 2.0]
        
        logger.info("History Inference Pipeline initialized successfully.")

    def preprocess_features(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert raw dictionary data into a normalized 1D numpy array.
        TODO: Replace this mock logic with your actual sklearn Pipeline/StandardScaler transformations.
        """
        # 1. Age (normalized)
        age = (patient_data.get("age", 50) - 50.0) / 15.0 
        
        # 2. Gender (one-hot: [is_male, is_female])
        gender = patient_data.get("gender", "other").lower()
        is_male = 1.0 if gender == "male" else 0.0
        is_female = 1.0 if gender == "female" else 0.0
        
        # 3. Vitals (normalized mock values: HR, BP_sys, BP_dia, Temp, SpO2)
        vitals = patient_data.get("vitals", {})
        hr = (vitals.get("heart_rate", 80) - 80.0) / 15.0
        bp_sys = (vitals.get("systolic_bp", 120) - 120.0) / 15.0
        bp_dia = (vitals.get("diastolic_bp", 80) - 80.0) / 10.0
        temp = (vitals.get("temperature", 98.6) - 98.6) / 1.0
        spo2 = (vitals.get("spo2", 98.0) - 98.0) / 2.0
        
        # 4. Prior Conditions (multi-hot binary vector of length 14, matching disease taxonomy)
        conditions = patient_data.get("prior_conditions", [])
        # TODO: Map string conditions to specific indices based on your dataset
        condition_vector = [1.0 if cond in conditions else 0.0 for cond in ["hypertension", "diabetes", "copd", "heart_failure", "asthma", "obesity", "smoking", "ckd", "liver_disease", "cancer", "dementia", "stroke", "thyroid", "none"]]
        # Pad or truncate to ensure exact length if needed, here we assume 14
        
        # Concatenate all features
        features = [age, is_male, is_female, hr, bp_sys, bp_dia, temp, spo2] + condition_vector
        
        # Ensure exact input_dim
        if len(features) != self.input_dim:
            logger.warning(f"Feature length {len(features)} does not match expected input_dim {self.input_dim}. Padding/Truncating.")
            features = (features + [0.0] * self.input_dim)[:self.input_dim]
            
        return np.array(features, dtype=np.float32)

    def get_embedding(self, patient_data: Dict[str, Any]) -> torch.Tensor:
        """
        Run inference on structured patient data.
        Returns the 768-dim embedding tensor for multimodal fusion.
        """
        features_np = self.preprocess_features(patient_data)
        features_tensor = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            embeddings = outputs["embeddings"]
            
        return embeddings.squeeze(0).cpu()

# Global instance
history_inference = HistoryInferencePipeline()
