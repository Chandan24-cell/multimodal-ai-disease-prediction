"""ClinicalBERT semantic alignment between symptoms and image findings."""

from __future__ import annotations

import logging
from typing import Any

import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class MultimodalAnalyzer:
    """Create ClinicalBERT embeddings and compare symptoms to disease labels."""

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    CONSISTENT_THRESHOLD = 60.0
    NEUTRAL_THRESHOLD = 35.0

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.tokenizer = None
        self.model = None

    def _load_model(self) -> None:
        if self.model is not None:
            return
        logger.info("Loading ClinicalBERT model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _embed(self, texts: list[str]) -> torch.Tensor:
        self._load_model()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(self.device) for key, value in encoded.items()
        }
        with torch.no_grad():
            output = self.model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1)
        mask = mask.expand(output.last_hidden_state.size()).float()
        pooled = (output.last_hidden_state * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.cpu()

    def analyze(
        self, symptoms: str, top_predictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return 0-100 alignment and weighted multimodal scores for predictions."""
        cleaned_symptoms = symptoms.strip()
        if not cleaned_symptoms:
            raise ValueError("Symptoms are required for multimodal analysis.")
        predictions = top_predictions[:5]
        if not predictions:
            raise ValueError("At least one image prediction is required.")

        disease_names = [str(item["disease"]) for item in predictions]
        embeddings = self._embed([cleaned_symptoms, *disease_names])
        similarities = cosine_similarity(
            embeddings[0:1], embeddings[1:]
        ).flatten()

        alignments = []
        for prediction, similarity in zip(predictions, similarities):
            normalized = (float(similarity) + 1.0) / 2.0
            alignment_score = float(max(0.0, min(1.0, normalized)) * 100.0)
            image_confidence = float(prediction["confidence"]) * 100.0
            combined_score = image_confidence * 0.6 + alignment_score * 0.4
            alignments.append({
                "disease": prediction["disease"],
                "image_confidence": image_confidence,
                "symptom_alignment": alignment_score,
                "combined_score": combined_score,
            })

        top_alignment = alignments[0]["symptom_alignment"]
        if top_alignment >= self.CONSISTENT_THRESHOLD:
            consistency = "CONSISTENT"
        elif top_alignment >= self.NEUTRAL_THRESHOLD:
            consistency = "NEUTRAL"
        else:
            consistency = "INCONSISTENT"
        top_disease = alignments[0]["disease"]
        return {
            "alignments": alignments,
            "consistency": consistency,
            "summary": (
                f"Patient symptoms are {consistency} with {top_disease} diagnosis"
            ),
        }


multimodal_analyzer = MultimodalAnalyzer()