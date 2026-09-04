# backend/rag/retriever.py
import logging
from typing import Dict, List

from rag.vector_store import vector_store

logger = logging.getLogger(__name__)


class MedicalRetriever:
    """Formulates the RAG query based on the multimodal prediction and retrieves context."""

    def __init__(self):
        self.store = vector_store

    def retrieve_context(
        self,
        predicted_diseases: List[str],
        symptom_text: str,
        top_k: int = 3,
    ) -> List[Dict[str, str]]:
        """Combine disease names and symptom context into a query and return relevant chunks."""
        if not predicted_diseases and not symptom_text.strip():
            logger.warning("No diseases or symptoms provided for RAG query.")
            return []

        disease_str = ", ".join(disease.strip() for disease in predicted_diseases if disease and disease.strip())
        if not disease_str:
            disease_str = "radiological findings"

        if symptom_text and symptom_text.strip():
            query = (
                f"Radiological guidelines and workup for {disease_str}. "
                f"Clinical context: {symptom_text.strip()}"
            )
        else:
            query = f"Radiological guidelines and workup for {disease_str}."

        logger.info("RAG Query: %s", query)
        context = self.store.search(query, top_k=top_k)
        logger.info("Retrieved %s context chunks.", len(context))
        return context


medical_retriever = MedicalRetriever()