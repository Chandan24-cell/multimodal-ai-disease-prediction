# backend/rag/retriever.py
import logging
from typing import List, Dict

from rag.vector_store import vector_store

logger = logging.getLogger(__name__)

class MedicalRetriever:
    """
    Formulates the RAG query based on the multimodal prediction and retrieves context.
    """
    def __init__(self):
        self.store = vector_store

    def retrieve_context(
        self, 
        predicted_diseases: List[str], 
        symptom_text: str, 
        top_k: int = 3
    ) -> List[Dict[str, str]]:
        """
        Combines predicted diseases and symptoms into a semantic query and retrieves medical context.
        
        Input: 
            - predicted_diseases: List of strings (e.g., ["Pneumonia", "Effusion"])
            - symptom_text: Raw string of patient symptoms.
        Output: 
            - List of retrieved context dictionaries.
        """
        if not predicted_diseases and not symptom_text.strip():
            logger.warning("No diseases or symptoms provided for RAG query.")
            return []

        # Formulate query
        # We prioritize the predicted disease, but include symptoms for specific context
        disease_str = ", ".join(predicted_diseases)
        query = f"Clinical guidelines, diagnosis, and treatment for {disease_str}. Patient presents with: {symptom_text}"
        
        logger.info(f"RAG Query: {query}")
        
        # Retrieve
        context = self.store.search(query, top_k=top_k)
        
        logger.info(f"Retrieved {len(context)} context chunks.")
        return context

# Global instance
medical_retriever = MedicalRetriever()