# backend/rag/vector_store.py
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rag.chunker import RecursiveTextChunker

logger = logging.getLogger(__name__)


class MedicalVectorStore:
    """Manage the optional FAISS vector index for the RAG pipeline."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", index_path: str = None):
        if index_path is None:
            index_path = str(Path(__file__).resolve().parent.parent / "data" / "vector_store")
        self.model_name = model_name
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "medical_index.faiss")
        self.chunks_file = os.path.join(index_path, "chunks.npy")
        os.makedirs(index_path, exist_ok=True)

        self.embedding_model = None
        self.dimension: Optional[int] = None
        self.chunks: List[Dict[str, str]] = []
        self.index = None
        self._load_index()

    def _load_index(self):
        """Load the FAISS index only when FAISS is installed and files exist."""
        if not (os.path.exists(self.index_file) and os.path.exists(self.chunks_file)):
            logger.info("No existing index found. It will be initialized when building the index.")
            return
        try:
            import faiss
            self.index = faiss.read_index(self.index_file)
            self.dimension = self.index.d
            self.chunks = np.load(self.chunks_file, allow_pickle=True).tolist()
            logger.info("Loaded index with %s vectors.", self.index.ntotal)
        except Exception:
            logger.exception("FAISS index could not be loaded; RAG search is disabled")
            self.index = None

    def _get_embedding_model(self):
        """Load the embedding model lazily and degrade gracefully on failure."""
        if self.embedding_model is None:
            try:
                logger.info("Loading embedding model: %s", self.model_name)
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.model_name, device="cpu")
                self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            except Exception:
                logger.exception("Embedding model failed to load; RAG search is disabled")
                self.embedding_model = None
                return None
        return self.embedding_model

    def build_index(self, docs_dir: str = "knowledge_base/documents"):
        """Ingest documents, build embeddings, and persist a FAISS index."""
        try:
            from rag.document_loader import DocumentLoader
        except Exception:
            logger.exception("Document loader unavailable; RAG index build is disabled")
            return
        docs = DocumentLoader(docs_dir).load_documents()
        if not docs:
            logger.warning("No documents found to build index.")
            return

        self.chunks = RecursiveTextChunker(chunk_size=500, chunk_overlap=50).chunk_documents(docs)
        embedding_model = self._get_embedding_model()
        if embedding_model is None:
            return
        try:
            import faiss
            texts = [chunk["text"] for chunk in self.chunks]
            embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
            faiss.write_index(self.index, self.index_file)
            np.save(self.chunks_file, self.chunks)
            logger.info("FAISS index built and saved with %s vectors.", self.index.ntotal)
        except Exception:
            logger.exception("RAG index build failed")
            self.index = None

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Return relevant chunks, or an empty list when RAG is unavailable."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty. Cannot perform search.")
            return []
        embedding_model = self._get_embedding_model()
        if embedding_model is None:
            return []
        try:
            import faiss
            query_embedding = embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, top_k)
            return [
                {
                    "text": self.chunks[index]["text"],
                    "metadata": self.chunks[index]["metadata"],
                    "score": float(score),
                }
                for index, score in zip(indices[0], scores[0])
                if index != -1
            ]
        except Exception:
            logger.exception("RAG search failed; returning empty results")
            return []


vector_store = MedicalVectorStore()
