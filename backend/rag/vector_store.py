# backend/rag/vector_store.py
import os
from pathlib import Path
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

from rag.document_loader import DocumentLoader
from rag.chunker import RecursiveTextChunker

logger = logging.getLogger(__name__)

class MedicalVectorStore:
    """
    Manages the FAISS vector index for the RAG pipeline.
    Uses cosine similarity (Inner Product on normalized vectors).
    
    Embedding Dimension: 384 (all-MiniLM-L6-v2)
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", index_path: str = None):
        if index_path is None:
            index_path = str(Path(__file__).resolve().parent.parent / "data" / "vector_store")
        self.model_name = model_name
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "medical_index.faiss")
        self.chunks_file = os.path.join(index_path, "chunks.npy")
        
        os.makedirs(index_path, exist_ok=True)
        
        # Do not download or initialize the embedding model at application import
        # time. A RAG query is the first operation that needs it.
        self.embedding_model: Optional[SentenceTransformer] = None
        self.dimension: Optional[int] = None
        
        self.chunks: List[Dict[str, str]] = []
        self.index = None
        
        # Try to load existing index
        self._load_index()

    def _load_index(self):
        """Loads the FAISS index and chunks from disk if they exist."""
        if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
            logger.info("Loading existing FAISS index from disk...")
            self.index = faiss.read_index(self.index_file)
            self.dimension = self.index.d
            self.chunks = np.load(self.chunks_file, allow_pickle=True).tolist()
            logger.info(f"Loaded index with {self.index.ntotal} vectors.")
        else:
            logger.info("No existing index found. It will be initialized when building the index.")
            self.index = None

    def _get_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model only when a query or index build requires it."""
        if self.embedding_model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.embedding_model = SentenceTransformer(self.model_name, device="cpu")
                self.dimension = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Successfully loaded embedding model. Dimension: {self.dimension}")
            except Exception as e:
                logger.error(
                    f"Failed to load embedding model '{self.model_name}': {e}. "
                    f"Vector store will return empty results. Please verify sentence-transformers installation."
                )
                self.embedding_model = None
                return None
        return self.embedding_model

    def build_index(self, docs_dir: str = "knowledge_base/documents"):
        """Ingests documents, chunks them, embeds them, and builds the FAISS index."""
        logger.info("Starting RAG index build process...")
        
        # 1. Load and Chunk
        loader = DocumentLoader(docs_dir)
        docs = loader.load_documents()
        
        if not docs:
            logger.warning("No documents found to build index. Please add PDFs/TXTs to knowledge_base/documents/")
            return
            
        chunker = RecursiveTextChunker(chunk_size=500, chunk_overlap=50)
        self.chunks = chunker.chunk_documents(docs)
        
        # 2. Try to get embedding model
        embedding_model = self._get_embedding_model()
        if embedding_model is None:
            logger.error("Cannot build index: embedding model failed to load. Index will remain empty.")
            return
        
        # 3. Embed
        texts = [chunk["text"] for chunk in self.chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        
        try:
            embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Error during embedding: {e}. Index build failed.")
            self.chunks = []
            self.index = None
            return
        
        # 4. Normalize for Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        # 5. Add to FAISS
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        # 6. Save to disk
        try:
            faiss.write_index(self.index, self.index_file)
            np.save(self.chunks_file, self.chunks)
            logger.info(f"FAISS index built and saved with {self.index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")


    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Searches the vector store for the top-k most relevant chunks.
        
        Input: Query string (e.g., "Pneumonia symptoms and treatment guidelines").
        Output: List of top-k chunk dictionaries.
        
        Returns empty list if model fails to load or index is empty.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty. Cannot perform search.")
            return []
        
        # Try to get embedding model; if it fails, return empty results
        embedding_model = self._get_embedding_model()
        if embedding_model is None:
            logger.warning("Embedding model failed to load. Returning empty search results.")
            return []
            
        try:
            # Embed and normalize query
            query_embedding = embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Retrieve chunks
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1:  # FAISS returns -1 if not enough results
                    chunk = self.chunks[idx]
                    results.append({
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "score": float(score)
                    })
                    
            return results
        except Exception as e:
            logger.error(f"Error during vector search: {e}. Returning empty results.")
            return []

# Global instance
vector_store = MedicalVectorStore()
