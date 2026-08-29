# backend/rag/vector_store.py
import os
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple
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
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", index_path: str = "data/vector_store"):
        self.model_name = model_name
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "medical_index.faiss")
        self.chunks_file = os.path.join(index_path, "chunks.npy")
        
        os.makedirs(index_path, exist_ok=True)
        
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device="cpu")
        self.dimension = self.embedding_model.get_sentence_embedding_dimension() # 384
        
        self.chunks: List[Dict[str, str]] = []
        self.index = None
        
        # Try to load existing index
        self._load_index()

    def _load_index(self):
        """Loads the FAISS index and chunks from disk if they exist."""
        if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
            logger.info("Loading existing FAISS index from disk...")
            self.index = faiss.read_index(self.index_file)
            self.chunks = np.load(self.chunks_file, allow_pickle=True).tolist()
            logger.info(f"Loaded index with {self.index.ntotal} vectors.")
        else:
            logger.info("No existing index found. Initializing empty FAISS index.")
            # IndexFlatIP = Inner Product (Cosine similarity when vectors are normalized)
            self.index = faiss.IndexFlatIP(self.dimension)

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
        
        # 2. Embed
        texts = [chunk["text"] for chunk in self.chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # 3. Normalize for Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        # 4. Add to FAISS
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        # 5. Save to disk
        faiss.write_index(self.index, self.index_file)
        np.save(self.chunks_file, self.chunks)
        
        logger.info(f"FAISS index built and saved with {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Searches the vector store for the top-k most relevant chunks.
        
        Input: Query string (e.g., "Pneumonia symptoms and treatment guidelines").
        Output: List of top-k chunk dictionaries.
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty. Cannot perform search.")
            return []
            
        # Embed and normalize query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
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

# Global instance
vector_store = MedicalVectorStore()