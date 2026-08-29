# backend/rag/chunker.py
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RecursiveTextChunker:
    """
    Splits large documents into smaller, overlapping chunks for embedding.
    
    Input: List of document dictionaries (from DocumentLoader).
    Output: List of chunk dictionaries with 'text', 'metadata', and 'chunk_id'.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Splits documents into smaller chunks."""
        all_chunks = []
        
        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]
            
            # Split by paragraphs first to preserve semantic boundaries
            paragraphs = text.split("\n\n")
            
            current_chunk = ""
            chunk_id = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                    
                # If adding this paragraph exceeds chunk_size, save current chunk and start new one
                if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                    all_chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {**metadata, "chunk_id": chunk_id}
                    })
                    chunk_id += 1
                    
                    # Keep overlap
                    current_chunk = current_chunk[-self.chunk_overlap:] + "\n" + paragraph
                else:
                    current_chunk += "\n" + paragraph
                    
            # Add the last remaining chunk
            if current_chunk.strip():
                all_chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {**metadata, "chunk_id": chunk_id}
                })
                
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents.")
        return all_chunks