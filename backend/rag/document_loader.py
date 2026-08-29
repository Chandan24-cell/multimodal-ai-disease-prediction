# backend/rag/document_loader.py
import os
import logging
from typing import List, Dict
from pypdf import PdfReader

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Loads text and PDF documents from the knowledge base directory.
    
    Input: Path to a directory containing .txt and .pdf files.
    Output: List of dictionaries containing 'text' and 'metadata' (source file).
    """
    def __init__(self, docs_dir: str = "knowledge_base/documents"):
        self.docs_dir = docs_dir
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            logger.warning(f"Created missing directory: {self.docs_dir}. Please add medical documents here.")

    def load_documents(self) -> List[Dict[str, str]]:
        """Scans the directory and extracts text from supported file types."""
        documents = []
        
        for filename in os.listdir(self.docs_dir):
            file_path = os.path.join(self.docs_dir, filename)
            
            if filename.endswith(".txt"):
                text = self._load_txt(file_path)
            elif filename.endswith(".pdf"):
                text = self._load_pdf(file_path)
            else:
                continue
                
            if text.strip():
                documents.append({
                    "text": text,
                    "metadata": {"source": filename}
                })
                
        logger.info(f"Loaded {len(documents)} documents from {self.docs_dir}")
        return documents

    def _load_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text