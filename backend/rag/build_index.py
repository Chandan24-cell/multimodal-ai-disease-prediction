# backend/rag/build_index.py
import logging
import sys
import os

# Keep native numerical libraries single-threaded on macOS to avoid backend crashes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Ensure backend is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.vector_store import vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info("Starting RAG Index Builder...")
    
    # Initialize store and build
    vector_store.build_index(docs_dir="../knowledge_base/documents")
    
    logging.info("RAG Index build complete.")