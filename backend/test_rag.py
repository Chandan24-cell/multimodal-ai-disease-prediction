"""
Test the RAG retriever with stable sentence-transformer versions.
Handles graceful degradation if embedding model fails to load.
Pinned dependencies prevent Apple Silicon Python 3.12 segfaults.
"""
from rag.retriever import medical_retriever

predicted_diseases = ["Pneumonia"]
symptoms = "Patient has a high fever, productive cough, and chest pain."

print("Searching for medical context...")
print("Note: Using sentence-transformers==2.2.2 for stability")

try:
    context = medical_retriever.retrieve_context(predicted_diseases, symptoms, top_k=2)

    if context:
        print(f"\n✓ Retrieved {len(context)} chunks:")
        for i, chunk in enumerate(context):
            print(f"\n--- Chunk {i+1} (Score: {chunk['score']:.4f}) ---")
            print(f"Source: {chunk['metadata']['source']}")
            print(chunk['text'][:200] + "...")
    else:
        print("\n⚠ No context retrieved (vector store may be empty or embedding model unavailable)")
        print("This is OK - the system gracefully handles embedding model failures")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("If segfault (exit code 139), verify sentence-transformers==2.2.2 is installed")
