from rag.retriever import medical_retriever

predicted_diseases = ["Pneumonia"]
symptoms = "Patient has a high fever, productive cough, and chest pain."

print("Searching for medical context...")
context = medical_retriever.retrieve_context(predicted_diseases, symptoms, top_k=2)

print(f"\nRetrieved {len(context)} chunks:")
for i, chunk in enumerate(context):
    print(f"\n--- Chunk {i+1} (Score: {chunk['score']:.4f}) ---")
    print(f"Source: {chunk['metadata']['source']}")
    print(chunk['text'][:200] + "...")