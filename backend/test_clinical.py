from inference.text_inference import text_inference
   
sample_symptoms = "Patient presents with acute shortness of breath, chest pain, and a persistent dry cough for the past 3 days."
   
predictions, embedding = text_inference.predict(sample_symptoms)
   
print("Top 3 Predictions:")
sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
for cls, prob in sorted_preds:
    print(f"  - {cls}: {prob:.4f}")
       
print(f"\nEmbedding shape: {embedding.shape}") # Should be torch.Size([768])