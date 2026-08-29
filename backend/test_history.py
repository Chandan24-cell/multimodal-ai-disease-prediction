from inference.history_inference import history_inference

sample_patient_data = {
    "age": 65,
    "gender": "male",
    "vitals": {
        "heart_rate": 88,
        "systolic_bp": 135,
        "diastolic_bp": 85,
        "temperature": 99.1,
        "spo2": 96
    },
    "prior_conditions": ["hypertension", "smoking"]
}

embedding = history_inference.get_embedding(sample_patient_data)

print(f"Successfully generated history embedding!")
print(f"Embedding shape: {embedding.shape}") # Should be torch.Size([768])
print(f"Sample values: {embedding[:5]}")