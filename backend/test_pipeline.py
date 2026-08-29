from inference.pipeline import master_pipeline
from PIL import Image
import io

# 1. Dummy Image
img = Image.new('RGB', (224, 224), color='red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
image_bytes = img_byte_arr.getvalue()

# 2. Dummy Text
symptom_text = "Patient has severe chest pain and shortness of breath."

# 3. Dummy History
patient_data = {
    "age": 70,
    "gender": "male",
    "vitals": {
        "heart_rate": 100,
        "systolic_bp": 150,
        "diastolic_bp": 95,
        "temperature": 101.2,
        "spo2": 92
    },
    "prior_conditions": ["hypertension"]
}

# Run the master pipeline
results = master_pipeline.run(
    image_bytes,
    symptom_text,
    patient_data
)

print("\n--- FINAL RESULTS ---")
print(f"Final Diseases: {results['final_diseases']}")
print(f"Confidence Score: {results['confidence_score']:.4f}")
print(f"Fused Features Shape: {results['fused_features'].shape}")