from explainability.pipeline import explainability_pipeline
from PIL import Image
import io

# 1. Dummy Image
img = Image.new('RGB', (224, 224), color='blue')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
image_bytes = img_byte_arr.getvalue()

# 2. Dummy Text
symptom_text = "Patient has severe chest pain and shortness of breath."

# 3. Dummy History
patient_data = {
    "age": 70, "gender": "male",
    "vitals": {"heart_rate": 100, "systolic_bp": 150, "diastolic_bp": 95, "temperature": 101.2, "spo2": 92},
    "prior_conditions": ["hypertension"]
}

# Target class: Let's say index 6 is "Pneumonia"
target_idx = 6

print("Generating explanations... (This may take a few seconds)")
explanations = explainability_pipeline.generate_full_explanation(
    image_bytes, symptom_text, patient_data, target_idx
)

print("\n--- EXPLAINABILITY RESULTS ---")
print(f"Image Heatmap generated: {'Yes' if explanations['image_heatmap'].startswith('data:image') else 'No'}")
print(f"Text Attention Tokens: {len(explanations['text_attention'])} tokens analyzed.")
print("Top 3 important words:", sorted(explanations['text_attention'], key=lambda x: x['attention_weight'], reverse=True)[:3])
print("SHAP Values (Top 3 features):", sorted(explanations['tabular_shap'].items(), key=lambda x: abs(x[1]), reverse=True)[:3])