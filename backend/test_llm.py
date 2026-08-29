import asyncio
from llm.report_generator import report_generator

async def main():
    patient_data = {
        "age": 68, "gender": "male",
        "symptoms": "Acute shortness of breath, fever, and productive cough for 3 days.",
        "vitals": {"heart_rate": 105, "systolic_bp": 140, "spo2": 91},
        "prior_conditions": ["hypertension", "copd"]
    }
       
    predictions = {
        "final_diseases": ["Pneumonia", "Effusion"],
        "confidence_score": 0.87,
        "image_prediction": {"Pneumonia": 0.85, "Normal": 0.15},
        "text_prediction": {"Pneumonia": 0.82, "Bronchitis": 0.18}
    }
       
    explainability = {
        "tabular_shap": {"age": 0.15, "spo2": -0.22, "copd": 0.18},
        "text_attention": [{"token": "shortness", "attention_weight": 0.85}, {"token": "fever", "attention_weight": 0.72}]
    }
       
    rag_context = [
        {
            "text": "Pneumonia in COPD patients often presents with exacerbated dyspnea and fever. Chest X-ray typically shows lobar consolidation. Treatment requires prompt antibiotics and oxygen support.",
            "metadata": {"source": "copd_guidelines.pdf"}
        }
    ]

    print("Generating AI Medical Report... (This may take 10-30 seconds depending on the LLM)")
    report = await report_generator.generate_report(patient_data, predictions, explainability, rag_context)
       
    print("\n" + "="*50)
    print(report["generated_report"])
    print("="*50)
    print(f"\nReferences: {report['retrieved_references']}")

if __name__ == "__main__":
    asyncio.run(main())