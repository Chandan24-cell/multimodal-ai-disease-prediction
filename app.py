#!/usr/bin/env python3
"""Standalone Gradio interface for the multimodal healthcare prototype."""
import io
import base64
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from inference.image_inference import image_inference


def predict_disease(image, symptoms, age, gender):
    """Run image prediction and return formatted results with an optional heatmap."""
    if image is None:
        return "Please upload a medical image.", None

    if isinstance(image, Image.Image):
        image_bytes = io.BytesIO()
        image.convert("RGB").save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()
    else:
        image_bytes = Path(image).read_bytes()

    predictions, _ = image_inference.predict(image_bytes)
    ranked_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    top_predictions = ranked_predictions[:5]

    patient_summary = f"Age: {age or 'Not provided'}\nGender: {gender or 'Not provided'}"
    symptom_summary = symptoms.strip() if symptoms else "Not provided"
    output_lines = [
        "AI Predictions",
        "===============",
        "",
        *[f"{index}. {name}: {probability:.1%}" for index, (name, probability) in enumerate(top_predictions, 1)],
        "",
        patient_summary,
        f"Symptoms: {symptom_summary}",
        "",
        "Research prototype only. Not for clinical diagnosis.",
    ]

    heatmap = None
    try:
        from explainability.gradcam import ViTGradCAM

        target_index = image_inference.class_names.index(top_predictions[0][0])
        gradcam = ViTGradCAM(image_inference.model, image_inference.device)
        heatmap_data = gradcam.generate_heatmap(
            image_inference.preprocess_image(image_bytes), target_index, image_bytes
        )
        heatmap = Image.open(io.BytesIO(base64.b64decode(heatmap_data.split(",", 1)[1])))
    except Exception:
        heatmap = None

    return "\n".join(output_lines), heatmap


with gr.Blocks(title="Multimodal Healthcare AI") as demo:
    gr.Markdown("# Multimodal Healthcare AI")
    gr.Markdown("Upload a medical image and provide optional patient context for research analysis.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Medical Image", type="pil")
            symptoms_input = gr.Textbox(label="Symptoms", lines=4, placeholder="Describe symptoms")
            age_input = gr.Number(label="Age", minimum=0, maximum=130, precision=0)
            gender_input = gr.Dropdown(["Male", "Female", "Other", "Prefer not to say"], label="Gender")
            submit_btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="AI Predictions", lines=10)
            grad_cam_output = gr.Image(label="Grad-CAM Visualization")

    submit_btn.click(
        fn=predict_disease,
        inputs=[image_input, symptoms_input, age_input, gender_input],
        outputs=[output_text, grad_cam_output],
    )

    gr.Markdown(
        "---\n**Disclaimer:** This is a research prototype for educational purposes only. "
        "Not for clinical diagnosis."
    )


if __name__ == "__main__":
    demo.launch()
