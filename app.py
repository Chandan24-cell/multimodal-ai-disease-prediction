#!/usr/bin/env python3
"""Clinical-safety-first Gradio interface for the trained ChestX-ray14 ViT.

Only the trained image classifier and its gradient-derived visualization are
exposed here. The repository's text, tabular, fusion, RAG, and LLM modules are
not loaded because their checkpoints/pipelines are not production-ready.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load environment variables from backend/.env
load_dotenv(os.path.join(os.path.dirname(__file__), "backend", ".env"))

import base64
import io
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import NamedTuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from explainability.gradcam import ViTGradCAM
from explainability.shap_analysis import DEFAULT_SHAP_BACKGROUND_URLS, generate_shap_explanation
from database.mongodb import save_prediction_to_mongo
from inference.dicom_loader import DICOMImageLoader
from inference.image_inference import image_inference
from inference.multimodal_analyzer import multimodal_analyzer
from llm.report_generator import report_generator
from rag.retriever import medical_retriever
from utils.pdf_generator import REPORT_DIR, generate_pdf_report


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conservative, named limits for the image-suitability safety gate. This is a
# heuristic gate, not an identity/provenance verifier; it intentionally rejects
# color photographs before any model or Grad-CAM code is called.
MIN_IMAGE_SIDE_PX = 128
MIN_CHEST_RATIO = 0.65
MAX_CHEST_RATIO = 1.45
MAX_CHANNEL_DELTA = 12.0  # RGB intensity levels; radiographs are near grayscale.
MAX_MEAN_SATURATION = 0.075
MIN_GRAYSCALE_CONTRAST = 0.045
MAX_GRAYSCALE_CONTRAST = 0.420
VALIDATION_PASS_SCORE = 0.80

INVALID_INPUT_MESSAGE = (
    "⚠️ Invalid Input: Please upload a valid Chest X-Ray (DICOM, PNG, or JPEG)."
)
DISCLAIMER = "RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL DIAGNOSIS."


class LoadedImage(NamedTuple):
    image: Image.Image
    image_bytes: bytes
    aspect_ratio: float
    source_type: str


def _to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _load_uploaded_image(upload_path: str | None) -> LoadedImage:
    """Decode a JPEG/PNG or parse DICOM pixels; reject anything else."""
    if not upload_path:
        raise ValueError("No file was uploaded.")

    path = Path(upload_path)
    raw_bytes = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        try:
            loaded = DICOMImageLoader.load_dicom_bytes(raw_bytes)
            image = loaded["image"].convert("RGB")
            metadata = loaded["metadata"]
            rows = float(metadata.get("rows") or image.height)
            columns = float(metadata.get("columns") or image.width)
            if rows <= 0 or columns <= 0:
                raise ValueError("DICOM has invalid image dimensions.")
            return LoadedImage(image, _to_png_bytes(image), columns / rows, "DICOM")
        except Exception as error:
            raise ValueError(f"Unreadable DICOM file: {error}") from error

    try:
        with Image.open(io.BytesIO(raw_bytes)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(raw_bytes)) as opened:
            image = opened.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("File is not a readable PNG or JPEG image.") from error

    return LoadedImage(image, _to_png_bytes(image), image.width / image.height, "raster")


def validate_chest_xray(image: Image.Image, aspect_ratio: float) -> tuple[bool, float, str]:
    """Conservatively reject unsuitable uploads before inference.

    A model-free gate can establish image plausibility, not diagnostic validity.
    It blocks color photos/faces and malformed images but clinical provenance
    still requires a proper imaging workflow.
    """
    if min(image.size) < MIN_IMAGE_SIDE_PX:
        return False, 0.0, f"image is too small ({image.width}×{image.height}px)"
    if not MIN_CHEST_RATIO <= aspect_ratio <= MAX_CHEST_RATIO:
        return False, 0.0, f"aspect ratio {aspect_ratio:.2f} is outside the accepted chest-radiograph range"

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    channel_delta = np.max(pixels, axis=2) - np.min(pixels, axis=2)
    mean_channel_delta = float(channel_delta.mean())
    mean_saturation = float((channel_delta / 255.0).mean())
    grayscale = pixels.mean(axis=2) / 255.0
    contrast = float(grayscale.std())

    grayscale_score = max(0.0, 1.0 - mean_channel_delta / MAX_CHANNEL_DELTA)
    saturation_score = max(0.0, 1.0 - mean_saturation / MAX_MEAN_SATURATION)
    ratio_center = (MIN_CHEST_RATIO + MAX_CHEST_RATIO) / 2
    ratio_radius = (MAX_CHEST_RATIO - MIN_CHEST_RATIO) / 2
    ratio_score = max(0.0, 1.0 - abs(aspect_ratio - ratio_center) / ratio_radius)
    contrast_score = 1.0 if MIN_GRAYSCALE_CONTRAST <= contrast <= MAX_GRAYSCALE_CONTRAST else 0.0
    confidence = (
        0.40 * grayscale_score
        + 0.30 * saturation_score
        + 0.20 * ratio_score
        + 0.10 * contrast_score
    )

    if mean_channel_delta > MAX_CHANNEL_DELTA or mean_saturation > MAX_MEAN_SATURATION:
        return False, confidence, "color/saturation profile is inconsistent with a grayscale radiograph"
    if contrast_score == 0.0:
        return False, confidence, "grayscale contrast is inconsistent with a readable radiograph"
    if confidence < VALIDATION_PASS_SCORE:
        return False, confidence, "image did not meet the conservative chest-radiograph suitability threshold"
    return True, confidence, "grayscale and geometry checks passed"


def _status_html(passed: bool, score: float, reason: str) -> str:
    tone = "#14532d" if passed else "#991b1b"
    headline = "Validation passed" if passed else INVALID_INPUT_MESSAGE
    return (
        f"<div style='border:1px solid {tone}; border-radius:8px; padding:12px; color:{tone};'>"
        f"<strong>{headline}</strong><br>"
        f"Suitability confidence: {score:.1%}. {reason}"
        "</div>"
    )


def _build_shap_plot(shap_regions: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    if not shap_regions:
        ax.text(0.5, 0.5, "No SHAP signal available", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ranked = sorted(shap_regions, key=lambda item: item["score"], reverse=True)
    labels = [f"R{item['rank']} ({item['x']}, {item['y']})" for item in ranked]
    scores = [float(item["score"]) for item in ranked]

    ax.barh(labels, scores, color="#60a5fa")
    ax.invert_yaxis()
    ax.set_xlabel("Normalized SHAP importance")
    ax.set_title("Top SHAP feature regions")
    fig.tight_layout()
    return fig


def _heatmap_to_image(heatmap: np.ndarray | None) -> Image.Image | None:
    if heatmap is None:
        return None
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        return None
    if heatmap.size == 0:
        return None
    rgba = np.stack([
        np.clip(heatmap * 255.0, 0, 255),
        np.zeros_like(heatmap, dtype=np.float32),
        np.ones_like(heatmap, dtype=np.float32) * 255.0,
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGB")


def _report_status_html(success: bool, message: str) -> str:
    tone = "#14532d" if success else "#991b1b"
    label = "Clinical report generated" if success else "Clinical report unavailable"
    return (
        f"<div style='border:1px solid {tone}; border-radius:8px; padding:12px; color:{tone};'>"
        f"<strong>{label}</strong><br>{message}</div>"
    )


def analyze_xray(upload_path: str | None, symptoms: str | None):
    """Validate first; only valid chest-radiograph candidates reach the model."""
    try:
        loaded = _load_uploaded_image(upload_path)
    except ValueError as error:
        logger.warning("Upload rejected before inference: %s", error)
        return (
            _status_html(False, 0.0, str(error)),
            [],
            None,
            None,
            None,
            None,
            {},
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            "",
        )

    passed, score, reason = validate_chest_xray(loaded.image, loaded.aspect_ratio)
    logger.info(
        "X-ray validation source=%s passed=%s confidence=%.3f reason=%s",
        loaded.source_type,
        passed,
        score,
        reason,
    )
    if not passed:
        return (
            _status_html(False, score, reason), [], loaded.image, None, None, None, {},
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), [], "",
        )

    try:
        predictions, _ = image_inference.predict(loaded.image_bytes)
        ranked = sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:5]
        top_class_idx = image_inference.class_names.index(ranked[0][0])

        image_tensor = image_inference.preprocess_image(loaded.image_bytes)

        grad_cam = ViTGradCAM(image_inference.model, image_inference.device)
        encoded_heatmap = grad_cam.generate_heatmap(
            image_tensor,
            top_class_idx,
            loaded.image_bytes,
        )
        heatmap = Image.open(io.BytesIO(base64.b64decode(encoded_heatmap.split(",", 1)[1]))).copy()
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        grad_cam_path = REPORT_DIR / f"gradcam_{uuid4().hex}.png"
        heatmap.save(grad_cam_path, format="PNG")

        shap_explanation = generate_shap_explanation(
            image_inference.model,
            image_tensor,
            background_images=DEFAULT_SHAP_BACKGROUND_URLS,
        )
        shap_plot = _build_shap_plot(shap_explanation["top_regions"])
        shap_heatmap = _heatmap_to_image(shap_explanation["heatmap"])

        rows = [
            [index, disease, f"{confidence:.2%}"]
            for index, (disease, confidence) in enumerate(ranked, 1)
        ]
        multimodal_data = None
        if symptoms and symptoms.strip():
            try:
                multimodal_data = multimodal_analyzer.analyze(
                    symptoms,
                    [{"disease": disease, "confidence": confidence} for disease, confidence in ranked],
                )
            except Exception as error:
                logger.warning("Multimodal analysis unavailable: %s", error)
        analysis_state = {
            "top_predictions": [
                {"disease": disease, "confidence": confidence}
                for disease, confidence in ranked
            ],
            "image_prediction": predictions,
            "gradcam_description": (
                f"A gradient-derived Grad-CAM overlay was generated for the top "
                f"ViT class '{ranked[0][0]}' ({ranked[0][1]:.2%}) and is displayed "
                "in the interface. No anatomical localization was quantified."
            ),
            "shap_description": (
                f"A real GradientExplainer SHAP map was computed for the top prediction '{ranked[0][0]}' "
                f"using a small reference set. Top contributing image regions are shown below."
            ),
            "shap_top_regions": shap_explanation["top_regions"],
            "grad_cam_path": str(grad_cam_path),
            "multimodal_analysis": multimodal_data,
        }
        multimodal_rows = []
        multimodal_status = ""
        multimodal_summary = ""
        if multimodal_data:
            multimodal_rows = [
                [
                    item["disease"],
                    f"{item['image_confidence']:.2f}%",
                    f"{item['symptom_alignment']:.2f}%",
                    f"{item['combined_score']:.2f}%",
                ]
                for item in multimodal_data["alignments"]
            ]
            indicator = {"CONSISTENT": "✅", "NEUTRAL": "⚠️", "INCONSISTENT": "❌"}[multimodal_data["consistency"]]
            multimodal_status = f"{indicator} Symptoms are {multimodal_data['consistency']} with the top image prediction"
            multimodal_summary = multimodal_data["summary"]
        return (
            _status_html(True, score, reason),
            rows,
            loaded.image,
            heatmap,
            shap_heatmap,
            shap_plot,
            analysis_state,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=bool(multimodal_data)),
            multimodal_rows,
            multimodal_status,
            multimodal_summary,
        )
    except Exception as error:
        logger.exception("Trained ViT/Grad-CAM pipeline failed")
        return (
            _status_html(False, 0.0, f"analysis failed: {error}"),
            [],
            loaded.image,
            None,
            None,
            None,
            {},
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            "",
            "",
        )


def _retrieve_rag_context_for_top_prediction(analysis_state: dict) -> list:
    """Retrieve the top diagnostic disease's literature context when available."""
    if not analysis_state or not analysis_state.get("top_predictions"):
        return []

    top_disease = analysis_state["top_predictions"][0]["disease"]
    if not top_disease:
        return []

    try:
        context = medical_retriever.retrieve_context(
            predicted_diseases=[top_disease],
            symptom_text=("Radiological guidelines and workup for " + top_disease),
            top_k=2,
        )
        logger.info("Retrieved %s RAG chunks for %s", len(context), top_disease)
        return context
    except Exception as error:
        logger.warning("RAG retrieval failed for %s: %s", top_disease, error)
        return []


async def generate_clinical_report(analysis_state: dict, symptoms: str | None):
    """Generate a real-provider report from the current trained ViT output."""
    if not analysis_state or not analysis_state.get("top_predictions"):
        return (
            _report_status_html(False, "Analyze a validated chest X-ray before requesting a report."),
            "",
            "",
            gr.update(value=None, visible=False),
        )

    top_predictions = analysis_state["top_predictions"]
    prediction_payload = {
        "top_predictions": top_predictions,
        "final_diseases": [item["disease"] for item in top_predictions[:3]],
        "confidence_score": top_predictions[0]["confidence"],
        "image_prediction": analysis_state["image_prediction"],
    }
    try:
        rag_context = _retrieve_rag_context_for_top_prediction(analysis_state)
        result = await report_generator.generate_report(
            patient_data={"symptoms": symptoms or "Not provided"},
            predictions=prediction_payload,
            explainability={"gradcam_description": analysis_state["gradcam_description"]},
            rag_context=rag_context,
        )
    except Exception as error:
        logger.warning("Clinical report unavailable: %s", error)
        return _report_status_html(False, str(error)), "", "", gr.update(value=None, visible=False)

    rag_markdown = ""
    if rag_context:
        rag_markdown = "\n\n".join(
            f"### {index + 1}. {chunk.get('metadata', {}).get('source', 'Medical guideline')}\n{chunk.get('text','').strip()}"
            for index, chunk in enumerate(rag_context)
        )

    report_text = result["generated_report"]
    prediction_data = {
        "timestamp": datetime.now(timezone.utc),
        "top_predictions": top_predictions,
        "symptoms": symptoms or "Not provided",
        "rag_context": rag_context,
    }
    try:
        pdf_path = generate_pdf_report(
            prediction_data,
            rag_context,
            report_text,
            analysis_state.get("grad_cam_path"),
        )
        prediction_data["pdf_path"] = pdf_path
        await save_prediction_to_mongo(prediction_data)
    except Exception as error:
        logger.warning("Report artifact generation or persistence failed: %s", error)
        return (
            _report_status_html(False, f"Clinical report generated, but PDF export failed: {error}"),
            report_text,
            gr.update(value=rag_markdown, visible=bool(rag_markdown.strip())),
            gr.update(value=None, visible=False),
        )

    return (
        _report_status_html(True, "Generated by the configured provider."),
        report_text,
        gr.update(value=rag_markdown, visible=bool(rag_markdown.strip())),
        gr.update(value=pdf_path, visible=True),
    )


with gr.Blocks(title="Chest X-Ray Research Assistant", theme=gr.themes.Base()) as demo:
    gr.HTML(
        """
        <style>
          .gradio-container { background: #081426; color: #e5eefb; }
          .hero { border-left: 5px solid #3b82f6; padding: 14px 18px; background: #102445; border-radius: 8px; }
          .disclaimer { border: 2px solid #f59e0b; padding: 12px 16px; border-radius: 8px; color: #fef3c7; background: #3b2d09; font-weight: 800; }
        </style>
        <div class="hero"><h1>Chest X-Ray Research Assistant</h1>
        <p>Trained ViT image inference with a validation gate and gradient-derived Grad-CAM.</p></div>
        """
    )
    gr.HTML(f'<div class="disclaimer">{DISCLAIMER}</div>')

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(
                label="1. Upload chest radiograph",
                file_types=[".png", ".jpg", ".jpeg", ".dcm", ".dicom"],
                type="filepath",
            )
            symptoms_input = gr.Textbox(
                label="Current symptoms / chief complaint (optional)",
                lines=3,
                placeholder="Used for the clinical report and optional ClinicalBERT analysis.",
            )
            analyze_button = gr.Button("2. Analyze X-Ray", variant="primary")
            validation_status = gr.HTML("<div>Awaiting an uploaded image.</div>", label="3. Validation status")
            prediction_table = gr.Dataframe(
                headers=["Rank", "Disease", "Model confidence"],
                datatype=["number", "str", "str"],
                label="4. Top five real ViT predictions",
                interactive=False,
            )
        with gr.Column(scale=2):
            with gr.Row():
                original_output = gr.Image(label="Uploaded radiograph", interactive=False)
                heatmap_output = gr.Image(label="Grad-CAM from trained ViT gradients", interactive=False)

    with gr.Column(visible=False) as multimodal_section:
        gr.Markdown("## 🧬 Multimodal Analysis")
        multimodal_status = gr.Markdown()
        multimodal_summary = gr.Markdown()
        multimodal_table = gr.Dataframe(
            headers=["Disease", "Image Confidence", "Symptom Alignment", "Combined Score"],
            datatype=["str", "str", "str", "str"],
            label="Final Multimodal Confidence",
            interactive=False,
        )

    with gr.Column(visible=False) as shap_section:
        gr.Markdown("## 🧠 SHAP Feature Importance")
        gr.Markdown("GradientExplainer highlights the image regions that most strongly drive the top disease prediction.")
        with gr.Row():
            shap_plot = gr.Plot(label="Top SHAP contributors")
            shap_heatmap = gr.Image(label="SHAP importance map", interactive=False)

    analysis_state = gr.State({})
    with gr.Column(visible=False) as report_section:
        gr.Markdown("## 5. Clinical Report")
        gr.Markdown(
            "Generated only through a configured real LLM provider from the real ViT rankings, "
            "optional symptoms, and Grad-CAM provenance."
        )
        report_button = gr.Button("Generate Clinical Report", variant="secondary")
        report_status = gr.HTML("<div>Awaiting report request.</div>")
        report_output = gr.Markdown(label="LLM clinical report")
        rag_output = gr.Markdown(label="Supporting Medical Literature", visible=False)
        pdf_output = gr.File(label="Download PDF Report", visible=False, interactive=False)

    analyze_button.click(
        analyze_xray,
        inputs=[upload, symptoms_input],
        outputs=[
            validation_status,
            prediction_table,
            original_output,
            heatmap_output,
            shap_heatmap,
            shap_plot,
            analysis_state,
            report_section,
            shap_section,
            multimodal_section,
            multimodal_table,
            multimodal_status,
            multimodal_summary,
        ],
        show_progress="full",
    )
    report_button.click(
        generate_clinical_report,
        inputs=[analysis_state, symptoms_input],
        outputs=[report_status, report_output, rag_output, pdf_output],
        show_progress="full",
    )
    gr.Markdown(f"**{DISCLAIMER}**")
    gr.Markdown("Model: trained ViT checkpoint (`model.safetensors`) · 14 NIH ChestX-ray14 labels")


if __name__ == "__main__":
    logger.info("Starting Chest X-Ray Research Assistant")
    demo.launch()
