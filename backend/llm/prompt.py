"""Prompt construction for the image-model clinical report feature."""

from typing import Any, Dict, List


class PromptBuilder:
    """Build constrained, evidence-labelled report prompts."""

    MEDICAL_DISCLAIMER = (
        "RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL DIAGNOSIS. This AI-generated "
        "text is not medical advice and must be reviewed by a licensed clinician."
    )

    @staticmethod
    def build_system_prompt() -> str:
        return f"""You draft a cautious research-prototype radiology support report.

The only imaging evidence supplied is a trained ViT's multi-label probability
ranking and the fact that a Grad-CAM visualization was generated. You are not a
radiologist and must not claim to have inspected image pixels, heatmap anatomy,
or findings not explicitly supplied. Probabilities are not diagnoses.

Rules:
1. Begin and end with this exact disclaimer: {PromptBuilder.MEDICAL_DISCLAIMER}
2. Do not invent symptoms, patient facts, anatomical observations, guidelines,
   test results, citations, or certainty.
3. Use conditional language such as "model-ranked possibility" and
   "consider clinical correlation".
4. Format strictly in Markdown with these headings:
   ## Chief Complaint
   ## Model-Ranked Radiographic Possibilities
   ## Differential Diagnosis
   ## Recommended Workup
   ## Clinical Pearls
   ## Explainability Note
   ## Disclaimer
"""

    @staticmethod
    def build_user_prompt(
        patient_data: Dict[str, Any],
        predictions: Dict[str, Any],
        explainability: Dict[str, Any],
        rag_context: List[Dict[str, str]],
    ) -> str:
        symptoms = patient_data.get("symptoms", "") or "Not provided"
        ranked = predictions.get("top_predictions", [])
        if not ranked:
            ranked = [
                {"disease": disease, "confidence": score}
                for disease, score in predictions.get("image_prediction", {}).items()
            ]

        prediction_lines = []
        for item in ranked[:5]:
            disease = item.get("disease", "Unknown")
            confidence = float(item.get("confidence", 0.0))
            prediction_lines.append(f"- {disease}: {confidence:.2%} (ViT probability, not a diagnosis)")
        if not prediction_lines:
            prediction_lines.append("- No model predictions were provided.")

        gradcam_description = explainability.get(
            "gradcam_description",
            "No Grad-CAM artifact was supplied.",
        )
        context_lines = []
        for chunk in rag_context:
            source = chunk.get("metadata", {}).get("source", "Unknown source")
            snippet = chunk.get("text", "")[:300]
            context_lines.append(f"- {source}: {snippet}")

        return f"""### INPUTS
Current symptoms / chief complaint: {symptoms}

### TRAINED IMAGE MODEL OUTPUT
{chr(10).join(prediction_lines)}

### EXPLAINABILITY ARTIFACT
{gradcam_description}

### RETRIEVED CONTEXT
{chr(10).join(context_lines) if context_lines else "No literature was supplied; do not add citations."}

Write the requested structured report using only these inputs and the safety rules.
"""
