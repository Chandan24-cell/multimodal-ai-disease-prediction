# backend/api/prediction.py
import json
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from typing import Dict, Any
import logging

from database.schemas import PredictionResponse, ExplainabilityResponse
from inference.image_inference import image_inference
from api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/", response_model=PredictionResponse)
async def run_multimodal_prediction(
    image: UploadFile = File(..., description="Medical image (DICOM, PNG, JPG)"),
    symptom_text: str = Form(..., description="Patient symptoms"),
    patient_data_json: str = Form(..., description="Patient history JSON"),
    current_user: dict = Depends(get_current_user)
):
    """Run image-only prediction for standard image uploads."""
    try:
        patient_data = json.loads(patient_data_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for patient_data_json")
    
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file uploaded")

    file_extension = image.filename.lower().split('.')[-1] if image.filename else ''
    
    if file_extension in ['dcm', 'dicom']:
        raise HTTPException(
            status_code=400,
            detail="DICOM is not available in the free deployment"
        )

    original_image_bytes = image_bytes
    
    try:
        image_prediction, image_embedding = image_inference.predict(
            original_image_bytes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Pipeline failed: {str(e)}")
    
    fused_preds = image_prediction
    top_disease = max(fused_preds, key=fused_preds.get)
    exp_results = {
        "image_heatmap": "",
        "text_attention": [],
        "tabular_shap": {}
    }
    
    response = PredictionResponse(
        image_prediction=image_prediction,
        text_prediction=image_prediction,
        fused_prediction=image_prediction,
        final_diseases=[top_disease],
        confidence_score=float(fused_preds[top_disease]),
        explainability=ExplainabilityResponse(**exp_results)
    )
    return response
