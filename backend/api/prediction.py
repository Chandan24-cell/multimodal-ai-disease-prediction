# backend/api/prediction.py
import json
import io
import hashlib
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from cachetools import TTLCache
from typing import Dict, Any
import logging

from database.schemas import PredictionResponse, ExplainabilityResponse
from inference.pipeline import master_pipeline
from explainability.pipeline import explainability_pipeline
from api.auth import get_current_user
from inference.dicom_loader import DICOMImageLoader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Cache predictions for 1 hour based on image hash + text.
prediction_cache = TTLCache(maxsize=100, ttl=3600)


def get_cache_key(image_bytes: bytes, symptom_text: str, patient_data: dict) -> str:
    img_hash = hashlib.md5(image_bytes).hexdigest()
    return f"{img_hash}_{symptom_text}_{str(patient_data)}"

@router.post("/", response_model=PredictionResponse)
async def run_multimodal_prediction(
    image: UploadFile = File(..., description="Medical image (DICOM, PNG, JPG)"),
    symptom_text: str = Form(..., description="Patient symptoms"),
    patient_data_json: str = Form(..., description="Patient history JSON"),
    current_user: dict = Depends(get_current_user)
):
    """Run multimodal prediction supporting both DICOM and standard images."""
    try:
        patient_data = json.loads(patient_data_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for patient_data_json")
    
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file uploaded")

    cache_key = get_cache_key(image_bytes, symptom_text, patient_data)
    cached_response = prediction_cache.get(cache_key)
    if cached_response is not None:
        logger.info("Returning cached prediction")
        return cached_response
    
    file_extension = image.filename.lower().split('.')[-1] if image.filename else ''
    
    try:
        if file_extension in ['dcm', 'dicom']:
            logger.info(f"Processing DICOM file: {image.filename}")
            dicom_data = DICOMImageLoader.load_dicom_bytes(image_bytes)
            patient_data['dicom_metadata'] = dicom_data['metadata']
            image_bytes = DICOMImageLoader.convert_to_png_bytes(dicom_data['image'])
            original_image_for_cam = dicom_data['image']
            img_byte_arr = io.BytesIO()
            original_image_for_cam.save(img_byte_arr, format='PNG')
            original_image_bytes = img_byte_arr.getvalue()
        else:
            logger.info(f"Processing standard image: {image.filename}")
            original_image_bytes = image_bytes
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
    
    try:
        ml_results = master_pipeline.run(original_image_bytes, symptom_text, patient_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Pipeline failed: {str(e)}")
    
    fused_preds = ml_results["fused_prediction"]
    top_disease = max(fused_preds, key=fused_preds.get)
    target_class_idx = list(fused_preds.keys()).index(top_disease)
    
    try:
        exp_results = explainability_pipeline.generate_full_explanation(
            image_bytes=original_image_bytes,
            symptom_text=symptom_text,
            patient_data=patient_data,
            target_class_idx=target_class_idx
        )
    except Exception as e:
        logger.warning(f"Explainability pipeline failed: {e}")
        exp_results = {
            "image_heatmap": "",
            "text_attention": [],
            "tabular_shap": {}
        }
    
    response = PredictionResponse(
        image_prediction=ml_results["image_prediction"],
        text_prediction=ml_results["text_prediction"],
        fused_prediction=ml_results["fused_prediction"],
        final_diseases=ml_results["final_diseases"],
        confidence_score=ml_results["confidence_score"],
        explainability=ExplainabilityResponse(**exp_results)
    )
    prediction_cache[cache_key] = response
    return response
