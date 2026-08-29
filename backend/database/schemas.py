# backend/database/schemas.py
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# ==========================================
# Enums
# ==========================================
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

# ==========================================
# Patient Schemas
# ==========================================
class PatientBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    age: int = Field(..., ge=0, le=120)
    gender: Gender
    contact: str = Field(..., min_length=10, max_length=15)

class PatientCreate(PatientBase):
    email: EmailStr
    medical_history: Optional[List[str]] = []

class PatientDB(PatientCreate):
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ==========================================
# Prediction Schemas
# ==========================================
class PredictionDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(alias="_id")
    patient_id: str
    image_prediction: Dict[str, float]  # e.g., {"Pneumonia": 0.8, "Normal": 0.2}
    text_prediction: Dict[str, float]
    history_prediction: Dict[str, float]
    fused_prediction: Dict[str, float]
    final_disease: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ==========================================
# Report Schemas
# ==========================================
class ReportDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(alias="_id")
    patient_id: str
    prediction_id: str
    generated_report: str
    retrieved_references: List[str]
    disclaimer: str = "DISCLAIMER: This is an AI-generated decision support report. It is not a substitute for professional medical advice, diagnosis, or treatment."
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ==========================================
# Auth Schemas
# ==========================================
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(alias="_id")
    username: str
    email: EmailStr
    full_name: str
    hashed_password: str
    is_active: bool = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None


# Add to the bottom of backend/database/schemas.py

# ==========================================
# API Request/Response Schemas
# ==========================================
class PredictionRequest(BaseModel):
    symptom_text: str
    patient_data: Dict[str, Any]

class ExplainabilityResponse(BaseModel):
    image_heatmap: str  # Base64 string
    text_attention: List[Dict[str, Any]]
    tabular_shap: Dict[str, float]

class PredictionResponse(BaseModel):
    patient_id: Optional[str] = None
    image_prediction: Dict[str, float]
    text_prediction: Dict[str, float]
    fused_prediction: Dict[str, float]
    final_diseases: List[str]
    confidence_score: float
    explainability: ExplainabilityResponse
    disclaimer: str = "DISCLAIMER: AI decision support prototype. Not for clinical diagnosis."

class ReportRequest(BaseModel):
    patient_data: Dict[str, Any]
    predictions: PredictionResponse
    disclaimer: str = "DISCLAIMER: AI decision support prototype. Not for clinical diagnosis."

class ReportResponse(BaseModel):
    report_id: str
    patient_id: Optional[str]
    generated_report: str
    retrieved_references: List[str]
    disclaimer: str

# ==========================================
# Continuous Learning Feedback Schemas
# ==========================================
class FeedbackCreate(BaseModel):
    prediction_id: str
    original_prediction: Dict[str, float]
    doctor_corrected_labels: List[str]  # The actual diseases the doctor identified
    notes: Optional[str] = None

class FeedbackDB(FeedbackCreate):
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(alias="_id")
    reviewed_by: str  # Doctor's username
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)
    is_processed: bool = False  # Flag for the continuous learning script
