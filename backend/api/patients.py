# backend/api/patients.py
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from bson import ObjectId

from database.mongodb import db
from database.schemas import PatientCreate, PatientDB
from api.auth import get_current_user

router = APIRouter(prefix="/patients", tags=["Patients"])

@router.post("/", response_model=PatientDB, status_code=status.HTTP_201_CREATED)
async def create_patient(patient: PatientCreate, current_user: dict = Depends(get_current_user)):
    """Register a new patient in the system."""
    patient_dict = patient.model_dump()
    result = await db.db.patients.insert_one(patient_dict)
    patient_dict["_id"] = str(result.inserted_id)
    return PatientDB(**patient_dict)

@router.get("/", response_model=List[PatientDB])
async def list_patients(skip: int = 0, limit: int = 20, current_user: dict = Depends(get_current_user)):
    """List all patients with pagination."""
    patients = await db.db.patients.find().skip(skip).limit(limit).to_list(limit)
    for p in patients:
        p["_id"] = str(p["_id"])
    return [PatientDB(**p) for p in patients]

@router.get("/{patient_id}", response_model=PatientDB)
async def get_patient(patient_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific patient by ID."""
    if not ObjectId.is_valid(patient_id):
        raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
    patient = await db.db.patients.find_one({"_id": ObjectId(patient_id)})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    patient["_id"] = str(patient["_id"])
    return PatientDB(**patient)