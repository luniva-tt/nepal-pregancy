"""Patients API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime
import json

from backend.app.database import get_db
from backend.app.models.patient import Patient
from backend.app.api.auth import get_current_user
from backend.app.models.user import User

router = APIRouter()

# Pydantic schemas
class PatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    blood_group: Optional[str] = None
    gestational_week: Optional[int] = None
    due_date: Optional[date] = None
    medical_history: Optional[dict] = None

class PatientUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    blood_group: Optional[str] = None
    gestational_week: Optional[int] = None
    due_date: Optional[date] = None
    medical_history: Optional[dict] = None

class PatientResponse(BaseModel):
    patient_id: str
    user_id: str
    name: str
    age: Optional[int] = None
    blood_group: Optional[str] = None
    gestational_week: Optional[int] = None
    due_date: Optional[date] = None
    medical_history: Optional[dict] = None
    created_at: datetime
    is_active: bool

@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(
    patient_data: PatientCreate,
    user_token: str,
    db: Session = Depends(get_db)
):
    """Create a new patient profile."""
    user = get_current_user(user_token, db)
    
    patient = Patient(
        user_id=user.user_id,
        name=patient_data.name,
        age=patient_data.age,
        blood_group=patient_data.blood_group,
        gestational_week=patient_data.gestational_week,
        due_date=patient_data.due_date,
        medical_history=json.dumps(patient_data.medical_history) if patient_data.medical_history else None
    )
    
    db.add(patient)
    db.commit()
    db.refresh(patient)
    
    return patient

@router.get("", response_model=List[PatientResponse])
def get_patients(user_token: str, db: Session = Depends(get_db)):
    """Get all patients for the current user."""
    user = get_current_user(user_token, db)
    
    patients = db.query(Patient).filter(
        Patient.user_id == user.user_id,
        Patient.is_active == True
    ).all()
    
    return patients

@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str, user_token: str, db: Session = Depends(get_db)):
    """Get a specific patient."""
    user = get_current_user(user_token, db)
    
    patient = db.query(Patient).filter(
        Patient.patient_id == patient_id,
        Patient.user_id == user.user_id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    return patient

@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: str,
    patient_data: PatientUpdate,
    user_token: str,
    db: Session = Depends(get_db)
):
    """Update a patient profile."""
    user = get_current_user(user_token, db)
    
    patient = db.query(Patient).filter(
        Patient.patient_id == patient_id,
        Patient.user_id == user.user_id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Update fields
    if patient_data.name:
        patient.name = patient_data.name
    if patient_data.age:
        patient.age = patient_data.age
    if patient_data.blood_group:
        patient.blood_group = patient_data.blood_group
    if patient_data.gestational_week:
        patient.gestational_week = patient_data.gestational_week
    if patient_data.due_date:
        patient.due_date = patient_data.due_date
    if patient_data.medical_history:
        patient.medical_history = json.dumps(patient_data.medical_history)
    
    db.commit()
    db.refresh(patient)
    
    return patient

@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(patient_id: str, user_token: str, db: Session = Depends(get_db)):
    """Soft delete a patient (sets is_active to False)."""
    user = get_current_user(user_token, db)
    
    patient = db.query(Patient).filter(
        Patient.patient_id == patient_id,
        Patient.user_id == user.user_id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    patient.is_active = False
    db.commit()
    
    return None
