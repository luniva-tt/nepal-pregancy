"""Vitals and Risk Assessment API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from backend.app.database import get_db
from backend.app.services.risk_service import RiskService
from backend.app.api.auth import get_current_user
from backend.app.models.patient import Patient

router = APIRouter()
risk_service = RiskService()

# Pydantic schemas
class VitalsInput(BaseModel):
    patient_id: str
    bp_systolic: int = None
    bp_diastolic: int = None
    heart_rate: int = None
    glucose: float = None
    hemoglobin: float = None
    weight: float = None

class RiskAssessmentResponse(BaseModel):
    assessment_id: str
    vital_id: str
    risk_level: str
    warnings: List[str]
    recommendations: List[str]
    clinical_interpretation: str
    vitals_analyzed: dict

class VitalsHistoryResponse(BaseModel):
    vital_id: str
    bp_systolic: int = None
    bp_diastolic: int = None
    heart_rate: int = None
    glucose: float = None
    hemoglobin: float = None
    weight: float = None
    recorded_at: str

@router.post("", response_model=RiskAssessmentResponse, status_code=status.HTTP_201_CREATED)
def record_vitals_and_assess(
    vitals_input: VitalsInput,
    user_token: str,
    db: Session = Depends(get_db)
):
    """Record vitals and perform risk assessment."""
    user = get_current_user(user_token, db)
    
    # Verify patient belongs to user
    patient = db.query(Patient).filter(
        Patient.patient_id == vitals_input.patient_id,
        Patient.user_id == user.user_id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Prepare vitals data
    vitals_data = {
        "bp_systolic": vitals_input.bp_systolic,
        "bp_diastolic": vitals_input.bp_diastolic,
        "heart_rate": vitals_input.heart_rate,
        "glucose": vitals_input.glucose,
        "hemoglobin": vitals_input.hemoglobin,
        "weight": vitals_input.weight
    }
    
    try:
        assessment = risk_service.assess_risk(
            vitals_data=vitals_data,
            patient_id=vitals_input.patient_id,
            db=db
        )
        return assessment
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{patient_id}", response_model=List[VitalsHistoryResponse])
def get_vitals_history(
    patient_id: str,
    user_token: str,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get vitals history for a patient."""
    user = get_current_user(user_token, db)
    
    # Verify patient belongs to user
    patient = db.query(Patient).filter(
        Patient.patient_id == patient_id,
        Patient.user_id == user.user_id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Get vitals from database
    from backend.app.models.vitals import Vitals
    vitals_list = db.query(Vitals).filter(
        Vitals.patient_id == patient_id
    ).order_by(
        Vitals.recorded_at.desc()
    ).limit(limit).all()
    
    return [
        {
            "vital_id": v.vital_id,
            "bp_systolic": v.bp_systolic,
            "bp_diastolic": v.bp_diastolic,
            "heart_rate": v.heart_rate,
            "glucose": v.glucose,
            "hemoglobin": v.hemoglobin,
            "weight": v.weight,
            "recorded_at": v.recorded_at.isoformat()
        }
        for v in vitals_list
    ]

@router.get("/{patient_id}/latest", response_model=VitalsHistoryResponse)
def get_latest_vitals(
    patient_id: str,
    user_token: str,
    db: Session = Depends(get_db)
):
    """Get the latest vitals for a patient."""
    user = get_current_user(user_token, db)
    
    # Verify patient belongs to user
    patient = db.query(Patient).filter(
        Patient.patient_id == patient_id,
        Patient.user_id == user.user_id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    vitals = risk_service.get_latest_vitals(patient_id, db)
    if not vitals:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No vitals found for this patient"
        )
    
    return vitals
