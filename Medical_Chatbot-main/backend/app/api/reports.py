"""Reports API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import os

from backend.app.database import get_db
from backend.app.services.report_service import ReportService
from backend.app.api.auth import get_current_user
from backend.app.models.patient import Patient

router = APIRouter()
report_service = ReportService()

# Pydantic schemas
class ReportGenerateRequest(BaseModel):
    patient_id: str

class ReportResponse(BaseModel):
    report_id: str
    report_path: str
    report_type: str
    metadata: dict
    generated_at: str

@router.post("/generate", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
def generate_report(
    request: ReportGenerateRequest,
    user_token: str,
    db: Session = Depends(get_db)
):
    """Generate a PDF report for a patient."""
    user = get_current_user(user_token, db)
    
    # Verify patient belongs to user
    patient = db.query(Patient).filter(
        Patient.patient_id == request.patient_id,
        Patient.user_id == user.user_id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Prepare patient data
    patient_data = {
        "name": patient.name,
        "age": patient.age,
        "week": patient.gestational_week
    }
    
    try:
        report = report_service.generate_report(
            patient_id=request.patient_id,
            patient_data=patient_data,
            db=db
        )
        return report
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{patient_id}", response_model=List[ReportResponse])
def get_patient_reports(
    patient_id: str,
    user_token: str,
    db: Session = Depends(get_db)
):
    """Get all reports for a patient."""
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
    
    reports = report_service.get_patient_reports(patient_id, db)
    return reports

@router.get("/download/{report_id}")
def download_report(
    report_id: str,
    user_token: str,
    db: Session = Depends(get_db)
):
    """Download a PDF report."""
    user = get_current_user(user_token, db)
    
    report = report_service.get_report_by_id(report_id, db)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Check file exists
    if not os.path.exists(report["report_path"]):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found"
        )
    
    return FileResponse(
        path=report["report_path"],
        filename=os.path.basename(report["report_path"]),
        media_type="application/pdf"
    )
