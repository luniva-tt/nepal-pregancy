"""Risk Service - Wraps the existing RiskEvaluator class with database integration."""
import sys
import os
import json
from typing import Dict
from sqlalchemy.orm import Session
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.risk_engine import RiskEvaluator
from backend.app.models.risk_assessment import RiskAssessment
from backend.app.models.vitals import Vitals

class RiskService:
    """Service layer for risk assessment operations."""
    
    def __init__(self):
        """Initialize the risk evaluator."""
        self.evaluator = RiskEvaluator()
    
    def assess_risk(
        self,
        vitals_data: Dict,
        patient_id: str,
        db: Session
    ) -> Dict:
        """
        Assess risk based on vitals and save to database.
        
        Args:
            vitals_data: Dictionary with bp_systolic, bp_diastolic, glucose, heart_rate
            patient_id: Patient ID
            db: Database session
            
        Returns:
            Risk assessment results
        """
        try:
            # Save vitals to database first
            vitals = Vitals(
                patient_id=patient_id,
                bp_systolic=vitals_data.get("bp_systolic"),
                bp_diastolic=vitals_data.get("bp_diastolic"),
                heart_rate=vitals_data.get("heart_rate"),
                glucose=vitals_data.get("glucose"),
                hemoglobin=vitals_data.get("hemoglobin"),
                weight=vitals_data.get("weight")
            )
            db.add(vitals)
            db.commit()
            db.refresh(vitals)
            
            # Perform risk assessment
            assessment_result = self.evaluator.assess_risk(vitals_data)
            
            # Determine clinical interpretation
            risk_level = assessment_result["risk_level"]
            interpretation = "Vitals are within normal limits."
            if risk_level == "High":
                interpretation = "Critical values detected. Immediate attention required."
            elif risk_level == "Medium":
                interpretation = "Abnormal values detected. Monitoring recommended."
            
            # Determine recommendations
            recommendations = []
            if risk_level == "High":
                recommendations = [
                    "Consult Obstetrician immediately",
                    "Daily BP/Glucose monitoring",
                    "Referral Required: Yes - Urgent"
                ]
            elif risk_level == "Medium":
                recommendations = [
                    "Schedule follow-up within 1 week",
                    "Weekly monitoring"
                ]
            else:
                recommendations = [
                    "Continue regular antenatal visits"
                ]
            
            # Save risk assessment
            risk_assessment = RiskAssessment(
                patient_id=patient_id,
                vital_id=vitals.vital_id,
                risk_level=risk_level,
                warnings=json.dumps(assessment_result["warnings"]),
                recommendations=json.dumps(recommendations),
                clinical_interpretation=interpretation
            )
            db.add(risk_assessment)
            db.commit()
            db.refresh(risk_assessment)
            
            return {
                "assessment_id": risk_assessment.assessment_id,
                "vital_id": vitals.vital_id,
                "risk_level": risk_level,
                "warnings": assessment_result["warnings"],
                "recommendations": recommendations,
                "clinical_interpretation": interpretation,
                "vitals_analyzed": assessment_result["vitals_analyzed"]
            }
        except Exception as e:
            db.rollback()
            raise Exception(f"Risk assessment failed: {str(e)}")
    
    def get_risk_history(
        self,
        patient_id: str,
        db: Session,
        limit: int = 20
    ) -> list:
        """Get risk assessment history for a patient."""
        assessments = db.query(RiskAssessment).filter(
            RiskAssessment.patient_id == patient_id
        ).order_by(
            RiskAssessment.assessed_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "assessment_id": assessment.assessment_id,
                "risk_level": assessment.risk_level,
                "warnings": json.loads(assessment.warnings) if assessment.warnings else [],
                "recommendations": json.loads(assessment.recommendations) if assessment.recommendations else [],
                "clinical_interpretation": assessment.clinical_interpretation,
                "assessed_at": assessment.assessed_at.isoformat()
            }
            for assessment in assessments
        ]
    
    def get_latest_vitals(
        self,
        patient_id: str,
        db: Session
    ) -> Dict:
        """Get the latest vitals for a patient."""
        vital = db.query(Vitals).filter(
            Vitals.patient_id == patient_id
        ).order_by(
            Vitals.recorded_at.desc()
        ).first()
        
        if vital:
            return {
                "vital_id": vital.vital_id,
                "bp_systolic": vital.bp_systolic,
                "bp_diastolic": vital.bp_diastolic,
                "heart_rate": vital.heart_rate,
                "glucose": vital.glucose,
                "hemoglobin": vital.hemoglobin,
                "weight": vital.weight,
                "recorded_at": vital.recorded_at.isoformat()
            }
        return None
