"""Report Service - Wraps the existing PDFReportGenerator with database integration."""
import sys
import os
import json
from typing import Dict
from sqlalchemy.orm import Session

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.report_generator import PDFReportGenerator
from backend.app.models.report import Report
from backend.app.services.rag_service import RAGService
from backend.app.services.risk_service import RiskService

class ReportService:
    """Service layer for report generation."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.generator = PDFReportGenerator()
        self.rag_service = RAGService()
        self.risk_service = RiskService()
    
    def generate_report(
        self,
        patient_id: str,
        patient_data: Dict,
        db: Session
    ) -> Dict:
        """
        Generate a PDF report for a patient.
        
        Args:
            patient_id: Patient ID
            patient_data: Patient information dict
            db: Database session
            
        Returns:
            Report metadata including file path
        """
        try:
            # Get latest risk assessment
            risk_history = self.risk_service.get_risk_history(patient_id, db, limit=1)
            if risk_history:
                risk_assessment = risk_history[0]
                # Convert back to the format expected by report generator
                risk_data = {
                    "risk_level": risk_assessment["risk_level"],
                    "warnings": risk_assessment["warnings"],
                    "vitals_analyzed": {}
                }
                
                # Get latest vitals
                latest_vitals = self.risk_service.get_latest_vitals(patient_id, db)
                if latest_vitals:
                    risk_data["vitals_analyzed"] = {
                        "bp_systolic": latest_vitals.get("bp_systolic", 0),
                        "bp_diastolic": latest_vitals.get("bp_diastolic", 0),
                        "glucose": latest_vitals.get("glucose", 0),
                        "heart_rate": latest_vitals.get("heart_rate", 0)
                    }
            else:
                risk_data = {
                    "risk_level": "None Performed",
                    "warnings": [],
                    "vitals_analyzed": {}
                }
            
            # Get conversation history
            conversations = self.rag_service.get_conversation_history(patient_id, db, limit=10)
            conversation_log = [
                (conv["question"], conv["answer"])
                for conv in conversations
            ]
            
            # Generate PDF
            report_path = self.generator.generate_report(
                patient_data,
                risk_data,
                conversation_log
            )
            
            # Save report metadata to database
            report = Report(
                patient_id=patient_id,
                report_path=report_path,
                report_type="pregnancy_assessment",
                report_metadata=json.dumps({
                    "patient_name": patient_data.get("name"),
                    "gestational_week": patient_data.get("week"),
                    "risk_level": risk_data["risk_level"]
                })
            )
            db.add(report)
            db.commit()
            db.refresh(report)
            
            return {
                "report_id": report.report_id,
                "report_path": report_path,
                "report_type": report.report_type,
                "metadata": json.loads(report.report_metadata) if report.report_metadata else {},
                "generated_at": report.generated_at.isoformat()
            }
        except Exception as e:
            db.rollback()
            raise Exception(f"Report generation failed: {str(e)}")
    
    def get_patient_reports(
        self,
        patient_id: str,
        db: Session
    ) -> list:
        """Get all reports for a patient."""
        reports = db.query(Report).filter(
            Report.patient_id == patient_id
        ).order_by(
            Report.generated_at.desc()
        ).all()
        
        return [
            {
                "report_id": report.report_id,
                "report_path": report.report_path,
                "report_type": report.report_type,
                "metadata": json.loads(report.report_metadata) if report.report_metadata else {},
                "generated_at": report.generated_at.isoformat()
            }
            for report in reports
        ]
    
    def get_report_by_id(
        self,
        report_id: str,
        db: Session
    ) -> Dict:
        """Get a specific report."""
        report = db.query(Report).filter(
            Report.report_id == report_id
        ).first()
        
        if report:
            return {
                "report_id": report.report_id,
                "report_path": report.report_path,
                "report_type": report.report_type,
                "metadata": json.loads(report.report_metadata) if report.report_metadata else {},
                "generated_at": report.generated_at.isoformat()
            }
        return None
