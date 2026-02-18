import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.sql import func
from backend.app.database import Base

class RiskAssessment(Base):
    """Risk assessment model for storing evaluation results."""
    __tablename__ = "risk_assessments"

    assessment_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False, index=True)
    vital_id = Column(String(36), ForeignKey("vitals.vital_id", ondelete="SET NULL"), nullable=True)
    risk_level = Column(String(50), nullable=False, index=True)
    warnings = Column(Text, nullable=True)  # JSON string
    recommendations = Column(Text, nullable=True)  # JSON string
    clinical_interpretation = Column(Text, nullable=True)
    assessed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        CheckConstraint("risk_level IN ('Low', 'Medium', 'High')", name="check_risk_level"),
    )

    def __repr__(self):
        return f"<RiskAssessment(patient_id='{self.patient_id}', risk_level='{self.risk_level}')>"
