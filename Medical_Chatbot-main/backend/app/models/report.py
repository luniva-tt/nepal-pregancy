import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from backend.app.database import Base

class Report(Base):
    """Report model for generated PDF reports metadata."""
    __tablename__ = "reports"

    report_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False, index=True)
    report_path = Column(String(500), nullable=False)
    report_type = Column(String(100), default="pregnancy_assessment")
    report_metadata = Column(Text, nullable=True)  # JSON string
    generated_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    def __repr__(self):
        return f"<Report(patient_id='{self.patient_id}', type='{self.report_type}')>"
