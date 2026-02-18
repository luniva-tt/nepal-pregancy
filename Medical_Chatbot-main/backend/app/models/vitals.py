import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from backend.app.database import Base

class Vitals(Base):
    """Vitals model for patient vital signs measurements."""
    __tablename__ = "vitals"

    vital_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False, index=True)
    bp_systolic = Column(Integer, nullable=True)
    bp_diastolic = Column(Integer, nullable=True)
    heart_rate = Column(Integer, nullable=True)
    glucose = Column(Float, nullable=True)
    hemoglobin = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    def __repr__(self):
        return f"<Vitals(patient_id='{self.patient_id}', bp={self.bp_systolic}/{self.bp_diastolic})>"
