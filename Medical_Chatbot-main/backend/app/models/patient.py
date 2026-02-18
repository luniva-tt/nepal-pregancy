import uuid
from sqlalchemy import Column, String, Integer, Date, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from backend.app.database import Base

class Patient(Base):
    """Patient model for storing patient medical information."""
    __tablename__ = "patients"

    patient_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    age = Column(Integer, nullable=True)
    blood_group = Column(String(10), nullable=True)
    gestational_week = Column(Integer, nullable=True)
    due_date = Column(Date, nullable=True)
    medical_history = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<Patient(name='{self.name}', gestational_week={self.gestational_week})>"
