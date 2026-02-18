"""
Database models and setup for the Maternal Health Webapp.
Uses SQLite via SQLAlchemy.
"""
import os
import uuid
from datetime import datetime, date
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, Date, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'maternal_health.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class User(Base):
    __tablename__ = "users"
    user_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    lmp_date = Column(Date, nullable=True)          # Last Menstrual Period
    due_date = Column(Date, nullable=True)
    emergency_contact = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def pregnancy_week(self):
        """Calculate current pregnancy week from LMP."""
        if self.lmp_date:
            today = date.today()
            delta = today - self.lmp_date
            return max(1, min(42, delta.days // 7))
        return None


class SymptomLog(Base):
    __tablename__ = "symptom_logs"
    log_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    symptom = Column(String(255), nullable=False)
    severity = Column(String(50), nullable=True)     # mild, moderate, severe
    notes = Column(Text, nullable=True)
    logged_at = Column(DateTime(timezone=True), server_default=func.now())


class RiskHistory(Base):
    __tablename__ = "risk_history"
    record_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    bp_systolic = Column(Integer)
    bp_diastolic = Column(Integer)
    heart_rate = Column(Integer)
    glucose = Column(Integer)
    risk_level = Column(String(50))
    warnings = Column(Text)       # JSON string
    assessed_at = Column(DateTime(timezone=True), server_default=func.now())


class ChatHistory(Base):
    __tablename__ = "chat_history"
    chat_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def init_db():
    Base.metadata.create_all(bind=engine)
