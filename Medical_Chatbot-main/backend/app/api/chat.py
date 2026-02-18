"""Chat/RAG API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uuid

from backend.app.database import get_db
from backend.app.services.rag_service import RAGService
from backend.app.api.auth import get_current_user
from backend.app.models.patient import Patient

router = APIRouter()
rag_service = RAGService()

# Pydantic schemas
class ChatRequest(BaseModel):
    patient_id: str
    question: str
    session_id: str = None

class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: List[dict]

class ConversationHistory(BaseModel):
    conversation_id: str
    question: str
    answer: str
    sources: List[dict]
    created_at: str

@router.post("/ask", response_model=ChatResponse)
def ask_question(
    chat_request: ChatRequest,
    user_token: str,
    db: Session = Depends(get_db)
):
    """Ask a question using RAG."""
    user = get_current_user(user_token, db)
    
    # Verify patient belongs to user
    patient = db.query(Patient).filter(
        Patient.patient_id == chat_request.patient_id,
        Patient.user_id == user.user_id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Generate session ID if not provided
    session_id = chat_request.session_id or str(uuid.uuid4())
    
    try:
        response = rag_service.ask_question(
            question=chat_request.question,
            patient_id=chat_request.patient_id,
            session_id=session_id,
            db=db
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/history/{patient_id}", response_model=List[ConversationHistory])
def get_conversation_history(
    patient_id: str,
    user_token: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get conversation history for a patient."""
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
    
    conversations = rag_service.get_conversation_history(patient_id, db, limit)
    return conversations

@router.delete("/history/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    user_token: str,
    db: Session = Depends(get_db)
):
    """Delete a conversation."""
    user = get_current_user(user_token, db)
    
    success = rag_service.delete_conversation(conversation_id, db)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return None
