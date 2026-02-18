"""RAG Service - Wraps the existing PregnancyRAG class with database integration."""
import sys
import os
import json
from typing import Dict, List
from sqlalchemy.orm import Session

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.rag_pipeline import PregnancyRAG
from backend.app.models.conversation import Conversation

class RAGService:
    """Service layer for RAG operations with database persistence."""
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        try:
            self.rag = PregnancyRAG()
        except Exception as e:
            print(f"Warning: RAG initialization failed: {e}")
            self.rag = None
    
    def ask_question(
        self,
        question: str,
        patient_id: str,
        session_id: str,
        db: Session
    ) -> Dict:
        """
        Ask a question using RAG and save to database.
        
        Args:
            question: User's question
            patient_id: Patient ID
            session_id: Session identifier
            db: Database session
            
        Returns:
            Dictionary with answer and sources
        """
        if not self.rag:
            raise ValueError("RAG system not initialized. Please run ingestion first.")
        
        # Get response from RAG
        try:
            response = self.rag.ask(question)
            answer = response["answer"]
            source_docs = response["source_docs"]
            
            # Format sources for database storage
            sources = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "content": doc.page_content[:200]
                }
                for doc in source_docs
            ]
            
            # Save to database
            conversation = Conversation(
                patient_id=patient_id,
                question=question,
                answer=answer,
                sources=json.dumps(sources),
                session_id=session_id
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            
            return {
                "conversation_id": conversation.conversation_id,
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            db.rollback()
            raise Exception(f"RAG query failed: {str(e)}")
    
    def get_conversation_history(
        self,
        patient_id: str,
        db: Session,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get conversation history for a patient.
        
        Args:
            patient_id: Patient ID
            db: Database session
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation dictionaries
        """
        conversations = db.query(Conversation).filter(
            Conversation.patient_id == patient_id
        ).order_by(
            Conversation.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "conversation_id": conv.conversation_id,
                "question": conv.question,
                "answer": conv.answer,
                "sources": json.loads(conv.sources) if conv.sources else [],
                "created_at": conv.created_at.isoformat()
            }
            for conv in conversations
        ]
    
    def delete_conversation(
        self,
        conversation_id: str,
        db: Session
    ) -> bool:
        """Delete a conversation."""
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
        
        if conversation:
            db.delete(conversation)
            db.commit()
            return True
        return False
