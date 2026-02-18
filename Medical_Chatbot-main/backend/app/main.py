"""
FastAPI main application - Pregnancy Health RAG System Web API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.app.database import init_db

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")
    yield
    print("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Pregnancy Health RAG API",
    description="Production-ready API for Pregnancy Health Monitoring with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Pregnancy Health RAG API"}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Pregnancy Health RAG API",
        "docs": "/docs",
        "health": "/health"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    print(f"ERROR: Unhandled exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": traceback.format_exc()}
    )

from fastapi.responses import JSONResponse

from backend.app.api import auth, patients, chat, vitals, reports
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(patients.router, prefix="/api/patients", tags=["Patients"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat/RAG"])
app.include_router(vitals.router, prefix="/api/vitals", tags=["Vitals"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
