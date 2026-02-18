"""Authentication API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime

from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.utils.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token
)

router = APIRouter()

# Pydantic schemas
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    age: int = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    email: str
    name: str

class UserResponse(BaseModel):
    user_id: str
    email: str
    name: str
    age: int = None
    created_at: datetime

@router.post("/register", response_model=Token)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            detail="Email already registered"
        )
    
    try:
        print(f"DEBUG: Registering user {user_data.email}")
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            email=user_data.email,
            password_hash=hashed_password,
            name=user_data.name,
            age=user_data.age
        )
        print("DEBUG: User object created")
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create access token
        access_token = create_access_token(data={"sub": new_user.user_id})
        
        print(f"DEBUG: User {new_user.email} registered successfully with ID {new_user.user_id}")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": new_user.user_id,
            "email": new_user.email,
            "name": new_user.name
        }
    except Exception as e:
        db.rollback()
        print(f"ERROR: Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login user and return JWT token."""
    user = db.query(User).filter(User.email == credentials.email).first()
    
    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create access token
    access_token = create_access_token(data={"sub": user.user_id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name
    }

# Dependency to get current user from token
def get_current_user(token: str, db: Session = Depends(get_db)) -> User:
    """Get current user from JWT token."""
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user_id = payload.get("sub")
    user = db.query(User).filter(User.user_id == user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

@router.get("/me", response_model=UserResponse)
def get_me(token: str, db: Session = Depends(get_db)):
    """Get current user information."""
    user = get_current_user(token, db)
    return user
