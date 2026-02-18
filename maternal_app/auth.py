"""
Authentication utilities: password hashing (SHA256+salt) and JWT token management.
"""
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
import jwt

SECRET_KEY = os.getenv("SECRET_KEY", "maternal-health-secret-key-change-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours


def hash_password(password: str) -> str:
    """Hash password with SHA256 + random salt."""
    salt = secrets.token_hex(16)
    pw_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${pw_hash}"


def verify_password(plain: str, hashed: str) -> bool:
    """Verify password against stored hash."""
    try:
        salt, pw_hash = hashed.split("$", 1)
        return hashlib.sha256((salt + plain).encode()).hexdigest() == pw_hash
    except ValueError:
        return False


def create_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    """Returns user_id or None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None
