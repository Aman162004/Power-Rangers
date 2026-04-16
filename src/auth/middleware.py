"""Authentication middleware and dependencies for FastAPI."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import Optional
from src.auth.jwt_handler import verify_token, extract_user_id_from_token
from src.auth.database import UserDB

security = HTTPBearer(auto_error=False)


def get_current_user(credentials=Depends(security)):
    """Get current authenticated user from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = int(payload.get("sub"))
    user = UserDB.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


def get_current_user_optional(credentials=Depends(security)):
    """Get current user if authenticated, otherwise None."""
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        return None
    
    user_id = int(payload.get("sub"))
    user = UserDB.get_user_by_id(user_id)
    
    return user


def require_admin(user: dict = Depends(get_current_user)):
    """Require user to be admin."""
    if user.get("role") != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def require_analyst_or_admin(user: dict = Depends(get_current_user)):
    """Require user to be analyst or admin."""
    if user.get("role") not in ["ANALYST", "ADMIN"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analyst or Admin access required",
        )
    return user
