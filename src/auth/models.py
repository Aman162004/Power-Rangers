"""Pydantic models for authentication requests and responses."""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class RegisterRequest(BaseModel):
    """User registration request schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    invite_token: str = Field(..., description="Admin-provided invite token")


class UserResponse(BaseModel):
    """User response schema (safe, no password)."""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: int
    created_at: str
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    refresh_token: str


class InviteTokenRequest(BaseModel):
    """Create invite token request (admin only)."""
    expires_in_hours: int = Field(default=48, ge=1, le=720)


class InviteTokenResponse(BaseModel):
    """Invite token response."""
    token: str
    expires_at: str
    created_at: str


class UpdateUserRoleRequest(BaseModel):
    """Update user role request (admin only)."""
    role: str = Field(..., pattern="^(OPERATOR|ANALYST|ADMIN)$")


class LogoutRequest(BaseModel):
    """Logout request schema."""
    refresh_token: str
