"""JWT token creation and validation."""

import jwt
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local-development dependency
    def load_dotenv() -> None:
        return None

load_dotenv()

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_HOURS", "2"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> tuple[str, datetime]:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt, expire


def create_refresh_token(user_id: int) -> str:
    """Create refresh token."""
    expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    expire = datetime.utcnow() + expires_delta
    
    data = {
        "sub": str(user_id),
        "type": "refresh",
        "exp": expire
    }
    
    encoded_jwt = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode token without verification (for getting user_id from expired tokens)."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        return payload
    except jwt.InvalidTokenError:
        return None


def extract_user_id_from_token(token: str) -> Optional[int]:
    """Extract user_id from token payload."""
    payload = verify_token(token)
    if payload:
        return int(payload.get("sub"))
    return None
