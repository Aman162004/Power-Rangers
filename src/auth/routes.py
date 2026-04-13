"""Authentication API routes."""

from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timedelta
from src.auth.models import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
    RefreshTokenRequest,
    InviteTokenRequest,
    InviteTokenResponse,
    UpdateUserRoleRequest,
    LogoutRequest,
)
from src.auth.database import UserDB, InviteTokenDB, RefreshTokenDB, hash_password, init_db
from src.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    verify_token,
)
from src.auth.middleware import get_current_user, require_admin

router = APIRouter(prefix="/api/auth", tags=["authentication"])

# Initialize database on module import
init_db()


def format_user_response(user: dict) -> UserResponse:
    """Convert database user dict to response model."""
    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        is_active=user["is_active"],
        created_at=user["created_at"],
    )


@router.post("/register", response_model=TokenResponse)
def register(request: RegisterRequest):
    """Register a new user with an invite token."""
    # Validate invite token
    invite = InviteTokenDB.get_invite_token(request.invite_token)
    if not invite:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired invite token",
        )
    
    # Create user
    user = UserDB.create_user(
        username=request.username,
        email=request.email,
        password=request.password,
        full_name=request.full_name or "",
        role="OPERATOR",  # Default role, can be changed by admin
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists",
        )
    
    # Mark invite token as used
    InviteTokenDB.mark_token_used(request.invite_token, user["id"])
    
    # Create tokens
    access_token, expires_at = create_access_token({"sub": str(user["id"])})
    refresh_token = create_refresh_token(user["id"])
    
    # Store refresh token hash
    expires_in_days = 7
    RefreshTokenDB.create_refresh_token(
        user["id"],
        datetime.utcnow() + timedelta(days=expires_in_days),
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=2 * 3600,  # 2 hours in seconds
        user=format_user_response(user),
    )


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    """Authenticate user and return JWT tokens."""
    user = UserDB.authenticate(request.username, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    
    # Create tokens
    access_token, expires_at = create_access_token({"sub": str(user["id"])})
    refresh_token = create_refresh_token(user["id"])
    
    # Store refresh token hash
    expires_in_days = 7
    RefreshTokenDB.create_refresh_token(
        user["id"],
        datetime.utcnow() + timedelta(days=expires_in_days),
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=2 * 3600,  # 2 hours in seconds
        user=format_user_response(user),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(request: RefreshTokenRequest):
    """Refresh access token using refresh token."""
    payload = verify_token(request.refresh_token)
    
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    user_id = int(payload.get("sub"))
    
    # Verify refresh token in database
    if not RefreshTokenDB.verify_refresh_token(user_id, request.refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked or expired",
        )
    
    user = UserDB.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    # Create new access token
    access_token, expires_at = create_access_token({"sub": str(user["id"])})
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=request.refresh_token,  # Return same refresh token
        expires_in=2 * 3600,
        user=format_user_response(user),
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_info(user: dict = Depends(get_current_user)):
    """Get current logged-in user information."""
    return format_user_response(user)


@router.post("/logout")
def logout(request: LogoutRequest, user: dict = Depends(get_current_user)):
    """Logout user (revoke refresh token)."""
    user_id = user["id"]
    RefreshTokenDB.revoke_refresh_token(user_id, request.refresh_token)
    
    return {"message": "Successfully logged out"}


# ============================================================================
# Admin-only endpoints
# ============================================================================


@router.post("/admin/invite", response_model=InviteTokenResponse)
def create_invite_token(
    request: InviteTokenRequest,
    admin_user: dict = Depends(require_admin),
):
    """Create an invite token for new users (admin only)."""
    expires_at = datetime.utcnow() + timedelta(hours=request.expires_in_hours)
    token = InviteTokenDB.create_invite_token(admin_user["id"], expires_at)
    
    return InviteTokenResponse(
        token=token,
        expires_at=expires_at.isoformat(),
        created_at=datetime.utcnow().isoformat(),
    )


@router.get("/admin/users", response_model=list[UserResponse])
def list_all_users(admin_user: dict = Depends(require_admin)):
    """Get list of all users (admin only)."""
    users = UserDB.get_all_users()
    return [format_user_response(user) for user in users]


@router.put("/admin/users/{user_id}/role", response_model=UserResponse)
def update_user_role(
    user_id: int,
    request: UpdateUserRoleRequest,
    admin_user: dict = Depends(require_admin),
):
    """Update user role (admin only)."""
    user = UserDB.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Prevent admin from removing their own admin role via this endpoint
    if user_id == admin_user["id"] and request.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove your own admin role",
        )
    
    success = UserDB.update_user_role(user_id, request.role)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update role",
        )
    
    updated_user = UserDB.get_user_by_id(user_id)
    return format_user_response(updated_user)


@router.delete("/admin/users/{user_id}")
def deactivate_user(
    user_id: int,
    admin_user: dict = Depends(require_admin),
):
    """Deactivate a user (soft delete, admin only)."""
    user = UserDB.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Prevent admin from deactivating themselves
    if user_id == admin_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account",
        )
    
    success = UserDB.deactivate_user(user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate user",
        )
    
    return {"message": f"User {user_id} has been deactivated"}
