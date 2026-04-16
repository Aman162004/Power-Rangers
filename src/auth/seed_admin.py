"""Seed database with initial admin user."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.auth.database import init_db, UserDB, verify_password


DEFAULT_ADMIN_PASSWORD = "changeme123!"
DEFAULT_ANALYST_PASSWORD = "analyst123!"
DEFAULT_OPERATOR_PASSWORD = "operator123!"


def ensure_user(
    username: str,
    email: str,
    password: str,
    full_name: str,
    role: str,
    label: str,
) -> None:
    """Create a demo user if missing, or reset password if it changed."""
    existing_user = UserDB.get_user_by_username(username)
    if existing_user:
        if verify_password(password, existing_user["password_hash"]):
            print(f"✓ {label} user already exists")
            return

        updated = UserDB.update_password(existing_user["id"], password)
        if updated:
            print(f"✓ {label} user already exists")
            print(f"✓ {label} password reset to default for local development")
            print(f"  Username: {username}")
            print(f"  Default password: {password}")
            return

        print(f"✗ {label} exists but password reset failed")
        return

    created_user = UserDB.create_user(
        username=username,
        email=email,
        password=password,
        full_name=full_name,
        role=role,
    )

    if created_user:
        print(f"✓ {label} user created successfully")
        print(f"  Username: {username}")
        print(f"  Email: {email}")
        print(f"  Default password: {password}")
    else:
        print(f"✗ Failed to create {label} user")


def seed_admin():
    """Create initial demo users for local development."""
    init_db()
    ensure_user(
        username="admin",
        email="admin@powerrangers.local",
        password=DEFAULT_ADMIN_PASSWORD,
        full_name="System Administrator",
        role="ADMIN",
        label="Admin",
    )
    ensure_user(
        username="analyst_demo",
        email="analyst@powerrangers.local",
        password=DEFAULT_ANALYST_PASSWORD,
        full_name="Energy Analyst Demo",
        role="ANALYST",
        label="Analyst",
    )
    ensure_user(
        username="operator_demo",
        email="operator@powerrangers.local",
        password=DEFAULT_OPERATOR_PASSWORD,
        full_name="Power Grid Operator Demo",
        role="OPERATOR",
        label="Operator",
    )


if __name__ == "__main__":
    seed_admin()
