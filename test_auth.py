#!/usr/bin/env python3
"""Test authentication system."""

import sys
import secrets
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.auth.database import RefreshTokenDB, UserDB, init_db
from src.auth.jwt_handler import create_access_token, create_refresh_token, verify_token
from src.auth.middleware import get_current_user

# Test 1: Database initialization
init_db()
print("✓ Database initialized")

# Test 2: Admin user exists
admin = UserDB.get_user_by_username("admin")
print(f"✓ Admin user exists: {admin['username']}")

# Test 3: Authentication
test = UserDB.authenticate("admin", "changeme123!")
print(f"✓ Auth works: {test['username']}")

# Test 4: JWT token creation
token, expires = create_access_token({"sub": str(admin["id"])}, session_version=UserDB.get_user_session_version(admin["id"]) or 0)
print(f"✓ JWT token created: {token[:20]}...")

# Test 5: JWT token verification
payload = verify_token(token)
print(f"✓ JWT verified: user_id={payload['sub']}")


class Credentials:
	def __init__(self, credentials: str):
		self.credentials = credentials


# Test 6: Session invalidation on re-login
temp_username = f"session_test_{secrets.token_hex(4)}"
temp_email = f"{temp_username}@example.com"
temp_password = "TestPass123!"
temp_user = UserDB.create_user(temp_username, temp_email, temp_password, "Session Test User")
if not temp_user:
	raise RuntimeError("Failed to create temporary auth test user")

try:
	initial_version = UserDB.get_user_session_version(temp_user["id"]) or 0
	old_access_token, _ = create_access_token({"sub": str(temp_user["id"])}, session_version=initial_version)
	old_refresh_token = create_refresh_token(temp_user["id"], session_version=initial_version)
	RefreshTokenDB.create_refresh_token(
		temp_user["id"],
		datetime.utcnow() + timedelta(days=7),
		token=old_refresh_token,
	)

	assert RefreshTokenDB.verify_refresh_token(temp_user["id"], old_refresh_token) is True

	new_version = UserDB.increment_session_version(temp_user["id"])
	if new_version is None:
		raise RuntimeError("Failed to increment session version")

	RefreshTokenDB.revoke_all_refresh_tokens_for_user(temp_user["id"])
	new_access_token, _ = create_access_token({"sub": str(temp_user["id"])}, session_version=new_version)

	try:
		get_current_user(Credentials(old_access_token))
		raise AssertionError("Old access token should have been rejected after re-login")
	except Exception as exc:
		if getattr(exc, "status_code", None) != 401:
			raise

	new_user = get_current_user(Credentials(new_access_token))
	assert new_user["id"] == temp_user["id"]

	assert RefreshTokenDB.verify_refresh_token(temp_user["id"], old_refresh_token) is False
	print("✓ Re-login invalidates previous session")
finally:
	UserDB.deactivate_user(temp_user["id"])

print("\n✅ All authentication tests passed!")
