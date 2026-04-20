#!/usr/bin/env python3
"""Test authentication system."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.auth.database import UserDB, init_db
from src.auth.jwt_handler import create_access_token, verify_token

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
token, expires = create_access_token({"sub": str(admin["id"])})
print(f"✓ JWT token created: {token[:20]}...")

# Test 5: JWT token verification
payload = verify_token(token)
print(f"✓ JWT verified: user_id={payload['sub']}")

print("\n✅ All authentication tests passed!")
