"""SQLite database setup and models for authentication."""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = os.getenv("AUTH_DB_PATH", str(PROJECT_ROOT / "data" / "auth.db"))


def get_db_connection():
    """Get SQLite connection with row factory."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database with required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL DEFAULT 'OPERATOR',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create roles table (reference table)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT
        )
    """)
    
    # Create invite tokens table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS invite_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            created_by INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            used_by INTEGER,
            used_at TIMESTAMP,
            FOREIGN KEY (created_by) REFERENCES users(id),
            FOREIGN KEY (used_by) REFERENCES users(id)
        )
    """)
    
    # Create refresh tokens table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            is_revoked INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Insert default roles if they don't exist
    cursor.execute("SELECT COUNT(*) FROM roles")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO roles (name, description) VALUES
            ('OPERATOR', 'Power Grid Operator - view forecasts and basic scenarios'),
            ('ANALYST', 'Energy Analyst - advanced reports and scenario modeling'),
            ('ADMIN', 'System Administrator - full access and user management')
        """)
    
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """Hash password using SHA256 (can upgrade to bcrypt later)."""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    return hash_password(password) == password_hash


class UserDB:
    """Database operations for users."""
    
    @staticmethod
    def create_user(username: str, email: str, password: str, full_name: str = "", role: str = "OPERATOR") -> Optional[dict]:
        """Create a new user."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            password_hash = hash_password(password)
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, full_name, role)
                VALUES (?, ?, ?, ?, ?)
            """, (username, email, password_hash, full_name, role))
            conn.commit()
            user_id = cursor.lastrowid
            
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = dict(cursor.fetchone())
            conn.close()
            return user
        except sqlite3.IntegrityError:
            conn.close()
            return None
    
    @staticmethod
    def get_user_by_username(username: str) -> Optional[dict]:
        """Get user by username."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ? AND is_active = 1", (username,))
        user = cursor.fetchone()
        conn.close()
        
        return dict(user) if user else None
    
    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[dict]:
        """Get user by ID."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE id = ? AND is_active = 1", (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        return dict(user) if user else None
    
    @staticmethod
    def get_user_by_email(email: str) -> Optional[dict]:
        """Get user by email."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (email,))
        user = cursor.fetchone()
        conn.close()
        
        return dict(user) if user else None
    
    @staticmethod
    def get_all_users() -> List[dict]:
        """Get all active users."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, username, email, full_name, role, is_active, created_at FROM users WHERE is_active = 1 ORDER BY created_at DESC")
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return users
    
    @staticmethod
    def update_user_role(user_id: int, role: str) -> bool:
        """Update user role."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users SET role = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        """, (role, user_id))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    
    @staticmethod
    def deactivate_user(user_id: int) -> bool:
        """Deactivate a user (soft delete)."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users SET is_active = 0, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        """, (user_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[dict]:
        """Authenticate user with username and password."""
        user = UserDB.get_user_by_username(username)
        if user and verify_password(password, user['password_hash']):
            return user
        return None

    @staticmethod
    def update_password(user_id: int, new_password: str) -> bool:
        """Update user password hash."""
        conn = get_db_connection()
        cursor = conn.cursor()

        new_hash = hash_password(new_password)
        cursor.execute(
            """
            UPDATE users
            SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_hash, user_id),
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()

        return success


class InviteTokenDB:
    """Database operations for invite tokens."""
    
    @staticmethod
    def create_invite_token(created_by: int, expires_at: datetime) -> str:
        """Create an invite token."""
        import secrets
        conn = get_db_connection()
        cursor = conn.cursor()
        
        token = secrets.token_urlsafe(32)
        
        cursor.execute("""
            INSERT INTO invite_tokens (token, created_by, expires_at)
            VALUES (?, ?, ?)
        """, (token, created_by, expires_at))
        conn.commit()
        conn.close()
        
        return token
    
    @staticmethod
    def get_invite_token(token: str) -> Optional[dict]:
        """Get invite token if valid and not used."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM invite_tokens 
            WHERE token = ? AND used_by IS NULL AND expires_at > CURRENT_TIMESTAMP
        """, (token,))
        result = cursor.fetchone()
        conn.close()
        
        return dict(result) if result else None
    
    @staticmethod
    def mark_token_used(token: str, used_by: int) -> bool:
        """Mark invite token as used."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE invite_tokens 
            SET used_by = ?, used_at = CURRENT_TIMESTAMP 
            WHERE token = ?
        """, (used_by, token))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    
    @staticmethod
    def get_unused_tokens(created_by: int) -> List[dict]:
        """Get all unused tokens created by a user."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, token, created_at, expires_at, used_by 
            FROM invite_tokens 
            WHERE created_by = ? AND used_by IS NULL 
            ORDER BY created_at DESC
        """, (created_by,))
        tokens = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return tokens


class RefreshTokenDB:
    """Database operations for refresh tokens."""
    
    @staticmethod
    def create_refresh_token(user_id: int, expires_at: datetime) -> str:
        """Create a refresh token."""
        import secrets
        conn = get_db_connection()
        cursor = conn.cursor()
        
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        cursor.execute("""
            INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
            VALUES (?, ?, ?)
        """, (user_id, token_hash, expires_at))
        conn.commit()
        conn.close()
        
        return token
    
    @staticmethod
    def verify_refresh_token(user_id: int, token: str) -> bool:
        """Verify refresh token."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM refresh_tokens 
            WHERE user_id = ? AND token_hash = ? AND is_revoked = 0 
            AND expires_at > CURRENT_TIMESTAMP
        """, (user_id, token_hash))
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    
    @staticmethod
    def revoke_refresh_token(user_id: int, token: str) -> bool:
        """Revoke a refresh token."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE refresh_tokens 
            SET is_revoked = 1 
            WHERE user_id = ? AND token_hash = ?
        """, (user_id, token_hash))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
