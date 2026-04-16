# Power Rangers Authentication System - Quick Reference

## Overview
Three-tier role-based authentication (RBAC) system with JWT tokens and SQLite backend.

## User Roles
- **OPERATOR**: View forecasts, basic scenario modeling
- **ANALYST**: Operator permissions + advanced reports, scenario export
- **ADMIN**: Full system access + user management

## Getting Started

### Setup (First Time)
```bash
cd /home/gaurav/Documents/GitHub/SE_SEM_IV/Power-Rangers

# Install backend dependencies
pip install -r requirements.txt

# Initialize database & admin user
python src/auth/seed_admin.py

# Install frontend dependencies
cd frontend && npm install

# Start backend (terminal 1)
cd ..
uvicorn backend.main:app --reload --port 8000

# Start frontend (terminal 2)
cd frontend
npm run dev
```

### Test Credentials
- **Username**: `admin`
- **Password**: `changeme123!`
- Login at: http://localhost:3001/login

## Backend API Endpoints

### Public (No Auth Required)
- `POST /api/auth/login` - Login with username/password
- `POST /api/auth/register` - Register with invite token
- `POST /api/auth/refresh` - Refresh expired token
- `POST /api/forecast` - Forecast endpoint (unchanged, still public)
- `GET /api/health` - Health check

### Protected (Any Authenticated User)
- `GET /api/auth/me` - Get current user info
- `POST /api/auth/logout` - Logout and revoke tokens

### Admin Only
- `POST /api/admin/invite` - Generate invite token for new user
- `GET /api/admin/users` - List all users
- `PUT /api/admin/users/{id}/role` - Change user role
- `DELETE /api/admin/users/{id}` - Deactivate user

## Database Schema

### Users Table
```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  full_name TEXT,
  role TEXT NOT NULL DEFAULT 'OPERATOR',
  is_active INTEGER DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### Invite Tokens Table
```sql
CREATE TABLE invite_tokens (
  id INTEGER PRIMARY KEY,
  token TEXT UNIQUE NOT NULL,
  created_by INTEGER NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP NOT NULL,
  used_by INTEGER,
  used_at TIMESTAMP
)
```

### Refresh Tokens Table
```sql
CREATE TABLE refresh_tokens (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  token_hash TEXT UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP NOT NULL,
  is_revoked INTEGER DEFAULT 0
)
```

## Authentication Flow

### Login
1. User submits username + password to `POST /api/auth/login`
2. Backend validates credentials
3. Issues access_token (JWT, 2 hours) + refresh_token (7 days)
4. Frontend stores both in localStorage
5. Frontend attaches `Authorization: Bearer <access_token>` to requests

### Token Refresh
1. When access_token expires, frontend gets 401 response
2. Frontend automatically sends `POST /api/auth/refresh` with refresh_token
3. Backend validates and issues new access_token
4. Request is retried with new token

### Logout
1. Frontend sends `POST /api/auth/logout` with refresh_token
2. Backend marks token as revoked
3. Frontend clears tokens from localStorage
4. Redirect to login page

## Project Structure

```
/src/auth/
  ├── database.py          # SQLite models & operations
  ├── jwt_handler.py       # Token creation/validation
  ├── middleware.py        # Route protection dependencies  
  ├── models.py            # Pydantic schemas
  ├── routes.py            # All auth endpoints
  ├── seed_admin.py        # Initialize admin user
  └── __init__.py

/frontend/src/
  ├── hooks/
  │   └── useAuth.ts       # Zustand auth state
  ├── lib/
  │   └── authClient.ts    # Axios JWT wrapper
  ├── components/
  │   ├── ProtectedRoute.tsx
  │   └── UserMenu.tsx
  ├── pages/
  │   ├── Login.tsx
  │   ├── Register.tsx
  │   └── Dashboard.tsx
  └── App.tsx              # React Router setup
```

## Troubleshooting

### "Invalid or expired token"
- Access token expired → Use refresh_token to get new one
- Refresh token expired → User must login again

### "Admin access required"
- Only users with role='ADMIN' can access admin endpoints

### Frontend stuck on loading
- Check backend: `curl http://localhost:8000/api/health`
- Check browser console for errors

---

**Database**: `data/auth.db` (SQLite)
**Secrets**: `.env` file (JWT_SECRET_KEY)
