# Power-Rangers

Electricity demand forecasting and peak-demand projection system for the Delhi power grid.

This repository currently uses:
- Python for data processing, model training, inference, and exports
- React + Vite for the frontend dashboard

## Project Structure

- `src/`: forecasting pipeline modules (feature engineering, training, inference, evaluation)
- `data/`: raw and generated forecast artifacts
- `config/`: runtime configuration
- `frontend/`: React dashboard (Vite + TypeScript + Tailwind)

## Setup

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

**Linux/Mac:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

## Backend (Python) Setup

1. Create and activate a virtual environment (see steps above).

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Seed the admin account:

```bash
python src/auth/seed_admin.py
```

4. Start the uvicorn server:

```bash
uvicorn backend.main:app --reload
```

## Frontend (React + Vite) Setup

```bash
cd frontend
npm i
npm run dev
```
