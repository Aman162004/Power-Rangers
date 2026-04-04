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

## Backend (Python) Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run forecasting pipeline/inference entrypoints as needed, for example:

```bash
python main.py
```

## Frontend (React + Vite) Setup

1. Go to frontend folder:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start development server:

```bash
npm run dev
```

4. Build production assets:

```bash
npm run build
```
