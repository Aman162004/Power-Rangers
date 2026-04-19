FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    OPERATIONAL_DIR=/tmp/power-rangers/operational \
    AUTH_DB_PATH=/tmp/auth.db \
    SLDC_LOAD_CACHE_DIR=/tmp/power-rangers/operational/raw \
    OPENMETEO_CACHE_NAME=/tmp/power-rangers/openmeteo

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY src ./src
COPY config ./config
COPY ["models/final model", "./models/final model"]
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]