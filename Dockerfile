FROM python:3.11-slim AS builder

WORKDIR /build

# System-level build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer cache.
# Re-installing packages is only triggered when requirements.txt changes.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


#lean runtime
FROM python:3.11-slim AS runtime

LABEL maintainer="journeyiq-mlops" \
      version="1.0.0" \
      description="JourneyIQ Travel Intelligence — REST API"

WORKDIR /app

# Runtime system dependency
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Bring in installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY api_server.py          .
COPY models/                models/

# Create non-root user for security — never run production containers as root
RUN useradd --create-home --shell /bin/bash journeyiq \
 && chown -R journeyiq:journeyiq /app
USER journeyiq

# Environment variables
ENV PORT=5050 \
    MODEL_PATH=models/flight_price_model.joblib \
    ENCODER_PATH=models/label_encoders.joblib \
    MODEL_VERSION=v1.0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE ${PORT}

# Health check — Kubernetes liveness probe uses this
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:${PORT}/api/v1/health'); exit(0 if r.status_code == 200 else 1)"

# Gunicorn is preferred over Flask's dev server in production.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5050", \
     "--workers", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "api_server:create_app()"]
