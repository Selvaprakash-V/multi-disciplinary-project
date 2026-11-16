# Dockerfile for AcoustiSolar simulation & ML development
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/workspace/packages/schemas/src:/workspace/packages/physics/src:/workspace/packages/daqtwin/src:/workspace/packages/dsp/src:/workspace/packages/ml/src:/workspace/packages/sdk/src"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install numpy scipy pandas pydantic pyyaml scikit-learn joblib pyarrow pytest pytest-cov ruff mypy matplotlib

COPY . .

CMD ["pytest", "-v"]

