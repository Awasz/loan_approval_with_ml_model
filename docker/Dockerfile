FROM python:3.12-slim

WORKDIR /app

# libgomp: OpenMP runtime needed by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir ".[api]"

# Bundle written by `make export-model`
COPY models/champion ./models/champion
ENV MODEL_DIR=/app/models/champion

EXPOSE 8000
CMD ["uvicorn", "loan_approval.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
