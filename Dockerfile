# Use a small Python base image for a reproducible local API container.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RESUME_MODEL_PATH=/app/models/resume_classifier.joblib

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "resume_insights.api:app", "--host", "0.0.0.0", "--port", "8000"]
