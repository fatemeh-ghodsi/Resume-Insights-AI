"""FastAPI application for resume-category inference.

This API is a portfolio demonstration, not an employment decision tool.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Text submitted for category prediction."""

    text: str = Field(min_length=20, max_length=30_000, description="Resume text to classify")


class PredictionResponse(BaseModel):
    predicted_category: str
    model: str


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    """Load a trusted model artifact created by this project."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Invalid model artifact: expected a dictionary with a 'pipeline' key")
    return artifact


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create an app; model loading is lazy so health checks work before training."""
    resolved_path = Path(model_path or os.getenv("RESUME_MODEL_PATH", "models/resume_classifier.joblib"))
    application = FastAPI(
        title="Resume Insights AI",
        version="0.3.0",
        description=(
            "Resume text category prediction API for educational and portfolio use. "
            "Predictions must not be used as the sole basis for employment decisions."
        ),
    )
    application.state.model_path = resolved_path
    application.state.model_artifact = None

    @application.get("/health")
    def health(request: Request) -> dict[str, str | bool]:
        return {
            "status": "ok",
            "model_loaded": request.app.state.model_artifact is not None,
        }

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        artifact = request.app.state.model_artifact
        if artifact is None:
            try:
                artifact = load_artifact(request.app.state.model_path)
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Model is not available. Train it first using the documented training command.",
                ) from exc
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=500, detail="The configured model artifact is invalid.") from exc
            request.app.state.model_artifact = artifact

        try:
            prediction = artifact["pipeline"].predict([payload.text])[0]
        except Exception as exc:
            # Do not expose model internals or submitted resume content in API errors.
            raise HTTPException(status_code=500, detail="Prediction failed.") from exc

        metadata = artifact.get("metadata", {})
        model_name = str(metadata.get("model", "resume-classifier"))
        return PredictionResponse(predicted_category=str(prediction), model=model_name)

    return application


app = create_app()
