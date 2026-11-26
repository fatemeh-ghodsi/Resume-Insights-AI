"""Run inference with a trained Resume Insights AI model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the most likely category for a resume.")
    parser.add_argument("--model", default="models/resume_classifier.joblib")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Resume text to classify")
    source.add_argument("--file", help="UTF-8 text file containing a resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.model)
    if not artifact_path.is_file():
        raise SystemExit(f"Model not found: {artifact_path}. Train it first with python -m resume_insights.train")

    resume_text = args.text if args.text is not None else Path(args.file).read_text(encoding="utf-8")
    artifact = joblib.load(artifact_path)
    prediction = artifact["pipeline"].predict([resume_text])[0]
    print(json.dumps({"predicted_category": str(prediction)}, indent=2))


if __name__ == "__main__":
    main()
