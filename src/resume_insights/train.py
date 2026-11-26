"""Train and evaluate the resume-category classifier."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from resume_insights.model import build_classifier, load_resume_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Resume Insights AI classifier.")
    parser.add_argument("--data", default="data/ResumeDataSet.csv", help="CSV containing Category and Resume columns")
    parser.add_argument("--model-out", default="models/resume_classifier.joblib", help="Path for the trained model artifact")
    parser.add_argument("--metrics-out", default="reports/metrics.json", help="Path for the JSON evaluation report")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data reserved for testing (0, 1)")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        raise SystemExit("--test-size must be between 0.05 and 0.5")

    texts, labels = load_resume_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    pipeline = build_classifier()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "TF-IDF + LinearSVC",
        "dataset": str(Path(args.data)),
        "samples_total": len(texts),
        "samples_train": len(x_train),
        "samples_test": len(x_test),
        "categories": sorted(set(labels)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=sorted(set(labels))),
        index=sorted(set(labels)),
        columns=sorted(set(labels)),
    ).to_csv(metrics_path.with_name("confusion_matrix.csv"), index=True)

    joblib.dump({"pipeline": pipeline, "metadata": {"model": metrics["model"], "categories": metrics["categories"]}}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
