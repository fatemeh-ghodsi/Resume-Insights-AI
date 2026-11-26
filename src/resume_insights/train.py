"""Train, benchmark, and evaluate resume-category classifiers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from resume_insights.model import build_classifier, clean_resume_text, load_resume_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark resume classifiers.")
    parser.add_argument("--data", default="data/ResumeDataSet.csv", help="CSV containing Category and Resume columns")
    parser.add_argument("--model-out", default="models/resume_classifier.joblib", help="Path for the LinearSVC model artifact")
    parser.add_argument("--metrics-out", default="reports/metrics.json", help="Path for the JSON evaluation report")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data reserved for testing (0, 1)")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_naive_bayes_classifier() -> Pipeline:
    """Recreate a TF-IDF + MultinomialNB baseline for a fair same-split comparison."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                preprocessor=clean_resume_text,
                ngram_range=(1, 2),
                min_df=1,
                max_features=150_000,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("classifier", MultinomialNB()),
        ]
    )


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        raise SystemExit("--test-size must be between 0.05 and 0.5")

    texts, labels = load_resume_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size,
        random_state=args.random_state, stratify=labels,
    )

    candidates = {
        "TF-IDF + MultinomialNB": build_naive_bayes_classifier(),
        "TF-IDF + LinearSVC": build_classifier(),
    }
    results = {}
    predictions_by_model = {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        predictions_by_model[name] = predictions
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
        }

    labels_sorted = sorted(set(labels))
    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(args.data)),
        "split": {"test_size": args.test_size, "random_state": args.random_state, "strategy": "stratified"},
        "samples_total": len(texts),
        "samples_train": len(x_train),
        "samples_test": len(x_test),
        "categories": labels_sorted,
        "models": results,
        "selected_model_artifact": "TF-IDF + LinearSVC",
        "historical_notebook_reference": {
            "model": "TF-IDF + MultinomialNB (original notebook)",
            "reported_test_accuracy": 0.9378,
            "note": "Historical notebook result only; its split/preprocessing may differ. Do not treat as a directly comparable benchmark."
        },
    }

    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = metrics_path.parent

    cm = confusion_matrix(y_test, predictions_by_model["TF-IDF + LinearSVC"], labels=labels_sorted)
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(
        report_dir / "confusion_matrix_linearsvc.csv", index=True
    )
    cm_nb = confusion_matrix(y_test, predictions_by_model["TF-IDF + MultinomialNB"], labels=labels_sorted)
    pd.DataFrame(cm_nb, index=labels_sorted, columns=labels_sorted).to_csv(
        report_dir / "confusion_matrix_naive_bayes.csv", index=True
    )

    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric_name, title in zip(
        axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro F1")
    ):
        values = [results[name][metric_name] for name in names]
        bars = ax.bar(["MultinomialNB", "LinearSVC"], values)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Resume classifier benchmark (same stratified test split)")
    fig.tight_layout()
    fig.savefig(report_dir / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 11))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted).plot(
        ax=ax, xticks_rotation=90, colorbar=False, cmap="Blues"
    )
    ax.set_title("LinearSVC confusion matrix")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix_linearsvc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    linearsvc = candidates["TF-IDF + LinearSVC"]
    joblib.dump(
        {"pipeline": linearsvc, "metadata": {"model": "TF-IDF + LinearSVC", "categories": labels_sorted}},
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for name, values in results.items():
        print(f"{name}: accuracy={values['accuracy']:.4f}, macro_f1={values['macro_f1']:.4f}, weighted_f1={values['weighted_f1']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Charts saved to: {report_dir}")


if __name__ == "__main__":
    main()
