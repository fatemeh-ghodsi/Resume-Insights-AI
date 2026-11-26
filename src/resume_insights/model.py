"""Text cleaning, dataset validation, and model construction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_SPACE_RE = re.compile(r"\s+")


def clean_resume_text(value: object) -> str:
    """Normalize resume text while retaining useful technical tokens (C++, C#, .NET)."""
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = text.replace("\x00", " ")
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def load_resume_dataset(path: str | Path) -> Tuple[list[str], list[str]]:
    """Load a CSV with Category and Resume columns and validate its contents."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"Category", "Resume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns: {', '.join(sorted(required))}. Missing: {', '.join(sorted(missing))}")

    frame = frame.dropna(subset=["Category", "Resume"]).copy()
    frame["Category"] = frame["Category"].astype(str).str.strip()
    frame["Resume"] = frame["Resume"].map(clean_resume_text)
    frame = frame[(frame["Category"] != "") & (frame["Resume"] != "")]

    counts = frame["Category"].value_counts()
    rare = counts[counts < 2]
    if not rare.empty:
        labels = ", ".join(f"{label} ({count})" for label, count in rare.items())
        raise ValueError("Each category needs at least 2 samples for a stratified split. Rare categories: " + labels)
    if frame["Category"].nunique() < 2:
        raise ValueError("At least two distinct resume categories are required.")

    return frame["Resume"].tolist(), frame["Category"].tolist()


def build_classifier() -> Pipeline:
    """Create a reproducible TF-IDF + linear SVM text-classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_resume_text,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=150_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )
