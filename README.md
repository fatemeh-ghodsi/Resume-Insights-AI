# Resume Insights AI

A reproducible NLP project that classifies resume text into job-related categories.

## Current implementation

The original notebook explored resume categorization with TF-IDF and Multinomial Naive Bayes. The new modular pipeline adds:

- CSV validation for the required `Category` and `Resume` columns
- Reusable text normalization and a scikit-learn Pipeline
- TF-IDF unigram/bigram features with a LinearSVC classifier
- Stratified train/test split and accuracy, macro-F1, weighted-F1 and per-class metrics
- Saved model artifact, JSON metrics and confusion-matrix CSV
- Command-line inference and basic automated tests

**Important:** the LinearSVC pipeline is a new candidate model, not a proven improvement yet. Run it on the dataset and compare its held-out metrics against the notebook's Naive Bayes baseline before claiming a performance gain.

## Project structure

```text
.
├── data/
│   └── ResumeDataSet.csv       # not committed; provide your own dataset
├── models/                     # generated model artifacts (not committed)
├── reports/                    # generated evaluation reports
├── src/
│   └── resume_insights/
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       └── predict.py
├── tests/
│   └── test_model.py
├── pyproject.toml
└── requirements.txt
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

## Dataset

Place a CSV at `data/ResumeDataSet.csv` with these columns:

| Column | Description |
|---|---|
| `Category` | Target job category |
| `Resume` | Resume text |

Rows with missing category or resume text are discarded. Each category must have at least two samples for a stratified split. The dataset is intentionally not included in this repository; use a dataset you are licensed to process.

## Train and evaluate

```bash
python -m resume_insights.train --data data/ResumeDataSet.csv
```

Optional arguments:

```bash
python -m resume_insights.train --help
```

The command writes:

- `models/resume_classifier.joblib`
- `reports/metrics.json`
- `reports/confusion_matrix.csv`

Metrics are calculated on a held-out stratified test split. Do not compare results across runs unless the dataset version and split settings are the same.

## Predict

After training:

```bash
python -m resume_insights.predict --text "Python developer with FastAPI and PostgreSQL experience"
```

Or provide a UTF-8 text file:

```bash
python -m resume_insights.predict --file path/to/resume.txt
```

## Run tests

```bash
pytest
```

## Responsible use and privacy

This is a portfolio/learning classifier, not a validated hiring or candidate-screening decision system. Predictions can reflect dataset bias and should never be used as the sole basis for employment decisions. Resume text may contain personal data: do not upload real candidates' resumes to public services or commit them to Git. Obtain an appropriate legal basis and consent where required, minimize retained data, and anonymize examples.

Only load model artifacts that you created or trust. Joblib uses Python serialization and loading an untrusted artifact can execute code.

## Next milestones

1. Benchmark Naive Bayes, LinearSVC and Logistic Regression on the same stratified split.
2. Add cross-validation and error analysis, including per-category precision/recall.
3. Add a FastAPI inference endpoint with request validation.
4. Add Docker, CI and a small synthetic demo dataset.
5. Consider multilingual models only after establishing a reliable baseline.

## License

Add a license file before redistributing this project or its dataset.
