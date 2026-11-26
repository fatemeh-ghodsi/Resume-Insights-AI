# Resume Insights AI

A reproducible NLP project that classifies resume text into job-related categories and exposes predictions through a REST API.

## Current implementation

- CSV validation for the required `Category` and `Resume` columns
- Reusable text normalization and a scikit-learn Pipeline
- TF-IDF unigram/bigram features with a LinearSVC classifier
- Stratified train/test split and accuracy, macro-F1, weighted-F1 and per-class metrics
- Saved model artifact, JSON metrics and confusion-matrix CSV
- Command-line inference
- FastAPI health and prediction endpoints with lazy model loading
- Automated unit and API tests
- Dockerfile for containerized inference

**Model note:** LinearSVC is a candidate model, not a proven improvement over the original notebook's Multinomial Naive Bayes baseline. Train and compare both on the same held-out split before claiming a performance gain.

## Project structure

```text
.
├── data/                       # committed resume classification dataset
├── models/                     # generated model artifacts (not committed)
├── reports/                    # generated evaluation reports
├── src/resume_insights/
│   ├── __init__.py
│   ├── api.py                   # FastAPI application
│   ├── model.py                 # preprocessing and model pipeline
│   ├── predict.py               # CLI inference
│   └── train.py                 # training and evaluation
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── Dockerfile
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
.venv\\Scripts\\Activate.ps1

pip install -e ".[dev]"
```

## Dataset

The repository includes `data/ResumeDataSet.csv` (about 3.1 MB). The original notebook output indicates 962 rows after dropping missing values. The automated workflow uses this committed dataset. Do not commit private or identifiable candidate resumes. To substitute a different appropriately licensed dataset, keep the same two-column schema and update the workflow path.

Expected columns:

| Column | Description |
|---|---|
| `Category` | Target job category |
| `Resume` | Resume text |

Rows with missing category or resume text are discarded. Each category must have at least two samples for a stratified split. The committed dataset is used by the demo workflow. Verify its provenance and license before redistributing it or using it outside this portfolio project.

## Train and evaluate

```bash
python -m resume_insights.train --data data/ResumeDataSet.csv
```

The command writes:

- `models/resume_classifier.joblib`
- `reports/metrics.json`
- `reports/confusion_matrix_linearsvc.csv` and `reports/confusion_matrix_naive_bayes.csv`
- `reports/model_comparison.png` (Accuracy and Macro-F1)
- `reports/confusion_matrix_linearsvc.png`

Metrics are calculated on a held-out stratified test split. Do not compare results across runs unless the dataset version and split settings are the same.

## Run the API locally

Train the model first, then start the server from the repository root:

```bash
uvicorn resume_insights.api:app --app-dir src --reload
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Predict a category

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Experienced Python developer building FastAPI and PostgreSQL services"}'
```

The API accepts resume text between 20 and 30,000 characters. It does not log submitted resume text. Set `RESUME_MODEL_PATH` to use a model artifact at a non-default location.

## Automated GitHub Actions

The ML pipeline runs on pushes to `main` and the development branch, pull requests to `main`, and manual dispatch. It installs dependencies, runs tests, validates the committed CSV, trains both TF-IDF + MultinomialNB and TF-IDF + LinearSVC on the same stratified split, and uploads the trained LinearSVC artifact plus metrics and charts as a 30-day workflow artifact. Open the repository **Actions** tab, select **ML pipeline**, and open the latest successful run to download `resume-insights-ml-results`.

The original notebook reported 0.9378 test accuracy for its Naive Bayes experiment. That is a historical result, not a directly comparable score: the notebook's split and preprocessing may differ. The automated benchmark provides an apples-to-apples comparison on one shared split.

## Run tests

```bash
pytest
```

## Run with Docker

Build after training the model:

```bash
docker build -t resume-insights-ai .
docker run --rm -p 8000:8000 resume-insights-ai
```

The trained artifact must exist at `models/resume_classifier.joblib` before building the image. For production deployments, use a private image registry and appropriate access controls.

## Responsible use and privacy

This is a portfolio/learning classifier, not a validated hiring or candidate-screening decision system. Predictions can reflect dataset bias and must never be used as the sole basis for employment decisions. Resume text may contain personal data: do not upload real candidates' resumes to public services or commit them to Git. Obtain an appropriate legal basis and consent where required, minimize retained data, and anonymize examples.

Only load model artifacts that you created or trust. Joblib uses Python serialization and loading an untrusted artifact can execute code. The API is a demonstration and should be deployed behind authentication, rate limiting, TLS and suitable privacy controls before any real-world use.

## Next milestones

1. Benchmark Naive Bayes, LinearSVC and Logistic Regression on the same stratified split.
2. Add cross-validation and error analysis, including per-category precision/recall.
3. Add CI and a small synthetic demo dataset.
4. Consider multilingual models only after establishing a reliable baseline.

## License

Add a license file before redistributing this project or its dataset.
