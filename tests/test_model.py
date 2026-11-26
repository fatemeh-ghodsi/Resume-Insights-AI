from resume_insights.model import build_classifier, clean_resume_text


def test_clean_resume_text_removes_urls_and_normalizes_whitespace():
    cleaned = clean_resume_text("Python   developer https://example.com\n C++ engineer")
    assert "https://example.com" not in cleaned
    assert "Python" in cleaned
    assert "C++" in cleaned
    assert "  " not in cleaned


def test_classifier_can_fit_and_predict_small_dataset():
    texts = [
        "Python FastAPI PostgreSQL backend API developer",
        "Python Django SQL backend software engineer",
        "REST API Docker Python server developer",
        "Registered nurse patient care hospital clinical",
        "Nursing healthcare patient treatment clinical care",
        "Hospital nurse medical patient support",
    ]
    labels = ["Python Developer"] * 3 + ["Healthcare"] * 3
    model = build_classifier()
    model.fit(texts, labels)
    assert model.predict(["FastAPI Python PostgreSQL API"])[0] == "Python Developer"
