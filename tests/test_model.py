"""
tests/test_model.py — Model tests.

These tests check prediction shape, output type, and basic accuracy.
A synthetic dataset is used so no external files are needed.

Run:  pytest tests -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import CATEGORICAL_COLS, NUMERIC_COLS, build_preprocessor


# ---------------------------------------------------------------------------
# Fixture: small synthetic dataset + trained model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_artifacts():
    """
    Create a small synthetic dataset, train a preprocessor and model,
    and return (model, preprocessor, X_test, y_test).
    """
    rng = np.random.default_rng(42)
    n = 500

    X = pd.DataFrame({
        "age":            rng.integers(18, 75, n).astype(float),
        "fnlwgt":         rng.integers(10000, 999999, n).astype(float),
        "education-num":  rng.integers(1, 16, n).astype(float),
        "capital-gain":   rng.integers(0, 99999, n).astype(float),
        "capital-loss":   rng.integers(0, 4356, n).astype(float),
        "hours-per-week": rng.integers(1, 99, n).astype(float),
        "workclass":      rng.choice(["Private", "Self-emp-not-inc", "Federal-gov"], n),
        "education":      rng.choice(["Bachelors", "HS-grad", "Masters", "Some-college"], n),
        "marital-status": rng.choice(["Never-married", "Married-civ-spouse", "Divorced"], n),
        "occupation":     rng.choice(["Prof-specialty", "Exec-managerial", "Sales", "Adm-clerical"], n),
        "relationship":   rng.choice(["Husband", "Not-in-family", "Wife", "Unmarried"], n),
        "race":           rng.choice(["White", "Black", "Asian-Pac-Islander"], n),
        "sex":            rng.choice(["Male", "Female"], n),
        "native-country": rng.choice(["United-States", "Mexico", "India"], n),
    })

    # Simple rule: higher education + more hours = higher income chance
    y = ((X["education-num"] > 10) & (X["hours-per-week"] > 35)).astype(int)

    split = int(0.8 * n)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_t, y_train)

    return model, preprocessor, X_test_t, y_test.values


# ---------------------------------------------------------------------------
# Test 1: Prediction shape and type
# ---------------------------------------------------------------------------

def test_prediction_shape_and_type(trained_artifacts):
    """Check predictions are 1D, match test size, and are binary."""
    model, _, X_test_t, y_test = trained_artifacts
    preds = model.predict(X_test_t)

    assert preds.ndim == 1, "Predictions should be 1-D"
    assert len(preds) == len(y_test), "Prediction count must match test set"
    assert set(preds).issubset({0, 1}), "Predictions must be 0 or 1"


# ---------------------------------------------------------------------------
# Test 2: Minimum accuracy check
# ---------------------------------------------------------------------------

def test_minimum_accuracy(trained_artifacts):
    """Model should reach at least 70% accuracy."""
    model, _, X_test_t, y_test = trained_artifacts
    preds = model.predict(X_test_t)
    acc = accuracy_score(y_test, preds)
    assert acc >= 0.70, (
        f"Model accuracy {acc:.2%} is below 70%"
    )


# ---------------------------------------------------------------------------
# Test 3: Probability output range check
# ---------------------------------------------------------------------------

def test_predict_proba_range(trained_artifacts):
    """Probabilities must stay within [0, 1]."""
    model, _, X_test_t, _ = trained_artifacts
    probs = model.predict_proba(X_test_t)[:, 1]
    assert probs.min() >= 0.0, "Probabilities cannot be negative"
    assert probs.max() <= 1.0, "Probabilities cannot exceed 1"