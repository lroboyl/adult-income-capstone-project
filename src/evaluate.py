"""
evaluate.py — Model Evaluation

This file contains helper functions for evaluating machine learning models and
comparing MLflow experiment runs.

It calculates common classification metrics, displays the results in a
readable format, and identifies the best-performing MLflow experiment.
"""

from __future__ import annotations

# ── Standard Library ─────────────────────────────────────────────────────────
# typing — defines the return type for dictionaries
from typing import Dict

# ── Third-Party Libraries ───────────────────────────────────────────────────
# numpy            — handles arrays used for predictions
# pandas           — provides the DataFrame type returned by mlflow.search_runs()
# scikit-learn metrics — calculates model evaluation metrics
#
# Metrics used:
# - Accuracy
# - Precision
# - Recall
# - F1 Score
# - ROC-AUC
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# compute_metrics()
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray | None = None) -> Dict[str, float]:
    """
    Calculates the model's performance on the test data.

    Inputs
    ------
    y_true : The actual labels.
    y_pred : The model's predicted labels.
    y_prob : (optional) The predicted probability for the positive class.

    Returns
    -------
    A dictionary containing:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC (when prediction probabilities are available)

    All values are rounded to four decimal places.

    Using predicted probabilities for ROC-AUC provides a better measure of
    model performance than predicted labels alone.
    """
    metrics: Dict[str, float] = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    if y_prob is not None:
        metrics["auc"] = round(roc_auc_score(y_true, y_prob), 4)

    return metrics


# =============================================================================
# print_metrics()
# =============================================================================

def print_metrics(model_name: str, metrics: Dict[str, float]) -> None:
    """
    Displays the evaluation metrics in a clean, easy-to-read format.

    This makes it easier to compare model performance while training
    multiple experiments.

    Inputs
    ------
    model_name : A label identifying the run, printed as a header
                 (e.g. the model_key string used in train.py, such as
                 "xgboost" or "random_forest_shallow").
    metrics    : A dict of metric name -> value, in the same shape
                 compute_metrics() returns (accuracy, precision, recall,
                 f1, and optionally auc). Any dict of string keys and
                 float-formattable values works — this function doesn't
                 require the exact keys compute_metrics() produces, just
                 that each value supports the :.4f format spec used below.
    """
    width = 40
    print(f"\n{'─' * width}")
    print(f"  {model_name}")
    print(f"{'─' * width}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")
    print(f"{'─' * width}\n")


# =============================================================================
# select_best_run()
# =============================================================================

def select_best_run(runs_df: pd.DataFrame, metric: str = "metrics.auc") -> str:
    """
    Find the best MLflow run using a selected evaluation metric.

    By default, the function compares runs using ROC-AUC, but any metric
    logged to MLflow can be used.

    The function:
    1. Checks that the DataFrame is not empty.
    2. Verifies the requested metric exists.
    3. Finds the run with the highest metric value.
    4. Returns the run ID.

    Runs with missing metric values are ignored. If no valid values are
    found for the selected metric, a clear error is raised.

    If two runs tie exactly on the chosen metric, the one that appears first in runs_df wins — not necessarily the one that finished training first or has any other meaningful claim to being 'better.' 
    This project logs metrics rounded to 4 decimal places... an exact tie is plausible... No deliberate tie-breaking rule is implemented; 
    ties are resolved by row order, and that's considered acceptable rather than a bug.

    """
    if runs_df.empty:
        raise ValueError("No runs found in the provided DataFrame.")
    if metric not in runs_df.columns:
        raise ValueError(f"Metric '{metric}' not found in runs DataFrame.")
    if runs_df[metric].isna().all():
        raise ValueError(
            f"Metric '{metric}' is missing (NaN) for every run — "
            "no best run can be selected."
        )

    best_row = runs_df.loc[runs_df[metric].idxmax()]
    return best_row["run_id"]