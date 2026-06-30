"""
train.py — Model Training and Experiment Tracking

This script trains multiple machine learning models on the Adult Income dataset
and tracks each experiment with MLflow.

Each model is trained using the same training and testing data so their
performance can be compared fairly. After all experiments finish, the script
identifies the best-performing model and saves it for the application to use.

What this script does
----------------------
1. Loads the project configuration from config.yaml.
2. Loads and cleans the Adult Income dataset.
3. Splits the data into training and testing sets.
4. Fits the preprocessing pipeline using only the training data to prevent data leakage.
5. Trains every model defined in the configuration file.
6. Evaluates each model using several classification metrics.
7. Logs model parameters, metrics, and artifacts to MLflow.
8. Saves each trained model and the fitted preprocessor locally.
9. Compares all MLflow runs and selects the best model based on ROC-AUC.
10. Saves the name of the best model so the application can load it automatically.

Usage:
    python src/train.py                         # uses configs/config.yaml
    python src/train.py --config path/to.yaml
"""

from __future__ import annotations

# ── Standard library ────────────────────────────────────────────────────────
# argparse — reads command-line arguments
# logging  — displays progress and error messages
# sys      — allows the script to modify the Python search path
# Path     — works with file and directory paths
import argparse
import logging
import sys
from pathlib import Path

# ── Third-party libraries ───────────────────────────────────────────────────
# joblib   — saves and loads trained models
# mlflow   — tracks machine learning experiments
# yaml     — reads the configuration file
import joblib
import mlflow
import mlflow.sklearn
import yaml

# ── Machine learning ────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Make sure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

# ── Project modules ─────────────────────────────────────────────────────────
# preprocess.py — data loading, cleaning, preprocessing
# evaluate.py   — metric calculation and experiment comparison
from preprocess import (
    build_preprocessor,
    clean_data,
    load_raw_data,
    split_features_target,
)
from evaluate import compute_metrics, print_metrics, select_best_run

# ── Logging ──────────────────────────────────────────────────────────────────
# Logging is configured to display timestamps, log levels, and progress
# messages while the models are training.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def load_config(path: str = "configs/config.yaml") -> dict:
    """
    Loads the YAML configuration file and returns it as a Python dictionary.

    The configuration contains settings such as:
    - Dataset location
    - Train/test split
    - Random seed
    - Model hyperparameters
    - MLflow settings

    Keeping these values in one file makes experiments easier to reproduce.
    """
    with open(path) as fh:
        return yaml.safe_load(fh)


# =============================================================================
# Building Models
# =============================================================================

def build_model(model_key: str, cfg: dict):
    """
    Creates the requested machine learning model using the settings stored in
    config.yaml.

    Supported models include:
    - Logistic Regression
    - Random Forest (default)
    - Random Forest (shallow)
    - Gradient Boosting
    - XGBoost

    The function returns:
    - The initialized model
    - The model's hyperparameters

    This keeps all model creation in one place and avoids repeating code.
    """
    params = cfg["models"][model_key]

    if model_key == "logistic_regression":
        return LogisticRegression(**params), params

    if model_key == "random_forest_default":
        return RandomForestClassifier(**params), params

    if model_key == "random_forest_shallow":
        return RandomForestClassifier(**params), params

    if model_key == "gradient_boosting":
        return GradientBoostingClassifier(**params), params

    if model_key == "xgboost":
        p = dict(params)
        # use_label_encoder was used in XGBoost <1.6 to control automatic
        # label encoding; it's fully removed from XGBClassifier's
        # constructor as of the version pinned in requirements.txt
        # (2.1.4) — confirmed via inspect.signature(), not just assumed.
        # config.yaml no longer sets this key, but the defensive pop stays
        # in case someone reintroduces it from an outdated example: passing
        # it would otherwise be silently absorbed into get_params() as
        # dead, confusing metadata rather than raising an error.
        p.pop("use_label_encoder", None)
        return XGBClassifier(**p), p

    raise ValueError(f"Unknown model key: {model_key}")


# =============================================================================
# Training One Model
# =============================================================================

def run_experiment(model_key: str, cfg: dict,
                   X_train, X_test, y_train, y_test,
                   preprocessor):
    """
    This function trains one model, evaluates it, and records the results in
    MLflow.

    For each experiment it:
    1. Creates the model.
    2. Starts a new MLflow run.
    3. Logs model parameters and dataset information.
    4. Transforms the training and testing data using the fitted preprocessor.
    5. Trains the model.
    6. Makes predictions on the test set.
    7. Calculates evaluation metrics.
    8. Logs all metrics to MLflow.
    9. Saves the trained model to MLflow.
    10. Saves a local copy of the model for the application.

    The function returns the MLflow run ID and the evaluation metrics.
    """

    model, params = build_model(model_key, cfg)

    with mlflow.start_run(run_name=model_key) as run:
        # ── Log hyperparameters ──────────────────────────────────────────
        mlflow.log_param("model_type", model_key)
        mlflow.log_param("data_version", cfg["data"]["path"])
        mlflow.log_param("test_size", cfg["data"]["test_size"])
        for k, v in params.items():
            mlflow.log_param(k, v)

        # ── Preprocess ───────────────────────────────────────────────────
        X_train_t = preprocessor.transform(X_train)
        X_test_t  = preprocessor.transform(X_test)

        # ── Train ────────────────────────────────────────────────────────
        logger.info("Training  %s ...", model_key)
        model.fit(X_train_t, y_train)

        # ── Evaluate ─────────────────────────────────────────────────────
        y_pred = model.predict(X_test_t)
        y_prob = (model.predict_proba(X_test_t)[:, 1]
                  if hasattr(model, "predict_proba") else None)
        metrics = compute_metrics(y_test.values, y_pred, y_prob)

        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        print_metrics(model_key, metrics)

        # ── Save model artifact ──────────────────────────────────────────
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Also save locally so the app can load it without MLflow
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        joblib.dump(model, models_dir / f"{model_key}.joblib")

        run_id = run.info.run_id
        logger.info("Run %s complete  (run_id=%s)", model_key, run_id)
        return run_id, metrics


# =============================================================================
# Comparing Experiments
# =============================================================================

def identify_best_run(cfg: dict) -> str:
    """
    After every model has been trained, this function compares all MLflow runs.

    It:
    - Retrieves every run from the configured experiment.
    - Sorts them by ROC-AUC.
    - Prints a comparison table.
    - Identifies the best-performing model.
    - Saves the model name to models/best_model.txt.

    The application later reads this file to automatically load the best model.
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    experiment_name = cfg["mlflow"]["experiment_name"]

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.auc DESC"],
    )

    if runs.empty:
        logger.warning("No runs found for experiment '%s'", experiment_name)
        return ""

    print("\n" + "═" * 70)
    print("  EXPERIMENT COMPARISON  (sorted by AUC ↓)")
    print("═" * 70)
    cols = ["tags.mlflow.runName", "metrics.accuracy",
            "metrics.precision", "metrics.recall",
            "metrics.f1", "metrics.auc"]
    available = [c for c in cols if c in runs.columns]
    print(runs[available].to_string(index=False))
    print("═" * 70)

    best_run_id = select_best_run(runs, metric="metrics.auc")
    best_name   = runs.loc[runs["run_id"] == best_run_id,
                           "tags.mlflow.runName"].values[0]
    best_auc    = runs.loc[runs["run_id"] == best_run_id,
                           "metrics.auc"].values[0]

    print(f"\n  ✓  Best run: '{best_name}'  (AUC = {best_auc:.4f})")
    print(f"     Run ID : {best_run_id}\n")

    # Persist best model key for the app to load.
    # Ensure models/ exists first — this function may be called on its own
    # (e.g. from a notebook or a separate script) without main() having
    # already created the directory.
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "best_model.txt", "w") as fh:
        fh.write(best_name)

    return best_run_id


# =============================================================================
# Main Program
# =============================================================================
#
# This is the entry point of the training pipeline.
#
# The workflow is:
# 1. Load the configuration file.
# 2. Load and clean the dataset.
# 3. Split the data into training and testing sets.
# 4. Fit the preprocessing pipeline using only the training data.
# 5. Save the fitted preprocessor.
# 6. Train every configured model.
# 7. Compare all MLflow experiments.
# 8. Select and save the best model.
#
# Why fit the preprocessor only on the training data?
# -----------------------------------------------------
# The preprocessing pipeline is fitted only on the training data to prevent
# data leakage.
#
# If information from the test data were used during preprocessing, the
# evaluation results would be overly optimistic because the model would
# indirectly see data it should not have access to during training.
#
# The same fitted preprocessor is saved and reused during inference so new
# data is transformed exactly the same way as the training data.
#
# Output
# -------
# After the script finishes, it produces:
# - Trained model files in models/
# - The fitted preprocessing pipeline
# - MLflow experiment logs
# - Evaluation metrics for every model
# - A comparison table of all experiments
# - best_model.txt, which identifies the highest-performing model for the
#   application

def main():
    parser = argparse.ArgumentParser(description="Train Adult Income models")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Load & clean data ─────────────────────────────────────────────────
    df_raw   = load_raw_data(cfg["data"]["path"])
    df_clean = clean_data(df_raw)
    X, y     = split_features_target(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size   = cfg["data"]["test_size"],
        random_state= cfg["data"]["random_state"],
        stratify    = y,
    )
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))

    # ── Fit preprocessor on training data only (no leakage) ──────────────
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Save preprocessor so the app can re-use it
    Path("models").mkdir(exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")

    # ── Configure MLflow once, before training any model ──────────────────
    # This is run-level configuration (which tracking server and which
    # experiment to log to), not per-model configuration, so it belongs
    # here rather than being repeated on every loop iteration inside
    # run_experiment().
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ── Train all configured models ───────────────────────────────────────
    model_keys = list(cfg["models"].keys())
    for key in model_keys:
        try:
            run_experiment(key, cfg, X_train, X_test, y_train, y_test, preprocessor)
        except Exception as exc:
            logger.error("Failed to train %s: %s", key, exc, exc_info=True)

    # ── Compare experiments and pick the best ────────────────────────────
    identify_best_run(cfg)


if __name__ == "__main__":
    main()