"""
preprocess.py - Data loading, cleaning, and preprocessing for the Adult Income dataset.

Prepares the data for model training and prediction by cleaning the dataset,
splitting features and the target, and building the preprocessing pipeline.

The Adult Income dataset predicts whether a person earns >$50K/year based on
census demographic and employment attributes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

NUMERIC_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]

CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

# Columns used by the LLM interface (subset that users typically know)
USER_FACING_FEATURES = [
    "age", "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "hours-per-week",
    "capital-gain", "capital-loss", "native-country",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load the Adult Income dataset from a CSV or the original UCI .data file.
    Handles both the header-less UCI format and a CSV with headers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download from https://archive.ics.uci.edu/dataset/2/adult "
            "and place it at data/adult.csv"
        )

    # Check if the file already has headers
    with open(path) as fh:
        first_line = fh.readline().strip()

    if first_line.startswith("age"):
        # Load CSV with headers
        df = pd.read_csv(path)
    else:
        # Load the original UCI format
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES,
                         skipinitialspace=True, na_values="?")

    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset without modifying the original dataframe.

    This function:
    - Strips whitespace
    - Replaces missing value markers
    - Converts the target to binary 0/1
    - Removes duplicate rows
    - Drops rows with missing values in required columns
    """
    df = df.copy()

    # Remove extra whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Replace '?' with NaN (missing values)
    # In the headerless UCI data loader, '?' is already treated as NaN using na_values="?".
    # However, this step is needed for the CSV-with-headers version, which does not handle it automatically.
    # It also catches cases like ' ?' after whitespace cleaning, ensuring they are normalized and treated as missing.
    df.replace("?", np.nan, inplace=True)

    # Convert the target column to binary values
    if "income" in df.columns:
        df["income"] = df["income"].str.replace(".", "", regex=False)
        df["income"] = (df["income"] == ">50K").astype(int)

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("Dropped %d duplicate rows", before - len(df))

    # Remove rows with missing required values
    required = NUMERIC_COLS + CATEGORICAL_COLS + ["income"]
    before = len(df)
    df.dropna(subset=required, inplace=True)
    logger.info("Dropped %d rows with missing values", before - len(df))

    logger.info("Clean dataset: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into features (X) and target (y)."""
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df["income"].copy()
    return X, y


# ---------------------------------------------------------------------------
# Sklearn preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Numeric features are standardized, and categorical features are one-hot encoded.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Convenience: full pipeline (preprocessor only, without estimator)
# ---------------------------------------------------------------------------

def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessing pipeline on the training data."""
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    return preprocessor


def transform(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Apply a fitted preprocessor to a dataset."""
    return preprocessor.transform(X)


# ---------------------------------------------------------------------------
# Build input for a single prediction (used by the LLM interface)
# ---------------------------------------------------------------------------

def build_inference_row(feature_dict: dict) -> pd.DataFrame:
    """
    Create a single-row dataframe from the feature values extracted by the LLM.

    Missing numeric values default to 0, and missing categorical values default
    to "Unknown".
    """
    row: dict = {}

    for col in NUMERIC_COLS:
        val = feature_dict.get(col, 0)
        try:
            row[col] = float(val)
        except (TypeError, ValueError):
            row[col] = 0.0

    for col in CATEGORICAL_COLS:
        row[col] = str(feature_dict.get(col, "Unknown"))

    return pd.DataFrame([row])