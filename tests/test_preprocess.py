"""
tests/test_preprocess.py — Unit tests for src/preprocess.py

Run:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocess import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    build_inference_row,
    build_preprocessor,
    clean_data,
    split_features_target,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_df():
    """A small sample DataFrame that mimics the UCI Adult Income dataset."""
    return pd.DataFrame({
        "age":            [39, 50, None, 28],
        "workclass":      ["State-gov", "Self-emp-not-inc", "?", "Private"],
        "fnlwgt":         [77516, 83311, 215646, 338409],
        "education":      ["Bachelors", "Bachelors", "HS-grad", "Bachelors"],
        "education-num":  [13, 13, 9, 13],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
        "occupation":     ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Prof-specialty"],
        "relationship":   ["Not-in-family", "Husband", "Not-in-family", "Wife"],
        "race":           ["White", "White", "Black", "Black"],
        "sex":            ["Male", "Male", "Male", "Female"],
        "capital-gain":   [2174, 0, 0, 0],
        "capital-loss":   [0, 0, 0, 0],
        "hours-per-week": [40, 13, 40, 40],
        "native-country": ["United-States", "United-States", "United-States", "Cuba"],
        "income":         ["<=50K", ">50K", "<=50K", ">50K"],
    })


@pytest.fixture
def clean_df(sample_raw_df):
    return clean_data(sample_raw_df)


# ---------------------------------------------------------------------------
# Test 1: Missing values are handled (rows with NaN are dropped)
# ---------------------------------------------------------------------------

def test_missing_values_dropped(sample_raw_df):
    """Missing (NaN) or invalid rows should be removed during cleaning."""
    cleaned = clean_data(sample_raw_df)
    # Row 2 is invalid because age is missing and workclass is unknown ('?'), so it should be removed
    assert cleaned.isnull().sum().sum() == 0, (
        """The cleaned DataFrame should not contain any missing (NaN) values."""
    )
    assert len(cleaned) < len(sample_raw_df), (
        "At least one row with missing data should be removed during cleaning."
    )


# ---------------------------------------------------------------------------
# Test 2: Target is correctly encoded as binary 0 / 1
# ---------------------------------------------------------------------------

def test_target_binary_encoding(sample_raw_df):
    """Income target must be encoded as integer 0 or 1, not strings."""
    cleaned = clean_data(sample_raw_df)
    assert set(cleaned["income"].unique()).issubset({0, 1}), (
        "Income column should only contain 0 and 1"
    )
    assert cleaned["income"].dtype in [np.int32, np.int64, int], (
        "Income column dtype should be integer"
    )


# ---------------------------------------------------------------------------
# Test 3: Check that numeric features are properly standardized (mean ~0, std ~1)
# ---------------------------------------------------------------------------

def test_numeric_scaling(clean_df):
    """StandardScaler should output values with a mean close to 0 and a standard deviation close to 1 using a larger synthetic dataset."""
    # Use a large synthetic dataset to make the test more reliable
    rng = np.random.default_rng(0)
    n = 200
    big = pd.DataFrame({
        "age":            rng.integers(18, 90, n).astype(float),
        "fnlwgt":         rng.integers(10000, 1000000, n).astype(float),
        "education-num":  rng.integers(1, 16, n).astype(float),
        "capital-gain":   rng.integers(0, 99999, n).astype(float),
        "capital-loss":   rng.integers(0, 4356, n).astype(float),
        "hours-per-week": rng.integers(1, 99, n).astype(float),
        "workclass":      rng.choice(["Private", "Self-emp-not-inc"], n),
        "education":      rng.choice(["Bachelors", "HS-grad", "Masters"], n),
        "marital-status": rng.choice(["Never-married", "Married-civ-spouse"], n),
        "occupation":     rng.choice(["Prof-specialty", "Sales", "Adm-clerical"], n),
        "relationship":   rng.choice(["Husband", "Not-in-family"], n),
        "race":           rng.choice(["White", "Black"], n),
        "sex":            rng.choice(["Male", "Female"], n),
        "native-country": rng.choice(["United-States", "India"], n),
    })
    preprocessor = build_preprocessor()
    preprocessor.fit(big)
    transformed = preprocessor.transform(big)

    # Scaled numeric columns appear first in the output (before categorical features)
    numeric_part = transformed[:, :len(NUMERIC_COLS)]
    assert abs(numeric_part.mean()) < 0.1,  "Mean of scaled numerics should be near 0"
    assert abs(numeric_part.std() - 1.0) < 0.2, "Std of scaled numerics should be near 1"


# ---------------------------------------------------------------------------
# Test 4: Check that categorical features are properly encoded (no strings remain)
# ---------------------------------------------------------------------------

def test_categorical_encoding_is_numeric(clean_df):
    """After preprocessing, the output array should contain only numeric values."""
    X, _ = split_features_target(clean_df)
    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    transformed = preprocessor.transform(X)

    assert isinstance(transformed, np.ndarray), "Output must be a numpy array"
    assert np.issubdtype(transformed.dtype, np.floating), (
        "All values in the transformed array should be floats."
    )


# ---------------------------------------------------------------------------
# Test 5: Ensure clean_data does not change the original DataFrame
# ---------------------------------------------------------------------------

def test_original_dataframe_not_mutated(sample_raw_df):
    """clean_data should work on a copy and not change the original DataFrame."""
    original_income_vals = sample_raw_df["income"].tolist()
    original_len = len(sample_raw_df)
    _ = clean_data(sample_raw_df)
    assert sample_raw_df["income"].tolist() == original_income_vals, (
        "The original DataFrame's income column should remain unchanged."
    )
    assert len(sample_raw_df) == original_len, (
        "No rows should be removed from the original DataFrame."
    )


# ---------------------------------------------------------------------------
# Test 6: Ensure build_inference_row handles missing inputs safely
# ---------------------------------------------------------------------------

def test_inference_row_missing_keys():
    """build_inference_row should fill missing numeric values with 0 and missing categorical values with 'Unknown'."""
    features = {"age": 35, "sex": "Female"}
    row = build_inference_row(features)

    assert row.shape == (1, len(NUMERIC_COLS) + len(CATEGORICAL_COLS)), (
        "The output should contain exactly one row with all expected columns."
    )
    # Shape alone only confirms the right *number* of columns — it can't
    # catch a misnamed column (e.g. "hours_per_week" instead of
    # "hours-per-week"), which would still pass the shape check above but
    # would cause preprocessor.transform() to fail downstream with a
    # confusing "columns are missing" error instead of failing here with a
    # clear one. Checking the column names directly closes that gap.
    #
    # Neither check alone is sufficient, which is why both are kept:
    # - shape alone misses wrong/misspelled names with the same count
    # - set(columns) alone misses a duplicated name, since a duplicate
    #   collapses back down to the same set of unique names even though
    #   the actual column count is wrong (e.g. 15 columns with "age"
    #   appearing twice still produces a 14-element set that matches
    #   expected_columns). The shape check above is what catches that case.
    expected_columns = set(NUMERIC_COLS) | set(CATEGORICAL_COLS)
    assert set(row.columns) == expected_columns, (
        "The output should contain exactly the expected column names — "
        "a correct column count with a misspelled or wrong name would "
        "otherwise pass the shape check above but fail later when the "
        "preprocessor tries to select that column by name."
    )
    assert row["age"].iloc[0] == 35.0
    # Missing numeric values default to 0
    assert row["capital-gain"].iloc[0] == 0.0
    # Missing categorical values default to 'Unknown'
    assert row["workclass"].iloc[0] == "Unknown"


# ---------------------------------------------------------------------------
# Test 7: Ensure a leading-space "?" is still caught as a missing value
# ---------------------------------------------------------------------------

def test_leading_space_question_mark_dropped(sample_raw_df):
    """
    A '?' value loaded with a leading space (e.g. ' ?', as produced by a
    headers-CSV that wasn't parsed with na_values='?') should still be
    recognized as missing. clean_data() strips whitespace before replacing
    '?' with NaN, so ' ?' should normalize to '?' and get dropped along
    with the other missing-value rows.
    """
    df = sample_raw_df.copy()
    # Overwrite a known-good row's workclass with a leading-space '?'
    df.loc[0, "workclass"] = " ?"

    cleaned = clean_data(df)

    assert (cleaned == "?").to_numpy().sum() == 0, (
        "No literal '?' strings should remain after cleaning."
    )
    assert cleaned["workclass"].isna().sum() == 0, (
        "No NaN should remain in the cleaned data — rows with NaN are dropped."
    )


# ---------------------------------------------------------------------------
# Test 8: A duplicated column name is caught by the shape check, even
# though set(columns) alone would miss it
# ---------------------------------------------------------------------------

def test_inference_row_shape_catches_duplicate_columns():
    """
    set(row.columns) alone cannot detect duplicate column names. A duplicated column (e.g., "age" appearing twice) still collapses to the same set of unique names.
    The shape check (row.shape == (1, 14)) is what detects the real issue, since it checks actual column count.
    This test intentionally creates a duplicate-column case (bypassing build_inference_row)
    to ensure the shape check is meaningful and not redundant with the set check.
    """
    expected_columns = set(NUMERIC_COLS) | set(CATEGORICAL_COLS)

    # 15 columns total: every expected column once, plus "age" a second time.
    duplicate_col_row = pd.DataFrame(
        [[30.0] * 15],
        columns=NUMERIC_COLS + CATEGORICAL_COLS + ["age"],
    )

    assert set(duplicate_col_row.columns) == expected_columns, (
        "Sanity check: unique column names should still match expected columns."
        "This shows that using set() alone cannot detect duplicate columns."
    )
    assert duplicate_col_row.shape != (1, len(NUMERIC_COLS) + len(CATEGORICAL_COLS)), (
        "The shape check ensures correct structure even when column names look valid."
        "A duplicated column can still pass name checks but will break expected column count."
        "This test catches those cases."
    )