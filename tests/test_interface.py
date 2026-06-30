"""
tests/test_interface.py — Interface tests for feature parsing and edge cases.

These tests mock the LLM so no API key is needed.

Run:  pytest tests -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from preprocess import build_preprocessor


# ---------------------------------------------------------------------------
# Helpers — mock LLM responses
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str):
    """Create a fake response that matches the LLM output format."""
    mock_choice  = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ---------------------------------------------------------------------------
# Test 1: Properly formatted LLM output
# ---------------------------------------------------------------------------

def test_parse_features_well_formed():
    """
    Simulate valid LLM JSON output and check it is parsed correctly.
    """
    expected = {
        "age": 42,
        "occupation": "Prof-specialty",
        "hours-per-week": 45,
        "workclass": "Private",
        "education": "Bachelors",
        "sex": "Male",
    }
    llm_json = json.dumps(expected)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_llm_response(llm_json)

    from app import parse_features_with_llm
    result = parse_features_with_llm(mock_client, "I'm a 42-year-old male software engineer.")

    assert result.get("age") == 42
    assert result.get("occupation") == "Prof-specialty"
    assert result.get("hours-per-week") == 45
    assert "error" not in result


# ---------------------------------------------------------------------------
# Test 2: Missing or vague input handling
# ---------------------------------------------------------------------------

def test_parse_features_incomplete_input():
    """
    Vague input should return an error response with a message.
    """
    error_payload = {
        "error": "need_more_info",
        "message": "Could you tell me your age and occupation?",
    }
    llm_json = json.dumps(error_payload)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_llm_response(llm_json)

    from app import parse_features_with_llm
    result = parse_features_with_llm(mock_client, "hello")

    assert "error" in result, "Expected error for vague input"
    assert "message" in result, "Error should include a message"


def test_parse_features_out_of_scope_input():
    """
    A genuinely off-topic query (not just vague — actually unrelated to
    income prediction) should still return the same error shape as a
    vague query, but with a message that acknowledges what was actually
    asked rather than a generic "give me more details" prompt. This is
    the out-of-scope case the project's edge-case-handling requirement
    asks for, distinct from test_parse_features_incomplete_input's
    merely-vague case ("hello").
    """
    error_payload = {
        "error": "need_more_info",
        "message": (
            "I can only help predict income based on demographic and "
            "employment details — I don't have access to weather "
            "information. Want to tell me a bit about your age, job, "
            "or education instead?"
        ),
    }
    llm_json = json.dumps(error_payload)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_llm_response(llm_json)

    from app import parse_features_with_llm
    result = parse_features_with_llm(mock_client, "what's the weather like today?")

    assert "error" in result, "Expected error for an out-of-scope query"
    assert "message" in result
    assert len(result["message"]) > 0, "Message should not be empty"


# ---------------------------------------------------------------------------
# Test 3: Markdown-wrapped JSON handling
# ---------------------------------------------------------------------------

def test_parse_features_markdown_fences():
    """Parser should handle ```json ... ``` blocks correctly."""
    expected = {"age": 55, "sex": "Female", "workclass": "Federal-gov"}
    llm_json = f"```json\n{json.dumps(expected)}\n```"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_llm_response(llm_json)

    from app import parse_features_with_llm
    result = parse_features_with_llm(mock_client, "55-year-old woman, federal government worker.")

    assert result.get("age") == 55
    assert result.get("sex") == "Female"
    assert "error" not in result


# ---------------------------------------------------------------------------
# Test 4: main()'s error-detection logic (is_error_response)
# ---------------------------------------------------------------------------

def test_is_error_response_checks_presence_not_truthiness():
    """
    main() decides whether to treat an LLM response as an error by calling
    is_error_response(parsed), which does `return "error" in parsed`. This
    test calls that function directly — not main() itself, which still
    requires a Streamlit runtime — so it genuinely exercises the same
    branching logic main() relies on, rather than just describing it.

    A falsy-but-present "error" key (None, False, 0, "") should still
    count as an error, since `in`/`not in` on a dict checks key presence,
    never the value's truthiness. If is_error_response() were ever changed
    to something like `return bool(parsed.get("error"))`, this test would
    fail, because that version would treat {"error": None} as NOT an
    error — silently inverting the intended behavior.
    """
    from app import is_error_response

    # Falsy-but-present error key: should still be treated as an error.
    assert is_error_response({"age": 42, "error": None}) is True
    assert is_error_response({"age": 42, "error": False}) is True
    assert is_error_response({"age": 42, "error": ""}) is True

    # No error key at all: should not be treated as an error.
    assert is_error_response({"age": 42}) is False

    # Real-world shaped payloads, mirroring what parse_features_with_llm
    # actually returns for the "need more info" and "parse failed" cases.
    assert is_error_response(
        {"error": "need_more_info", "message": "Could you tell me your age?"}
    ) is True


def test_parse_features_with_llm_preserves_falsy_error_value():
    """
    Confirms parse_features_with_llm() round-trips a falsy "error" value
    (e.g. {"age": 42, "error": None}) faithfully through json.loads(),
    rather than dropping or coercing it. This is a narrower check than
    test_is_error_response_checks_presence_not_truthiness() above — it's
    about the JSON parsing step, not the downstream branching decision.
    """
    falsy_error_payload = {"age": 42, "error": None}
    llm_json = json.dumps(falsy_error_payload)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_llm_response(llm_json)

    from app import parse_features_with_llm
    result = parse_features_with_llm(mock_client, "I'm 42.")

    # "error" in result is True here even though its value is falsy (None) —
    # correct dict semantics, consistent with what is_error_response()
    # checks above (which is what main() actually calls).
    assert "error" in result
    assert result["error"] is None
    assert result.get("age") == 42, (
        "The age value should still be present in the parsed dict — "
        "JSON round-tripping shouldn't drop or alter other keys."
    )


# ---------------------------------------------------------------------------
# Test 5: Model output validation
# ---------------------------------------------------------------------------

def test_run_model_output_types():
    """
    run_model should return:
    - a label (0 or 1)
    - a probability between 0 and 1
    """
    from sklearn.ensemble import RandomForestClassifier

    # Create small synthetic dataset
    rng = np.random.default_rng(1)
    n = 100
    X = pd.DataFrame({
        "age":            rng.integers(20, 60, n).astype(float),
        "fnlwgt":         rng.integers(50000, 500000, n).astype(float),
        "education-num":  rng.integers(5, 16, n).astype(float),
        "capital-gain":   rng.integers(0, 5000, n).astype(float),
        "capital-loss":   rng.integers(0, 500, n).astype(float),
        "hours-per-week": rng.integers(20, 60, n).astype(float),
        "workclass":      rng.choice(["Private", "Self-emp-not-inc"], n),
        "education":      rng.choice(["Bachelors", "HS-grad"], n),
        "marital-status": rng.choice(["Married-civ-spouse", "Never-married"], n),
        "occupation":     rng.choice(["Prof-specialty", "Adm-clerical"], n),
        "relationship":   rng.choice(["Husband", "Not-in-family"], n),
        "race":           rng.choice(["White", "Black"], n),
        "sex":            rng.choice(["Male", "Female"], n),
        "native-country": rng.choice(["United-States", "Mexico"], n),
    })
    y = rng.integers(0, 2, n)

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(preprocessor.transform(X), y)

    # Deliberately partial: only 5 of the 14 columns the preprocessor was
    # fit on are provided here, mimicking a real LLM extraction where most
    # fields go unmentioned. run_model() internally calls build_inference_row(),
    # which fills the other 9 columns with defaults (0 for numerics, "Unknown"
    # for categoricals) before the row reaches the preprocessor. This test
    # passing depends on that defaulting actually happening — without it,
    # preprocessor.transform() would raise a ValueError for the missing
    # columns instead of returning a usable prediction.
    features = {
        "age": 35,
        "education": "Bachelors",
        "sex": "Male",
        "workclass": "Private",
        "hours-per-week": 40,
    }

    from app import run_model
    label, prob = run_model(features, model, preprocessor)

    assert label in (0, 1), f"Label must be 0 or 1, got {label}"
    assert 0.0 <= prob <= 1.0, f"Probability must be in [0, 1], got {prob}"