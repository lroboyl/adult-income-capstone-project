"""
app.py

Purpose
-------
This file creates the Streamlit web application for the Adult Income Predictor.

The application combines a machine learning model with a Large Language Model
(LLM) so users can interact with the model using plain English instead of
filling out a form.

How the App Works
------------------
1. The user describes themselves in everyday language.
2. The LLM extracts the information needed by the machine learning model.
3. The trained model predicts whether the person's income is likely to be
   above or below $50,000 per year.
4. The LLM explains the prediction in clear, easy-to-understand language.

Main Parts of the File
-----------------------
Configuration
    - Loads API keys from the .env file.
    - Sets the Nebius AI model and application settings.

Load the Model
    - Loads the best trained machine learning model.
    - Loads the saved preprocessing pipeline.

Parse User Input
    - Sends the user's message to the LLM.
    - Extracts values like age, education, occupation, and work hours.
    - Returns the information as structured data.

Run the Prediction
    - Converts the extracted information into the correct format.
    - Applies the same preprocessing used during training.
    - Uses the trained model to make a prediction.

Explain the Prediction
    - Sends the prediction and feature values back to the LLM.
    - Generates a simple explanation of the result.
    - Includes a reminder that the prediction is only an estimate.

Streamlit Interface
    The web app provides:
    - A chat interface
    - Example prompts
    - Prediction results
    - An explanation of the prediction
    - A sidebar with project information and required input features

Running the App
----------------
    streamlit run src/app.py

This opens the application in your web browser, where users can describe
themselves and receive an income prediction with an explanation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, NotFoundError  # Nebius uses the OpenAI-compatible SDK

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    USER_FACING_FEATURES,
    build_inference_row,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# Loads API keys from the .env file and sets the Nebius AI model and
# application settings.
# ---------------------------------------------------------------------------

NEBIUS_API_KEY  = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
# No safe universal default here — model availability and naming on Nebius
# AI Studio changes over time, so this must be set explicitly in .env to
# match a model currently listed in your Nebius account's catalog
# (https://studio.nebius.com/ -> Models). The fallback below is only a
# last resort to avoid a crash on an unset variable; it is NOT guaranteed
# to be a valid model name and should not be relied on.
NEBIUS_MODEL    = os.getenv("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

MODELS_DIR = Path("models")

# ---------------------------------------------------------------------------
# Nebius / LLM client
# ---------------------------------------------------------------------------

@st.cache_resource
def get_llm_client() -> OpenAI:
    if not NEBIUS_API_KEY:
        st.error(
            "**NEBIUS_API_KEY not found.**  "
            "Create a `.env` file based on `.env.example` and add your key."
        )
        st.stop()
    return OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)


# ---------------------------------------------------------------------------
# Load the Model
# Loads the best trained machine learning model and the saved preprocessing
# pipeline.
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_and_preprocessor():
    """Load the best trained model and the fitted preprocessor."""
    # Determine which model to load
    best_model_file = MODELS_DIR / "best_model.txt"
    if best_model_file.exists():
        best_key = best_model_file.read_text().strip()
    else:
        # Fall back to scanning for any .joblib that isn't the preprocessor
        candidates = [p for p in MODELS_DIR.glob("*.joblib")
                      if p.name != "preprocessor.joblib"]
        if not candidates:
            return None, None
        best_key = candidates[0].stem

    model_path = MODELS_DIR / f"{best_key}.joblib"
    prep_path  = MODELS_DIR / "preprocessor.joblib"

    if not model_path.exists() or not prep_path.exists():
        return None, None

    model       = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    return model, preprocessor


# ---------------------------------------------------------------------------
# Parse User Input
# Sends the user's message to the LLM, extracts values like age, education,
# occupation, and work hours, and returns the information as structured data.
# ---------------------------------------------------------------------------

PARSE_SYSTEM_PROMPT = """
You are a feature-extraction assistant for an income prediction model trained
on US Census data. Your ONLY job is to parse a user's natural language message
and extract the values for the features listed below.

Features the model needs (return ONLY those that are clearly stated or
strongly implied by the user — do NOT invent values):

Numeric:
- age            (integer, years)
- fnlwgt         (integer, census weight — rarely provided, omit if absent)
- education-num  (integer 1-16; map: HS-grad≈9, Bachelors≈13, Masters≈14, Doctorate≈16)
- capital-gain   (integer, USD)
- capital-loss   (integer, USD)
- hours-per-week (integer)

Categorical (return the exact category name from the list):
- workclass:      Private | Self-emp-not-inc | Self-emp-inc | Federal-gov |
                  Local-gov | State-gov | Without-pay | Never-worked
- education:      Bachelors | Some-college | 11th | HS-grad | Prof-school |
                  Assoc-acdm | Assoc-voc | 9th | 7th-8th | 12th | Masters |
                  1st-4th | 10th | Doctorate | 5th-6th | Preschool
- marital-status: Married-civ-spouse | Divorced | Never-married | Separated |
                  Widowed | Married-spouse-absent | Married-AF-spouse
- occupation:     Tech-support | Craft-repair | Other-service | Sales |
                  Exec-managerial | Prof-specialty | Handlers-cleaners |
                  Machine-op-inspct | Adm-clerical | Farming-fishing |
                  Transport-moving | Priv-house-serv | Protective-serv |
                  Armed-Forces
- relationship:   Wife | Own-child | Husband | Not-in-family |
                  Other-relative | Unmarried
- race:           White | Asian-Pac-Islander | Amer-Indian-Eskimo |
                  Other | Black
- sex:            Male | Female
- native-country: United-States | Cuba | Jamaica | India | Mexico | South |
                  Japan | Philippines | Germany | Canada | ... (use closest match)

Return a JSON object with ONLY the features you can confidently extract.
Do NOT include a "confidence" key. Do NOT add comments.
Do NOT return features you are not sure about.
If you cannot extract at least age OR occupation from the message, return:
{"error": "need_more_info", "message": "<ask the user for the key missing details>"}

Examples:
  Input:  "I'm a 42-year-old software engineer working 45 hours a week for a private company."
  Output: {"age": 42, "occupation": "Prof-specialty", "hours-per-week": 45, "workclass": "Private"}

  Input:  "hello"
  Output: {"error": "need_more_info", "message": "To predict your income level, I need some basic information. Could you share your age, occupation, and work situation?"}

  Input:  "what's the weather like today?"
  Output: {"error": "need_more_info", "message": "I can only help predict income based on demographic and employment details — I don't have access to weather information. Want to tell me a bit about your age, job, or education instead?"}
"""


def parse_features_with_llm(client: OpenAI, user_message: str) -> Dict[str, Any]:
    """Call the LLM to extract feature values from the user's message."""
    try:
        response = client.chat.completions.create(
            model=NEBIUS_MODEL,
            messages=[
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )
    except NotFoundError:
        st.error(
            f"**Model not found: `{NEBIUS_MODEL}`**\n\n"
            "This usually means NEBIUS_MODEL in your `.env` doesn't match "
            "a model currently available in your Nebius AI Studio account. "
            "Check the model catalog at https://studio.nebius.com/ and "
            "update NEBIUS_MODEL in `.env` to the exact name listed there."
        )
        st.stop()
        # st.stop() halts the Streamlit script when running normally, but
        # doesn't raise an exception Python itself recognizes — this
        # explicit return prevents falling through to use `response` below,
        # which would otherwise be unbound after the except block runs.
        return {"error": "model_not_found",
                "message": f"Model '{NEBIUS_MODEL}' is not available. "
                           "Please check your Nebius configuration."}
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON: %s", raw)
        return {"error": "parse_failed",
                "message": "I had trouble understanding that. Could you rephrase?"}


def is_error_response(parsed: Dict[str, Any]) -> bool:
    """
    Return True if a parsed LLM response should be treated as an error
    rather than a usable set of extracted features.

    This is a key-presence check, not a truthiness check: any "error" key
    counts, regardless of what value it holds (including None or False).
    That matches the system prompt's documented contract, where parse
    failures and "need more info" responses always set "error" to a
    non-empty string — but this function doesn't depend on that contract
    being followed correctly by the LLM. It only checks for the key.

    Pulled out as its own function (rather than an inline `if "error" in
    parsed:` inside main()) specifically so this branching logic can be
    unit tested directly, without needing to mock Streamlit's runtime.
    """
    return "error" in parsed


# ---------------------------------------------------------------------------
# Run the Prediction
# Converts the extracted information into the correct format, applies the
# same preprocessing used during training, and uses the trained model to
# make a prediction.
# ---------------------------------------------------------------------------

def run_model(features: Dict[str, Any], model, preprocessor) -> Tuple[int, float]:
    """Run the trained model on parsed features. Returns (label, probability)."""
    row = build_inference_row(features)
    row_t = preprocessor.transform(row)
    label = int(model.predict(row_t)[0])
    prob  = float(model.predict_proba(row_t)[0][1])
    return label, prob


# ---------------------------------------------------------------------------
# Explain the Prediction
# Sends the prediction and feature values back to the LLM, generates a
# simple explanation of the result, and includes a reminder that the
# prediction is only an estimate.
# ---------------------------------------------------------------------------

EXPLAIN_SYSTEM_PROMPT = """
You are a helpful income-prediction assistant. Given a model's prediction and the
features that were used, explain the result in a clear, empathetic, and
conversational way.

Guidelines:
- State the prediction plainly (>$50K or ≤$50K).
- Give the probability as a percentage and explain what it means.
- Mention 2-3 features that most likely drove the prediction (use domain knowledge).
- Include a relevant caveat: the model is based on 1990s US Census data, so it
  reflects historical patterns and should not be used for legal or financial decisions.
- Keep the tone supportive, never judgmental.
- Format with a short header, then 2-3 brief paragraphs. No bullet walls.
"""


def generate_explanation(client: OpenAI, features: Dict[str, Any],
                         label: int, prob: float) -> str:
    """
    Ask the LLM to narrate the model prediction.

    By the time this is called, run_model() has already succeeded and a
    valid prediction exists — so unlike parse_features_with_llm(), a
    failure here shouldn't halt the app via st.stop() (that would discard
    a prediction the user is entitled to see). Instead this returns a
    plain-text fallback explanation, so the worst case is a less detailed
    response rather than a lost result or a raw traceback.
    """
    income_label = ">$50K" if label == 1 else "≤$50K"
    context = (
        f"Model prediction: {income_label} (probability of earning >$50K: {prob:.1%})\n"
        f"Features used: {json.dumps(features, indent=2)}"
    )
    try:
        response = client.chat.completions.create(
            model=NEBIUS_MODEL,
            messages=[
                {"role": "system",  "content": EXPLAIN_SYSTEM_PROMPT},
                {"role": "user",    "content": context},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except NotFoundError:
        logger.error("Explanation generation failed: model '%s' not found", NEBIUS_MODEL)
        return (
            f"**Prediction: {income_label}** ({prob:.1%} probability of earning >$50K)\n\n"
            f"_A detailed explanation couldn't be generated because the configured "
            f"model (`{NEBIUS_MODEL}`) isn't available. Check NEBIUS_MODEL in your "
            f".env file against your Nebius AI Studio model catalog._"
        )


def escape_markdown_dollars(text: str) -> str:
    """
    Escape literal "$" characters so Streamlit's markdown renderer doesn't
    mistake them for LaTeX/KaTeX math-mode delimiters.

    Streamlit's st.markdown() (and st.write()) treats a "$" as the start
    of an inline math expression and looks for a closing "$" to render
    everything in between as LaTeX. Since this app's prompts explicitly
    ask the LLM to state amounts like ">$50K" or "$50,000" (see
    EXPLAIN_SYSTEM_PROMPT), the generated explanation text is guaranteed
    to contain dollar signs — without escaping, two or more "$" anywhere
    in the text causes everything between them to render as a garbled
    math expression instead of plain text. Streamlit has no parameter to
    disable this (checked: st.markdown's signature has no such option),
    so escaping the text itself before rendering is the fix.
    """
    return text.replace("$", r"\$")


# ---------------------------------------------------------------------------
# Streamlit Interface
# Builds the web app: a chat interface, example prompts, prediction
# results, an explanation of the prediction, and a sidebar with project
# information and required input features.
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Income Predictor",
        page_icon="💼",
        layout="centered",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .title-block {
        background: linear-gradient(135deg, #1a1f36 0%, #2d3561 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem 1.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .title-block h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        margin: 0 0 0.4rem;
        color: white;
    }
    .title-block p { color: #a8b2d8; margin: 0; font-size: 0.95rem; }

    .prediction-box {
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        font-size: 1.05rem;
        font-weight: 500;
    }
    .pred-high { background: #ecfdf5; border-left: 4px solid #10b981; color: #065f46; }
    .pred-low  { background: #fef3c7; border-left: 4px solid #f59e0b; color: #78350f; }

    .feature-pill {
        display: inline-block;
        background: #f0f4ff;
        color: #3730a3;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="title-block">
        <h1>💼 Income Predictor</h1>
        <p>Describe yourself in plain English — the model will estimate whether your income is above or below $50K/year.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    # Rendered before any st.stop() call below, so it's still visible even
    # when the app halts early (e.g. no trained model, LLM parse failure,
    # or inference error).
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "This app combines a machine learning model trained on the "
            "[UCI Adult Income dataset](https://archive.ics.uci.edu/dataset/2/adult) "
            "with an LLM-powered natural language interface.\n\n"
            "**Model:** Best-performing classifier selected from 5 MLflow runs.\n\n"
            f"**LLM:** Nebius AI Studio (`{NEBIUS_MODEL}`) for feature parsing and explanation."
        )
        st.caption(
            "Your conversation lives only in this browser tab — "
            "refreshing the page or closing the tab clears it."
        )
        st.markdown("---")
        st.markdown("### Required Features")
        st.markdown(
            "The more of these you mention, the more accurate your "
            "prediction will be — but you don't need all of them. "
            "Anything left out is filled in with a sensible default."
        )
        feature_info = {
            "age": "Your current age",
            "workclass": "Employment sector (e.g. private, government, self-employed)",
            "education": "Highest level completed",
            "marital-status": "Marital status",
            "occupation": "Your job type",
            "relationship": "Household relationship (e.g. husband, unmarried)",
            "race": "Race",
            "sex": "Gender",
            "hours-per-week": "Typical weekly hours",
            "capital-gain": "Investment gains, if any",
            "capital-loss": "Investment losses, if any",
            "native-country": "Country of origin",
        }
        # feature_info's keys should always match USER_FACING_FEATURES —
        # this assertion catches the two lists drifting apart again, the
        # way the previous hand-maintained sidebar list silently did.
        assert set(feature_info.keys()) == set(USER_FACING_FEATURES), (
            "Sidebar feature_info is out of sync with USER_FACING_FEATURES"
        )
        for feat, desc in feature_info.items():
            st.markdown(f"- **{feat}**: {desc}")

        st.markdown("---")
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.last_features = {}
            st.rerun()

    # ── Load model ────────────────────────────────────────────────────────
    model, preprocessor = load_model_and_preprocessor()
    if model is None:
        st.warning(
            "**No trained model found.**  "
            "Run `python src/train.py` first to train and save a model."
        )
        st.stop()

    client = get_llm_client()

    # ── Session state ─────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_features" not in st.session_state:
        st.session_state.last_features = {}

    # ── Example prompts ───────────────────────────────────────────────────
    st.markdown("**Try an example:**")
    examples = [
        "I'm a 38-year-old married male software engineer, Bachelor's degree, working 50 hrs/week for a private company.",
        "28-year-old single woman, high school graduate, works in retail, about 35 hours a week.",
        "55-year-old male executive, Master's degree, self-employed, 60+ hours per week.",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i].button(f"Example {i+1}", use_container_width=True, help=ex):
            st.session_state._inject_example = ex

    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────
    # Messages that contain raw HTML (the prediction box, feature pills)
    # are tagged with "html": True when stored, so they render the same
    # way here on replay as they did the first time — without this, a
    # message rendered with unsafe_allow_html=True initially would show
    # up as literal, unescaped tag text once it moved into history and
    # got rendered through this loop instead.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=msg.get("html", False))

    # ── User input ────────────────────────────────────────────────────────
    injected = st.session_state.pop("_inject_example", None)
    user_input = st.chat_input("Describe yourself (age, job, education, hours worked…)")

    if injected:
        user_input = injected

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── Step 1: Parse features ───────────────────────────────────────
        with st.spinner("Parsing your information…"):
            parsed = parse_features_with_llm(client, user_input)

        if is_error_response(parsed):
            # LLM couldn't extract enough info — relay its clarifying question
            bot_reply = parsed.get("message", "Could you provide more details?")
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
            st.stop()

        # ── Step 2: Run model ────────────────────────────────────────────
        try:
            label, prob = run_model(parsed, model, preprocessor)
        except Exception as exc:
            logger.error("Model inference failed: %s", exc, exc_info=True)
            st.error(f"Model error: {exc}")
            st.stop()

        # ── Step 3: Show prediction badge ────────────────────────────────
        pred_class  = "pred-high" if label == 1 else "pred-low"
        pred_label  = ">$50K / year" if label == 1 else "≤$50K / year"
        pred_pct    = f"{prob:.1%} probability of earning >$50K"

        # ── Step 4: Show parsed features ────────────────────────────────
        pills = " ".join(
            f'<span class="feature-pill">{k}: {v}</span>'
            for k, v in parsed.items()
        )

        # ── Step 5: Generate LLM explanation ────────────────────────────
        with st.spinner("Generating explanation…"):
            explanation = generate_explanation(client, parsed, label, prob)
            # Escape $ before this text ever reaches st.markdown(), so the
            # prediction commentary (which always contains $-amounts, per
            # EXPLAIN_SYSTEM_PROMPT) doesn't get parsed as LaTeX.
            explanation = escape_markdown_dollars(explanation)

        # ── Assemble bot reply ───────────────────────────────────────────
        full_reply = (
            f'<div class="prediction-box {pred_class}">'
            f'  <strong>Predicted income range: {pred_label}</strong><br>'
            f'  <span style="font-weight:400; font-size:0.9rem;">{pred_pct}</span>'
            f'</div>'
            f'<p style="color:#6b7280; font-size:0.83rem; margin:0.5rem 0 1rem;">'
            f'  <strong>Features extracted:</strong> {pills}'
            f'</p>'
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": full_reply, "html": True}
        )
        st.session_state.messages.append({"role": "assistant", "content": explanation})
        st.session_state.last_features = parsed

        with st.chat_message("assistant"):
            st.markdown(full_reply, unsafe_allow_html=True)
            st.markdown(explanation)


if __name__ == "__main__":
    main()