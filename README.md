# Adult Income Predictor

Just describe yourself in plain English — *"I'm a 38-year-old software engineer with a Bachelor's degree, working 50 hours a week for a private company"* — and this app predicts whether your income is likely to be above or below $50K/year, then explains why.

Under the hood, an LLM pulls the relevant details out of your message and passes them to a machine learning model trained on US Census data. No forms, no dropdowns, just a conversation.

---

## Table of Contents

1. [How It Works & Architecture](#how-it-works--architecture)
2. [Setup](#setup)
3. [Running the App](#running-the-app)
4. [Results](#results)
5. [Repository Structure](#repository-structure)
6. [Reflection](#reflection)

---

## How It Works & Architecture

```
Your message  →  LLM extracts features  →  Preprocessing  →  ML model  →  Prediction  →  LLM explains
```

A few things worth knowing about the design:

- **The LLM is used twice** — once to parse your message into structured features, and once to explain the prediction in plain English. Splitting these into two separate calls makes each one easier to test and debug independently.
- **The preprocessing pipeline is saved alongside the model** so the exact same transformations applied during training are applied at inference time. No drift, no surprises.
- **MLflow tracks every training run** and automatically picks the best model. You can browse all five runs visually in the MLflow UI after training.

The reasoning behind specific preprocessing choices — why `StandardScaler` over `MinMaxScaler`, how missing values are handled, why ROC-AUC matters more than accuracy here — is documented in `notebooks/exploration.ipynb`.

---

## Setup

**You'll need:** Python 3.10 or later (tested up to 3.13), and a [Nebius AI Studio](https://studio.nebius.com/) account with an API key.

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/adult-income-predictor.git
cd adult-income-predictor
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Dependencies are pinned to versions verified in June 2026. They deliberately stay on the last stable minor before each library's most recent major bump (e.g. pandas 2.3.3, mlflow 2.22.5) to avoid unverified breaking changes. Upgrade freely once you've tested your environment.

### 2. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and fill in your Nebius API key and model name:

```
NEBIUS_API_KEY=your_api_key_here
NEBIUS_MODEL=meta-llama/Llama-3.3-70B-Instruct
```

> **Important:** `NEBIUS_MODEL` must exactly match a model name in your Nebius account's catalog — go to [studio.nebius.com](https://studio.nebius.com/) → Models and copy the string directly. Even small differences in casing or punctuation will cause a 404 error.

### 3. Download the dataset

Download the Adult Income dataset from https://archive.ics.uci.edu/dataset/2/adult.

Clicking **Download** gives you a zip file. Inside it, you'll find a file called `adult.data` (not `adult.csv`). Copy and rename it:

```bash
cp ~/Downloads/adult/adult.data data/adult.csv
```

The loader handles the original UCI format automatically — no other changes needed.

### 4. Train the models

```bash
python src/train.py
```

This trains five model configurations, logs everything to MLflow, and saves the best-performing model to `models/`. To browse the results visually:

```bash
mlflow ui --backend-store-uri mlruns
# then open http://localhost:5000
```

> **Mac users:** port 5000 is taken by AirPlay Receiver on macOS Monterey and later. Use a different port instead:
> ```bash
> mlflow ui --backend-store-uri mlruns --port 5001
> # then open http://127.0.0.1:5001
> ```

### 5. Run the tests

```bash
pytest tests/ -v
```

All 18 tests pass without needing a trained model or an API key. If something fails:

- **`ModuleNotFoundError`** — run `pip install -r requirements.txt`
- **`ImportError` for `app.py`** — the `openai` package specifically is missing; run `pip install openai`
- **Path errors** — make sure you're running pytest from the project root, not from inside `src/` or `tests/`

### 6. (Optional) Run with Docker

```bash
docker build -t adult-income-predictor .
docker run --rm -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  adult-income-predictor
```

The `-v` flag mounts your local `models/` directory into the container — the trained model isn't baked into the image, so this step is required.

---

## Running the App

```bash
streamlit run src/app.py
# then open http://localhost:8501
```

Try something like:

- *"I'm a 45-year-old married man with a Master's degree, working as an executive for 55 hours a week."*
- *"28 years old, single, retail sales, about 25 hours a week."*

The app extracts whatever features it can from your description, fills in sensible defaults for anything you didn't mention, runs the prediction, and explains the result.

### What happens if you're vague or off-topic

The app is designed to handle these gracefully rather than crash or give a confusing answer.

**Not enough to go on:**
> *"I work in finance."*

The app asks: *"To predict your income level, I need a bit more. Could you tell me your age, how many hours a week you typically work, and whether you're employed by a company or self-employed?"*

**Something unrelated:**
> *"What's the weather like today?"*

The app redirects: *"I can only help predict income based on demographic and employment details — I don't have access to weather information. Want to tell me a bit about your age, job, or education instead?"*

In both cases, the chat stays open and you can reply immediately.

---

## Results

Five models were trained and compared on the [UCI Adult Income dataset](https://archive.ics.uci.edu/dataset/2/adult) (~30,000 rows after cleaning). The winner was selected by **ROC-AUC** rather than accuracy, because only about 24% of the dataset earns above $50K — accuracy alone would be misleading.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | ~0.845 | ~0.741 | ~0.607 | ~0.668 | ~0.905 |
| Random Forest | ~0.864 | ~0.782 | ~0.637 | ~0.702 | ~0.924 |
| Random Forest (Shallow) | ~0.855 | ~0.756 | ~0.638 | ~0.692 | ~0.916 |
| Gradient Boosting | ~0.872 | ~0.796 | ~0.657 | ~0.720 | ~0.931 |
| **XGBoost** ✓ | **~0.875** | **~0.806** | **~0.661** | **~0.726** | **~0.935** |

**XGBoost** came out on top. All runs use `random_state: 42` throughout, so results are fully reproducible — the only thing that could shift the numbers is upgrading a library to a version that changes an algorithm's internals.

A few things the results confirmed: tree-based models clearly outperform logistic regression here (the relationship between features and income is non-linear), and the features that matter most are age, education, occupation, and hours worked per week.

---

## Repository Structure

```
adult-income-predictor/
├── configs/
│   └── config.yaml          ← all hyperparameters and settings live here
├── src/
│   ├── preprocess.py        ← cleaning, encoding, and the preprocessing pipeline
│   ├── train.py             ← trains 5 models, logs to MLflow, saves the best one
│   ├── evaluate.py          ← metrics and best-run selection
│   └── app.py               ← the Streamlit app
├── tests/
│   ├── conftest.py
│   ├── test_preprocess.py   ← 8 tests: cleaning, encoding, inference row building
│   ├── test_model.py        ← 3 tests: prediction shape, accuracy floor, probabilities
│   └── test_interface.py    ← 7 tests: LLM parsing, edge cases, model output types
├── notebooks/
│   └── exploration.ipynb    ← EDA and preprocessing rationale
├── data/
│   └── .gitkeep
├── Dockerfile
├── .env.example
└── requirements.txt
```

### Configuring training

Everything `train.py` does is controlled by `configs/config.yaml` — nothing is hardcoded in the Python source. To tune a model, change a value here and re-run training:

```yaml
data:
  path: data/adult.csv
  test_size: 0.2             # 80/20 split
  random_state: 42

preprocessing:
  numeric_columns:           # scaled with StandardScaler
    - age
    - fnlwgt
    - education-num
    - capital-gain
    - capital-loss
    - hours-per-week
  categorical_columns:       # one-hot encoded
    - workclass
    - education
    - marital-status
    - occupation
    - relationship
    - race
    - sex
    - native-country

mlflow:
  experiment_name: adult_income_predictor
  tracking_uri: mlruns

models:
  xgboost:
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 6
    eval_metric: logloss
    random_state: 42
  # ... plus logistic_regression, random_forest_default,
  #     random_forest_shallow, gradient_boosting
```

To add a new model entirely, add a new key under `models:` and add the matching branch in `build_model()` in `src/train.py`.

---

## Reflection

This project started as a way to see how well an LLM-powered interface could replace a structured form for a machine learning model. The short answer: pretty well, as long as you're careful about what happens when the LLM doesn't extract clean data.

The trickiest part wasn't the ML — it was building the interface robustly. The LLM doesn't always return what you expect, which means defensive parsing, clear fallback behavior, and tests that cover the awkward cases (missing features, malformed JSON, model not found errors) matter a lot more than they might seem at first.

A few specific things I learned along the way:

- **Separating concerns pays off.** Splitting the LLM into a feature-extraction call and an explanation call meant I could test each independently without a live API, and debug problems much more quickly.
- **Prompt engineering is more constrained than it looks.** Getting the LLM to reliably return JSON — with the right keys, the right values, no markdown fences — took more iteration than expected. The examples in the system prompt made a measurable difference.
- **MLflow is worth the setup overhead.** Being able to compare all five models in a single table, and have the best one selected automatically, made the training loop much cleaner to reason about.

Things I'd add with more time:

- **SHAP values** — so the explanation tells you not just *what* the model predicted but *why*, at a feature level
- **Dataset versioning with DVC** — right now the data folder is just git-ignored
- **User feedback loop** — a simple thumbs up/down that logs to a file for future retraining