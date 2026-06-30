"""
conftest.py — Pytest configuration for the Adult Income Predictor test suite.

Adds src/ to sys.path so tests can import project modules without installing
the package.
"""

import sys
from pathlib import Path

# Ensure the src directory is on the import path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))