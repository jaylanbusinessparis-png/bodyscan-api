"""
Body Fat Predictor — XGBoost regression model.

Input features
--------------
age             : int       (years)
sex             : int       (0 = female, 1 = male)
height_cm       : float
weight_kg       : float
bmi             : float     (derived)
waist_cm        : float
hip_cm          : float
shoulder_cm     : float
waist_hip_ratio : float     (derived)

Output
------
body_fat_pct    : float     (0–60 range)

Model is trained via training/train_model.py using synthetic data generated
from the US Navy Body Fat formula + Gaussian noise.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

MODEL_PATH = Path(__file__).parent.parent / "models" / "body_fat_xgb.joblib"
SCALER_PATH = Path(__file__).parent.parent / "models" / "feature_scaler.joblib"


class BodyFatPredictor:
    """
    Loads a trained XGBoost model and predicts body fat %.

    If the model file does not exist, falls back to the Navy formula
    so the pipeline remains usable without a trained artefact.
    """

    def __init__(self) -> None:
        self._model  = self._load(MODEL_PATH)
        self._scaler = self._load(SCALER_PATH)

    # ── Public ────────────────────────────────────────────────────────────

    def predict(
        self,
        age: int,
        sex: str,             # "male" | "female"
        height_cm: float,
        weight_kg: float,
        waist_cm: float,
        hip_cm: float,
        shoulder_cm: float,
    ) -> float:
        """Return estimated body fat percentage."""
        sex_int = 1 if sex.lower() == "male" else 0
        bmi = weight_kg / (height_cm / 100) ** 2
        whr = waist_cm / hip_cm if hip_cm > 0 else 0.9

        features = np.array([[
            age, sex_int, height_cm, weight_kg,
            bmi, waist_cm, hip_cm, shoulder_cm, whr,
        ]], dtype=float)

        if self._model is not None and self._scaler is not None:
            features_scaled = self._scaler.transform(features)
            bf = float(self._model.predict(features_scaled)[0])
        else:
            bf = self._navy_formula(sex, height_cm, waist_cm, hip_cm)

        return round(float(np.clip(bf, 3.0, 55.0)), 1)

    # ── Navy formula fallback ─────────────────────────────────────────────

    @staticmethod
    def _navy_formula(
        sex: str,
        height_cm: float,
        waist_cm: float,
        hip_cm: float,
        neck_cm: float = 38.0,   # average neck — we don't measure it yet
    ) -> float:
        """US Navy body fat formula (Hodgdon & Beckett, 1984)."""
        h = height_cm
        if sex.lower() == "male":
            val = 495 / (
                1.0324
                - 0.19077 * np.log10(waist_cm - neck_cm)
                + 0.15456 * np.log10(h)
            ) - 450
        else:
            val = 495 / (
                1.29579
                - 0.35004 * np.log10(waist_cm + hip_cm - neck_cm)
                + 0.22100 * np.log10(h)
            ) - 450
        return float(val)

    # ── Feature names (for training script) ──────────────────────────────

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "age", "sex", "height_cm", "weight_kg",
            "bmi", "waist_cm", "hip_cm", "shoulder_cm",
            "waist_hip_ratio",
        ]

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: Path):
        if path.exists():
            return joblib.load(path)
        return None
