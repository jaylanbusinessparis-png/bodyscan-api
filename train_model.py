"""
Train Body Fat XGBoost Model
============================

Generates a synthetic dataset using the US Navy Body Fat Formula as
ground truth, then trains an XGBoost regressor.

Usage
-----
    python training/train_model.py

Output
------
    models/body_fat_xgb.joblib
    models/feature_scaler.joblib
    models/training_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
N_SAMPLES   = 20_000


# ── Navy Body Fat Formula ─────────────────────────────────────────────────────

def navy_bf_male(waist_cm: float, neck_cm: float, height_cm: float) -> float:
    return 495 / (
        1.0324
        - 0.19077 * np.log10(np.maximum(waist_cm - neck_cm, 0.1))
        + 0.15456 * np.log10(height_cm)
    ) - 450


def navy_bf_female(waist_cm: float, hip_cm: float, neck_cm: float, height_cm: float) -> float:
    return 495 / (
        1.29579
        - 0.35004 * np.log10(np.maximum(waist_cm + hip_cm - neck_cm, 0.1))
        + 0.22100 * np.log10(height_cm)
    ) - 450


# ── Synthetic Dataset Generation ──────────────────────────────────────────────

def generate_dataset(n: int, seed: int = RANDOM_SEED) -> tuple:
    rng = np.random.default_rng(seed)

    # 50/50 sex split
    n_male   = n // 2
    n_female = n - n_male

    def sample_males(n_m):
        age       = rng.integers(18, 75, n_m).astype(float)
        height_cm = rng.normal(175, 8, n_m).clip(155, 200)
        weight_kg = rng.normal(80, 14, n_m).clip(50, 150)
        bmi       = weight_kg / (height_cm / 100) ** 2

        # Waist: correlated with BMI
        waist_cm  = (rng.normal(88, 10, n_m) + (bmi - 22) * 1.5).clip(65, 130)
        hip_cm    = waist_cm * rng.uniform(0.92, 1.05, n_m)
        neck_cm   = rng.normal(38, 3, n_m).clip(30, 50)
        shoulder_cm = waist_cm * rng.uniform(1.25, 1.45, n_m)

        bf = navy_bf_male(waist_cm, neck_cm, height_cm)
        # Add small noise to simulate real-world measurement variance
        bf += rng.normal(0, 1.5, n_m)
        bf = bf.clip(3, 40)

        sex = np.ones(n_m)
        return age, sex, height_cm, weight_kg, bmi, waist_cm, hip_cm, shoulder_cm, bf

    def sample_females(n_f):
        age       = rng.integers(18, 75, n_f).astype(float)
        height_cm = rng.normal(163, 7, n_f).clip(145, 185)
        weight_kg = rng.normal(65, 12, n_f).clip(40, 130)
        bmi       = weight_kg / (height_cm / 100) ** 2

        waist_cm  = (rng.normal(76, 9, n_f) + (bmi - 22) * 1.3).clip(55, 120)
        hip_cm    = waist_cm * rng.uniform(1.08, 1.25, n_f)
        neck_cm   = rng.normal(32, 2.5, n_f).clip(25, 42)
        shoulder_cm = waist_cm * rng.uniform(1.0, 1.2, n_f)

        bf = navy_bf_female(waist_cm, hip_cm, neck_cm, height_cm)
        bf += rng.normal(0, 1.5, n_f)
        bf = bf.clip(10, 55)

        sex = np.zeros(n_f)
        return age, sex, height_cm, weight_kg, bmi, waist_cm, hip_cm, shoulder_cm, bf

    # Combine
    cols_m = sample_males(n_male)
    cols_f = sample_females(n_female)

    age, sex, height, weight, bmi, waist, hip, shoulder, bf = [
        np.concatenate([m, f]) for m, f in zip(cols_m, cols_f)
    ]

    # Waist-to-hip ratio
    whr = waist / hip

    X = np.column_stack([age, sex, height, weight, bmi, waist, hip, shoulder, whr])
    y = bf

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── Train ─────────────────────────────────────────────────────────────────────

def train() -> None:
    print("Generating synthetic dataset …")
    X, y = generate_dataset(N_SAMPLES)
    print(f"  {len(y)} samples | BF% mean={y.mean():.1f} std={y.std():.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\nTraining XGBoost regressor …")
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train_s, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test_s)
    mae     = mean_absolute_error(y_test, y_pred)
    r2      = r2_score(y_test, y_pred)

    cv_mae = -cross_val_score(
        model, X_train_s, y_train,
        cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    ).mean()

    report = (
        f"XGBoost Body Fat Model — Training Report\n"
        f"{'=' * 45}\n"
        f"Training samples : {len(y_train):,}\n"
        f"Test samples     : {len(y_test):,}\n\n"
        f"Test MAE         : {mae:.2f}%\n"
        f"Test R²          : {r2:.4f}\n"
        f"5-fold CV MAE    : {cv_mae:.2f}%\n\n"
        f"Feature importance (top 9):\n"
    )

    feature_names = [
        "age", "sex", "height_cm", "weight_kg",
        "bmi", "waist_cm", "hip_cm", "shoulder_cm", "whr",
    ]
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        report += f"  {name:<20} {imp:.4f}\n"

    print("\n" + report)

    # ── Save ──────────────────────────────────────────────────────────────
    model_path  = MODELS_DIR / "body_fat_xgb.joblib"
    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    report_path = MODELS_DIR / "training_report.txt"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    report_path.write_text(report)

    print(f"Model saved  → {model_path}")
    print(f"Scaler saved → {scaler_path}")
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    train()
