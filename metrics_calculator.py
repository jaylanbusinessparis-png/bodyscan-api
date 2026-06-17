"""
Metrics Calculator

Given body fat % + raw measurements + user metadata, derives the
complete body composition panel shown in the dashboard:

  - Lean Mass (kg)
  - Fat Mass (kg)
  - Estimated Skeletal Muscle Mass (kg)
  - Waist Circumference (cm)
  - Visceral Fat Score (1–20)
  - BMI
  - Body Fat Category
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# ── Result ────────────────────────────────────────────────────────────────────

BodyFatCategory = Literal[
    "Essential Fat", "Athletic", "Fitness",
    "Average", "Obese"
]

ViscFatLabel = Literal["Low", "Moderate", "High", "Very High"]


@dataclass
class BodyCompositionResult:
    # Core
    body_fat_pct: float
    body_fat_category: BodyFatCategory

    # Mass breakdown (kg)
    lean_mass_kg: float
    fat_mass_kg: float
    skeletal_muscle_mass_kg: float

    # Waist
    waist_circumference_cm: float

    # Visceral fat
    visceral_fat_score: float          # 1–20
    visceral_fat_label: ViscFatLabel

    # Anthropometrics
    bmi: float
    hip_circumference_cm: float
    shoulder_circumference_cm: float
    waist_to_hip_ratio: float


# ── Category tables ───────────────────────────────────────────────────────────

# (min_bf, max_bf, category) per sex
_MALE_CATEGORIES = [
    (0,    5,   "Essential Fat"),
    (5,    14,  "Athletic"),
    (14,   18,  "Fitness"),
    (18,   25,  "Average"),
    (25,   100, "Obese"),
]

_FEMALE_CATEGORIES = [
    (0,    13,  "Essential Fat"),
    (13,   21,  "Athletic"),
    (21,   25,  "Fitness"),
    (25,   32,  "Average"),
    (32,   100, "Obese"),
]


# ── Calculator ────────────────────────────────────────────────────────────────

class MetricsCalculator:

    def calculate(
        self,
        body_fat_pct: float,
        weight_kg: float,
        height_cm: float,
        age: int,
        sex: str,           # "male" | "female"
        waist_cm: float,
        hip_cm: float,
        shoulder_cm: float,
        arm_width_cm: float = 10.0,
        thigh_width_cm: float = 15.0,
    ) -> BodyCompositionResult:

        # ── Core mass ────────────────────────────────────────────────────
        bf_frac   = body_fat_pct / 100.0
        fat_mass  = round(weight_kg * bf_frac, 2)
        lean_mass = round(weight_kg * (1 - bf_frac), 2)

        # ── Skeletal Muscle Mass ─────────────────────────────────────────
        smm = self._skeletal_muscle_mass(
            lean_mass=lean_mass,
            sex=sex,
            age=age,
            waist_cm=waist_cm,
            thigh_width_cm=thigh_width_cm,
            arm_width_cm=arm_width_cm,
        )

        # ── Visceral fat ─────────────────────────────────────────────────
        vis_score = self._visceral_fat_score(
            age=age,
            sex=sex,
            bmi=weight_kg / (height_cm / 100) ** 2,
            waist_cm=waist_cm,
            body_fat_pct=body_fat_pct,
        )
        vis_label = self._visceral_label(vis_score)

        # ── BMI ──────────────────────────────────────────────────────────
        bmi = round(weight_kg / (height_cm / 100) ** 2, 1)

        # ── Category ─────────────────────────────────────────────────────
        category = self._bf_category(body_fat_pct, sex)

        # ── WHR ──────────────────────────────────────────────────────────
        whr = round(waist_cm / hip_cm, 3) if hip_cm > 0 else 0.0

        return BodyCompositionResult(
            body_fat_pct=body_fat_pct,
            body_fat_category=category,
            lean_mass_kg=lean_mass,
            fat_mass_kg=fat_mass,
            skeletal_muscle_mass_kg=round(smm, 2),
            waist_circumference_cm=round(waist_cm, 1),
            visceral_fat_score=round(vis_score, 1),
            visceral_fat_label=vis_label,
            bmi=bmi,
            hip_circumference_cm=round(hip_cm, 1),
            shoulder_circumference_cm=round(shoulder_cm, 1),
            waist_to_hip_ratio=whr,
        )

    # ── Skeletal Muscle Mass ──────────────────────────────────────────────────

    def _skeletal_muscle_mass(
        self,
        lean_mass: float,
        sex: str,
        age: int,
        waist_cm: float,
        thigh_width_cm: float,
        arm_width_cm: float,
    ) -> float:
        """
        Simplified regression model for skeletal muscle mass from lean mass.

        Based on the Lee et al. (2000) formula concept:
          SMM ≈ lean_mass × sex_factor × age_penalty
        with a small adjustment for limb size proxy.

        True V1 would use a proper regression model trained on DXA data.
        """
        sex_factor = 0.60 if sex.lower() == "male" else 0.52

        # Age penalty: muscle mass decreases after ~30 (sarcopenia)
        age_penalty = 1.0 - max(0, (age - 30) * 0.004)

        # Limb proxy: normalised limb sizes increase SMM estimate slightly
        limb_boost = 1.0 + (arm_width_cm - 10) * 0.005 + (thigh_width_cm - 15) * 0.003

        smm = lean_mass * sex_factor * age_penalty * np.clip(limb_boost, 0.9, 1.15)
        return float(np.clip(smm, 10.0, lean_mass * 0.72))

    # ── Visceral Fat Score ────────────────────────────────────────────────────

    @staticmethod
    def _visceral_fat_score(
        age: int,
        sex: str,
        bmi: float,
        waist_cm: float,
        body_fat_pct: float,
    ) -> float:
        """
        Gradient-boosting–inspired analytical model for visceral fat score (1–20).

        Photos cannot directly visualise visceral fat; this is a statistical estimate
        based on validated predictors (age, sex, BMI, waist, body fat %).

        References: Shuster et al. (2012), Fox et al. (2007).
        """
        # Base from BMI
        score = bmi * 0.25

        # Waist is the strongest predictor
        waist_thresh = 94 if sex.lower() == "male" else 80
        waist_excess = max(0, waist_cm - waist_thresh)
        score += waist_excess * 0.18

        # Body fat contribution
        score += body_fat_pct * 0.10

        # Age contribution (visceral fat accumulates with age)
        score += max(0, age - 30) * 0.06

        # Sex adjustment (men accumulate more visceral fat at equivalent total fat)
        if sex.lower() == "male":
            score *= 1.15

        return float(np.clip(score, 1.0, 20.0))

    @staticmethod
    def _visceral_label(score: float) -> ViscFatLabel:
        if score <= 9:
            return "Low"
        elif score <= 14:
            return "Moderate"
        elif score <= 17:
            return "High"
        else:
            return "Very High"

    # ── Body Fat Category ─────────────────────────────────────────────────────

    @staticmethod
    def _bf_category(bf_pct: float, sex: str) -> BodyFatCategory:
        table = _MALE_CATEGORIES if sex.lower() == "male" else _FEMALE_CATEGORIES
        for lo, hi, label in table:
            if lo <= bf_pct < hi:
                return label
        return "Obese"
