"""
BodyScan Analyzer — Main pipeline orchestrator.

Ties together:
  PoseDetector → MeasurementEstimator → BodyFatPredictor → MetricsCalculator

Usage
-----
analyzer = BodyScanAnalyzer()
result   = analyzer.analyze(
    front_image=<bytes>,
    left_image=<bytes | None>,
    right_image=<bytes | None>,
    height_cm=180,
    weight_kg=82,
    age=30,
    sex="male",
)
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass
from typing import Optional

from body_fat_predictor import BodyFatPredictor
from measurement_estimator import BodyMeasurements, MeasurementEstimator
from metrics_calculator import BodyCompositionResult, MetricsCalculator
from pose_detector import MultiViewLandmarks, PoseDetector


# ── Full analysis result ──────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    success: bool
    error: Optional[str]

    # Composition
    body_fat_pct: Optional[float]
    body_fat_category: Optional[str]
    lean_mass_kg: Optional[float]
    fat_mass_kg: Optional[float]
    skeletal_muscle_mass_kg: Optional[float]

    # Measurements
    waist_circumference_cm: Optional[float]
    hip_circumference_cm: Optional[float]
    shoulder_circumference_cm: Optional[float]
    waist_to_hip_ratio: Optional[float]

    # Visceral
    visceral_fat_score: Optional[float]
    visceral_fat_label: Optional[str]

    # Anthropometrics
    bmi: Optional[float]

    # Pipeline metadata
    pose_detected: bool
    side_view_used: bool
    processing_time_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


# ── Analyzer ──────────────────────────────────────────────────────────────────

class BodyScanAnalyzer:
    """
    End-to-end body composition analysis from photos + user metadata.

    All heavy objects (models, MediaPipe graphs) are instantiated once
    and reused across calls — safe for long-lived server processes.
    """

    def __init__(self) -> None:
        self._pose       = PoseDetector()
        self._measurer   = MeasurementEstimator()
        self._predictor  = BodyFatPredictor()
        self._calculator = MetricsCalculator()

    def analyze(
        self,
        front_image: bytes,
        left_image:  Optional[bytes] = None,
        right_image: Optional[bytes] = None,
        height_cm:   float = 175,
        weight_kg:   float = 75,
        age:         int   = 30,
        sex:         str   = "male",
    ) -> AnalysisResult:
        t0 = time.monotonic()

        try:
            # ── 1. Pose detection ─────────────────────────────────────
            mv: MultiViewLandmarks = self._pose.detect_multiview(
                front=front_image,
                left=left_image,
                right=right_image,
            )
            pose_detected = mv.front is not None

            # ── 2. Measurement estimation ─────────────────────────────
            if pose_detected:
                meas: BodyMeasurements = self._measurer.estimate(
                    mv=mv,
                    height_cm=height_cm,
                    weight_kg=weight_kg,
                )
                waist_cm    = meas.waist_circumference_cm
                hip_cm      = meas.hip_circumference_cm
                shoulder_cm = meas.shoulder_circumference_cm
                arm_w       = meas.arm_width_cm
                thigh_w     = meas.thigh_width_cm
                side_used   = meas.side_view_used
            else:
                # Fall back to population averages when pose fails
                waist_cm    = self._fallback_waist(sex, weight_kg, height_cm)
                hip_cm      = waist_cm * 1.1
                shoulder_cm = waist_cm * 1.3
                arm_w       = 10.0
                thigh_w     = 15.0
                side_used   = False

            # ── 3. Body fat prediction ────────────────────────────────
            bf_pct = self._predictor.predict(
                age=age,
                sex=sex,
                height_cm=height_cm,
                weight_kg=weight_kg,
                waist_cm=waist_cm,
                hip_cm=hip_cm,
                shoulder_cm=shoulder_cm,
            )

            # ── 4. Derived metrics ────────────────────────────────────
            comp: BodyCompositionResult = self._calculator.calculate(
                body_fat_pct=bf_pct,
                weight_kg=weight_kg,
                height_cm=height_cm,
                age=age,
                sex=sex,
                waist_cm=waist_cm,
                hip_cm=hip_cm,
                shoulder_cm=shoulder_cm,
                arm_width_cm=arm_w,
                thigh_width_cm=thigh_w,
            )

            elapsed_ms = int((time.monotonic() - t0) * 1000)

            return AnalysisResult(
                success=True,
                error=None,
                body_fat_pct=comp.body_fat_pct,
                body_fat_category=comp.body_fat_category,
                lean_mass_kg=comp.lean_mass_kg,
                fat_mass_kg=comp.fat_mass_kg,
                skeletal_muscle_mass_kg=comp.skeletal_muscle_mass_kg,
                waist_circumference_cm=comp.waist_circumference_cm,
                hip_circumference_cm=comp.hip_circumference_cm,
                shoulder_circumference_cm=comp.shoulder_circumference_cm,
                waist_to_hip_ratio=comp.waist_to_hip_ratio,
                visceral_fat_score=comp.visceral_fat_score,
                visceral_fat_label=comp.visceral_fat_label,
                bmi=comp.bmi,
                pose_detected=pose_detected,
                side_view_used=side_used,
                processing_time_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return AnalysisResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                body_fat_pct=None,
                body_fat_category=None,
                lean_mass_kg=None,
                fat_mass_kg=None,
                skeletal_muscle_mass_kg=None,
                waist_circumference_cm=None,
                hip_circumference_cm=None,
                shoulder_circumference_cm=None,
                waist_to_hip_ratio=None,
                visceral_fat_score=None,
                visceral_fat_label=None,
                bmi=None,
                pose_detected=False,
                side_view_used=False,
                processing_time_ms=elapsed_ms,
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_waist(sex: str, weight_kg: float, height_cm: float) -> float:
        """Population-average waist estimate from BMI when pose fails."""
        bmi = weight_kg / (height_cm / 100) ** 2
        if sex.lower() == "male":
            return 65 + (bmi - 18.5) * 1.8
        return 60 + (bmi - 18.5) * 1.6
