"""
Measurement Estimator

Converts pixel-space landmarks + user height into real-world body
measurements (cm), then estimates circumferences via an ellipse model.

Strategy
--------
1. Compute a pixel-to-cm scale from the front image using the
   nose-to-ankle distance vs the reported user height.
2. Read widths (shoulder, hip, waist) from the front image.
3. Read depths (waist, hip) from the side image if available,
   otherwise fall back to population-average depth ratios.
4. Compute circumferences with Ramanujan's ellipse approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pose_detector import MultiViewLandmarks, PoseLandmarks


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BodyMeasurements:
    # Widths (front view, cm)
    shoulder_width_cm: float
    hip_width_cm: float
    waist_width_cm: float

    # Circumferences (cm)
    shoulder_circumference_cm: float
    waist_circumference_cm: float
    hip_circumference_cm: float

    # Auxiliary
    arm_width_cm: float
    thigh_width_cm: float

    # Quality metadata
    scale_px_per_cm: float   # pixels per cm used for conversion
    side_view_used: bool      # True when a side image supplied depth info


# ── Depth ratios (fallback when no side image) ────────────────────────────────
# depth / width for each region (population averages)
DEPTH_RATIOS = {
    "shoulder": 0.52,
    "waist":    0.72,   # waist tends to be more round
    "hip":      0.78,
}


# ── Estimator ─────────────────────────────────────────────────────────────────

class MeasurementEstimator:
    """
    Usage
    -----
    estimator = MeasurementEstimator()
    measurements = estimator.estimate(mv_landmarks, height_cm=180)
    """

    def estimate(
        self,
        mv: MultiViewLandmarks,
        height_cm: float,
        weight_kg: Optional[float] = None,
    ) -> BodyMeasurements:
        if mv.front is None:
            raise ValueError("Front-view landmarks are required for measurement estimation.")

        front = mv.front
        scale = self._compute_scale(front, height_cm)

        shoulder_w = self._shoulder_width(front, scale)
        hip_w      = self._hip_width(front, scale)
        waist_w    = self._waist_width(front, hip_w, scale)
        arm_w      = self._arm_width(front, scale)
        thigh_w    = self._thigh_width(front, scale)

        side_used = mv.left is not None or mv.right is not None
        side = mv.left or mv.right

        # Depths: use side image if available, else fall back to ratios
        if side is not None:
            waist_d = self._side_depth(side, "waist", scale, fallback=waist_w * DEPTH_RATIOS["waist"])
            hip_d   = self._side_depth(side, "hip",   scale, fallback=hip_w   * DEPTH_RATIOS["hip"])
            sh_d    = shoulder_w * DEPTH_RATIOS["shoulder"]
        else:
            waist_d = waist_w * DEPTH_RATIOS["waist"]
            hip_d   = hip_w   * DEPTH_RATIOS["hip"]
            sh_d    = shoulder_w * DEPTH_RATIOS["shoulder"]

        sh_circ    = self._ellipse_perimeter(shoulder_w / 2, sh_d / 2)
        waist_circ = self._ellipse_perimeter(waist_w / 2, waist_d / 2)
        hip_circ   = self._ellipse_perimeter(hip_w / 2, hip_d / 2)

        return BodyMeasurements(
            shoulder_width_cm=round(shoulder_w, 1),
            hip_width_cm=round(hip_w, 1),
            waist_width_cm=round(waist_w, 1),
            shoulder_circumference_cm=round(sh_circ, 1),
            waist_circumference_cm=round(waist_circ, 1),
            hip_circumference_cm=round(hip_circ, 1),
            arm_width_cm=round(arm_w, 1),
            thigh_width_cm=round(thigh_w, 1),
            scale_px_per_cm=round(scale, 4),
            side_view_used=side_used,
        )

    # ── Scale ─────────────────────────────────────────────────────────────

    def _compute_scale(self, front: PoseLandmarks, height_cm: float) -> float:
        """
        Pixels-to-cm scale from nose → ankle distance.
        We use 0.88 × height because the nose sits ~12% below the top of the head.
        """
        nose   = front.get("nose")
        l_ank  = front.get("left_ankle")
        r_ank  = front.get("right_ankle")

        if nose is None or (l_ank is None and r_ank is None):
            # Fallback: use image height as proxy
            return front.image_height / (height_cm * 0.88)

        ankle = self._avg(l_ank, r_ank)
        pixel_span = np.linalg.norm(ankle - nose)
        effective_height_cm = height_cm * 0.88

        return pixel_span / effective_height_cm  # pixels per cm

    # ── Width helpers ─────────────────────────────────────────────────────

    def _shoulder_width(self, f: PoseLandmarks, scale: float) -> float:
        ls = f.get("left_shoulder")
        rs = f.get("right_shoulder")
        if ls is None or rs is None:
            return 42.0  # population average fallback
        return float(np.linalg.norm(rs - ls)) / scale

    def _hip_width(self, f: PoseLandmarks, scale: float) -> float:
        lh = f.get("left_hip")
        rh = f.get("right_hip")
        if lh is None or rh is None:
            return 38.0
        return float(np.linalg.norm(rh - lh)) / scale

    def _waist_width(self, f: PoseLandmarks, hip_w_cm: float, scale: float) -> float:
        """
        MediaPipe doesn't have a waist landmark. We interpolate between
        the shoulder midpoint and hip midpoint at ~55% of the way down.
        Then estimate the pixel width at that y-coordinate using hip width
        scaled by the shoulder/hip ratio at waist level.
        """
        ls = f.get("left_shoulder")
        rs = f.get("right_shoulder")
        lh = f.get("left_hip")
        rh = f.get("right_hip")

        if None in (ls, rs, lh, rh):
            return hip_w_cm * 0.82  # fallback ratio

        sh_mid  = (ls + rs) / 2
        hip_mid = (lh + rh) / 2

        # Waist sits ~55% of the way from shoulders to hips in y
        waist_y = sh_mid[1] + (hip_mid[1] - sh_mid[1]) * 0.55

        # We don't have explicit waist x-coords, so we scale by a ratio
        sh_w_px = float(np.linalg.norm(rs - ls))
        hp_w_px = float(np.linalg.norm(rh - lh))

        # Linear interpolation at 55%
        waist_w_px = sh_w_px + (hp_w_px - sh_w_px) * 0.6
        return waist_w_px / scale

    def _arm_width(self, f: PoseLandmarks, scale: float) -> float:
        ls = f.get("left_shoulder")
        le = f.get("left_elbow")
        rs = f.get("right_shoulder")
        re = f.get("right_elbow")

        if ls and le:
            upper_arm_len = float(np.linalg.norm(le - ls)) / scale
        elif rs and re:
            upper_arm_len = float(np.linalg.norm(re - rs)) / scale
        else:
            return 10.0

        # Upper arm circumference ≈ 28-32% of upper arm length (rough estimate)
        return upper_arm_len * 0.30

    def _thigh_width(self, f: PoseLandmarks, scale: float) -> float:
        lh = f.get("left_hip")
        lk = f.get("left_knee")
        rh = f.get("right_hip")
        rk = f.get("right_knee")

        if lh and lk:
            thigh_len = float(np.linalg.norm(lk - lh)) / scale
        elif rh and rk:
            thigh_len = float(np.linalg.norm(rk - rh)) / scale
        else:
            return 15.0

        return thigh_len * 0.38

    # ── Side depth ────────────────────────────────────────────────────────

    def _side_depth(
        self,
        side: PoseLandmarks,
        region: str,
        scale: float,
        fallback: float,
    ) -> float:
        """
        Estimate depth at a body region from the side view.
        In the side profile, the 'width' of a body segment is its depth.
        We approximate by reading left/right landmark separation in the
        side image (which collapses to front-back distance).
        """
        if region == "waist":
            # In side view, waist depth ≈ distance between shoulder z and hip x extents
            # Use hip-to-shoulder in X axis as proxy for body thickness at waist
            ls = side.get("left_shoulder")
            lh = side.get("left_hip")
            if ls is not None and lh is not None:
                # x-distance in side view ≈ body front-to-back at mid torso
                return abs(float(ls[0] - lh[0])) / scale * 1.2
        elif region == "hip":
            lh = side.get("left_hip")
            rh = side.get("right_hip")
            if lh is not None and rh is not None:
                return float(np.linalg.norm(rh - lh)) / scale * 0.9

        return fallback

    # ── Geometry ──────────────────────────────────────────────────────────

    @staticmethod
    def _ellipse_perimeter(a: float, b: float) -> float:
        """Ramanujan's approximation for ellipse perimeter, given semi-axes a, b."""
        if a <= 0 or b <= 0:
            return 0.0
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

    @staticmethod
    def _avg(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> np.ndarray:
        if a is None:
            return b
        if b is None:
            return a
        return (a + b) / 2
