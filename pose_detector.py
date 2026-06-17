"""
Pose Detector — MediaPipe Pose landmark extraction.

Accepts raw image bytes (front, left, right) and returns
normalised pixel coordinates for key body landmarks.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import mediapipe.solutions.pose as mp_pose

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ── Types ────────────────────────────────────────────────────────────────────

Point2D = Tuple[float, float]   # (x_px, y_px)


@dataclass
class PoseLandmarks:
    """Pixel-space landmarks for one image view."""
    landmarks: Dict[str, Point2D]
    image_width: int
    image_height: int
    visibility: Dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> Optional[np.ndarray]:
        """Return landmark as np.array([x, y]) or None if not tracked."""
        pt = self.landmarks.get(name)
        return np.array(pt, dtype=float) if pt else None


@dataclass
class MultiViewLandmarks:
    front: Optional[PoseLandmarks] = None
    left: Optional[PoseLandmarks] = None
    right: Optional[PoseLandmarks] = None


# ── Constants ────────────────────────────────────────────────────────────────

# MediaPipe landmark index → semantic name
LANDMARK_MAP: Dict[int, str] = {
    0:  "nose",
    7:  "left_ear",
    8:  "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

MIN_VISIBILITY = 0.5  # discard landmarks below this confidence


# ── Detector ─────────────────────────────────────────────────────────────────

class PoseDetector:
    """
    Wraps MediaPipe Pose for static image landmark detection.

    Usage
    -----
    detector = PoseDetector()
    mv = detector.detect_multiview(front_bytes, left_bytes, right_bytes)
    """

    def __init__(self, model_complexity: int = 2) -> None:
        self._mp_pose = mp_pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ── Public ────────────────────────────────────────────────────────────

    def detect(self, image_bytes: bytes) -> Optional[PoseLandmarks]:
        """Detect pose landmarks from raw image bytes. Returns None on failure."""
        img_bgr = self._decode(image_bytes)
        if img_bgr is None:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(img_rgb)

        if not results.pose_landmarks:
            return None

        h, w = img_bgr.shape[:2]
        landmarks: Dict[str, Point2D] = {}
        visibility: Dict[str, float] = {}

        for idx, name in LANDMARK_MAP.items():
            lm = results.pose_landmarks.landmark[idx]
            if lm.visibility >= MIN_VISIBILITY:
                landmarks[name] = (lm.x * w, lm.y * h)
                visibility[name] = lm.visibility

        return PoseLandmarks(
            landmarks=landmarks,
            image_width=w,
            image_height=h,
            visibility=visibility,
        )

    def detect_multiview(
        self,
        front: bytes,
        left: Optional[bytes] = None,
        right: Optional[bytes] = None,
    ) -> MultiViewLandmarks:
        """Detect landmarks from up to three views."""
        return MultiViewLandmarks(
            front=self.detect(front),
            left=self.detect(left) if left else None,
            right=self.detect(right) if right else None,
        )

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _decode(image_bytes: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return None
        return img

    def __del__(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass
