"""
BodyScan AI — Analysis API Router

POST /analysis/run
  Accepts: multipart form (front_image required, left_image/right_image optional)
           + user metadata fields
  Returns: AnalysisResult JSON

GET /analysis/health
  Returns pipeline status (models loaded, etc.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Pipeline import (works whether run from project root or api/) ─────────────
#ROOT = Path(__file__).parent.parent
#sys.path.insert(0, str(ROOT))

from analyzer import BodyScanAnalyzer



# Singleton — instantiated once at import time (loads models, MediaPipe graph)
_analyzer: Optional[BodyScanAnalyzer] = None


def get_analyzer() -> BodyScanAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = BodyScanAnalyzer()
    return _analyzer


router = APIRouter(prefix="/analysis", tags=["analysis"])


# ── Response schema ───────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    success: bool
    error: Optional[str] = None

    # Composition
    body_fat_pct: Optional[float] = Field(None, description="Body fat percentage (0-60)")
    body_fat_category: Optional[str] = None
    lean_mass_kg: Optional[float] = None
    fat_mass_kg: Optional[float] = None
    skeletal_muscle_mass_kg: Optional[float] = None

    # Measurements
    waist_circumference_cm: Optional[float] = None
    hip_circumference_cm: Optional[float] = None
    shoulder_circumference_cm: Optional[float] = None
    waist_to_hip_ratio: Optional[float] = None

    # Visceral fat
    visceral_fat_score: Optional[float] = Field(None, description="Score 1-20")
    visceral_fat_label: Optional[str] = None

    # Anthropometrics
    bmi: Optional[float] = None

    # Pipeline metadata
    pose_detected: bool = False
    side_view_used: bool = False
    processing_time_ms: int = 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/run", response_model=AnalysisResponse, summary="Analyse body photos")
async def run_analysis(
    # Images
    front_image: Annotated[UploadFile, File(description="Front-facing body photo (required)")],
    left_image:  Annotated[Optional[UploadFile], File(description="Left profile photo")] = None,
    right_image: Annotated[Optional[UploadFile], File(description="Right profile photo")] = None,

    # User metadata
    height_cm: Annotated[float, Form(ge=100, le=250, description="Height in cm")] = 175,
    weight_kg: Annotated[float, Form(ge=30,  le=300, description="Weight in kg")]  = 75,
    age:       Annotated[int,   Form(ge=10,  le=100, description="Age in years")]   = 30,
    sex:       Annotated[str,   Form(description="'male' or 'female'")]             = "male",
) -> AnalysisResponse:

    # Validate sex
    if sex.lower() not in ("male", "female"):
        raise HTTPException(status_code=422, detail="sex must be 'male' or 'female'")

    # Read image bytes
    front_bytes = await front_image.read()
    left_bytes  = await left_image.read()  if left_image  else None
    right_bytes = await right_image.read() if right_image else None

    if len(front_bytes) == 0:
        raise HTTPException(status_code=400, detail="front_image is empty")

    analyzer = get_analyzer()
    result = analyzer.analyze(
        front_image=front_bytes,
        left_image=left_bytes,
        right_image=right_bytes,
        height_cm=height_cm,
        weight_kg=weight_kg,
        age=age,
        sex=sex,
    )

    return AnalysisResponse(**result.to_dict())


@router.get("/health", summary="Pipeline health check")
async def health() -> JSONResponse:
    model_path  = ROOT / "models" / "body_fat_xgb.joblib"
    scaler_path = ROOT / "models" / "feature_scaler.joblib"

    return JSONResponse({
        "status": "ok",
        "model_loaded":  model_path.exists(),
        "scaler_loaded": scaler_path.exists(),
    })
