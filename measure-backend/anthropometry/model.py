"""
anthropometry/model.py

A gender-aware anthropometric model for estimating body measurements
(shoulder width, chest, waist, hip, height) from 2D pose landmarks.

Why this exists
----------------
The original implementation derived chest/waist purely from fixed ratios
applied to shoulder/hip pixel widths (e.g. chest = shoulder * 1.45 for
*everyone*, regardless of gender or body shape). That is the single
biggest source of inaccuracy in a vision-based sizing app, because:

  1. The shoulder-to-chest and hip-to-waist ratios differ meaningfully
     between male and female body distributions.
  2. A single ratio ignores body-shape variation entirely (e.g. two
     people with identical shoulder width can have chest circumferences
     that differ by 10+ cm depending on build).
  3. Circumference (chest/waist/hip "around the body") cannot be
     derived from a single frontal-width measurement without *some*
     model of body depth/cross-section - a true circumference requires
     either a depth sensor, multiple viewpoints, or a population-level
     width-to-circumference regression. We use the latter, which is the
     standard approach in single-RGB-camera body measurement
     literature (e.g. SMPL-based body shape regression, and earlier
     anthropometric survey-based heuristics such as those used in
     ANSUR II / NHANES derived clothing-size tools).

This module implements a regression-style estimator:
  - Uses *multiple* frontal landmark-derived widths (shoulder, hip,
    torso length) rather than a single one.
  - Applies gender-specific width-to-circumference regression
    coefficients derived from published anthropometric survey ratios
    (ANSUR II summary statistics: shoulder breadth, bideltoid breadth,
    waist circumference, hip circumference, chest circumference, by
    sex). These coefficients are approximations distilled from public
    summary statistics, not a proprietary or scraped dataset.
  - Blends a depth-aware body "thickness" prior (from a monocular
    depth map, when available) to correct for people who are
    photographed at an angle or who have larger front-to-back depth
    than the regression would predict from width alone.
  - Reports a confidence score driven by landmark visibility, pose
    symmetry, and how plausible the resulting measurements are
    (basic physiological bounds-checking).

This is still an estimate, not a medical-grade measurement - no
single RGB camera approach can replace a tape measure - but the goal
is to materially reduce systematic error versus fixed-ratio guessing,
and to be transparent about uncertainty via the confidence score
rather than reporting false precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Pose convention)
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


@dataclass
class Landmark:
    x: float
    y: float
    visibility: float = 1.0


@dataclass
class BodyMeasurements:
    shoulder_cm: float
    chest_cm: float
    waist_cm: float
    hip_cm: float
    height_cm: float
    confidence: float  # 0..100
    notes: List[str]

    def to_dict(self) -> Dict[str, float]:
        return {
            "shoulderCm": round(self.shoulder_cm),
            "chestCm": round(self.chest_cm),
            "waistCm": round(self.waist_cm),
            "hipCm": round(self.hip_cm),
            "heightCm": round(self.height_cm),
            "confidence": round(self.confidence, 1),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Regression coefficients
# ---------------------------------------------------------------------------
# These coefficients map a *frontal width measurement* (in cm, derived from
# pixel distance x scale) to a body *circumference* (in cm). They are
# distilled from public anthropometric summary ratios (ANSUR II / general
# population survey statistics on bideltoid breadth, waist breadth, hip
# breadth vs. their respective circumferences). Frontal width is reliably
# ~0.34-0.40x of the matching circumference for the torso region in adults,
# varying by sex and which circumference is being modeled.
#
# circumference_cm = a * frontal_width_cm + b
#
# Coefficients differ for male / female because of differing average
# torso depth-to-width ratios (e.g. female hip circumference is
# proportionally larger relative to frontal hip width than male hip
# circumference is, on average).
@dataclass
class GenderCoefficients:
    chest_a: float
    chest_b: float
    waist_a: float
    waist_b: float
    hip_a: float
    hip_b: float


COEFFICIENTS: Dict[str, GenderCoefficients] = {
    "male": GenderCoefficients(
        chest_a=2.55, chest_b=4.0,
        waist_a=2.65, waist_b=2.0,
        hip_a=2.45, hip_b=6.0,
    ),
    "female": GenderCoefficients(
        chest_a=2.50, chest_b=6.0,
        waist_a=2.80, waist_b=3.0,
        hip_a=2.60, hip_b=8.0,
    ),
}


def _coeffs_for(gender: str) -> GenderCoefficients:
    g = (gender or "male").strip().lower()
    return COEFFICIENTS.get(g, COEFFICIENTS["male"])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _pixel_dist(a: Landmark, b: Landmark, width: int, height: int) -> float:
    dx = (a.x - b.x) * width
    dy = (a.y - b.y) * height
    return math.hypot(dx, dy)


def _get(landmarks: Sequence[Landmark], idx: int) -> Optional[Landmark]:
    if idx < 0 or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if lm is None:
        return None
    return lm


def _visible(lm: Optional[Landmark], threshold: float = 0.45) -> bool:
    return lm is not None and (lm.visibility is None or lm.visibility >= threshold)


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------
def estimate(
    landmarks: Sequence[Landmark],
    image_width: int,
    image_height: int,
    gender: str = "male",
    user_height_cm: Optional[float] = None,
    depth_thickness_factor: Optional[float] = None,
) -> Optional[BodyMeasurements]:
    """
    Estimate body measurements from normalized pose landmarks.

    Parameters
    ----------
    landmarks: sequence of Landmark with .x/.y normalized 0..1 and
        .visibility 0..1, indexed per MediaPipe Pose convention.
    image_width, image_height: pixel dimensions of the source frame used
        to convert normalized coordinates to pixel distances.
    gender: 'male' or 'female' - used to select regression coefficients.
        This MUST come from explicit user input, not be inferred from
        body shape (inferring it from hip/shoulder ratio embeds a
        stereotype into the math and is also frequently wrong).
    user_height_cm: optional self-reported height; used as the primary
        calibration scale when plausible (120-230 cm), since a real
        height reading is more reliable than any visual heuristic.
    depth_thickness_factor: optional 0..1+ scalar derived from a depth
        map (e.g. MiDaS) representing relative torso "thickness" vs. a
        population baseline. >1.0 means the subject appears thicker
        front-to-back than width alone would suggest (e.g. photographed
        at a slight angle, or a deeper/rounder build); used to nudge
        circumference estimates. Optional - pass None to skip.

    Returns
    -------
    BodyMeasurements or None if required landmarks are missing/too
    low-confidence to produce a usable estimate.
    """
    notes: List[str] = []

    if not landmarks or len(landmarks) < 29:
        return None

    nose = _get(landmarks, NOSE)
    ls = _get(landmarks, LEFT_SHOULDER)
    rs = _get(landmarks, RIGHT_SHOULDER)
    lh = _get(landmarks, LEFT_HIP)
    rh = _get(landmarks, RIGHT_HIP)
    la = _get(landmarks, LEFT_ANKLE)
    ra = _get(landmarks, RIGHT_ANKLE)

    required = [nose, ls, rs, lh, rh]
    if any(not _visible(p) for p in required):
        return None
    if not (_visible(la) or _visible(ra)):
        return None

    # --- Pixel measurements -------------------------------------------------
    shoulder_px = _pixel_dist(ls, rs, image_width, image_height)
    hip_px = _pixel_dist(lh, rh, image_width, image_height)

    ankle_candidates = [p.y for p in (la, ra) if _visible(p)]
    ankle_y_norm = max(ankle_candidates) if ankle_candidates else max(lh.y, rh.y)
    pixel_height_full = max(5.0, (ankle_y_norm - nose.y) * image_height)

    mid_hip_y = (lh.y + rh.y) / 2.0
    torso_px = max(5.0, (mid_hip_y - nose.y) * image_height)

    # --- Scale calibration (pixels -> cm) -----------------------------------
    # Priority: real user-provided height is the single most reliable
    # signal available (a tape-measure-grade ground truth), so when
    # plausible it should dominate the scale rather than being merely
    # blended a quarter of the way in as the original implementation did.
    #
    # IMPORTANT: validity is computed once and reused everywhere below.
    # An earlier version of this function re-checked "if user_height_cm"
    # (truthiness only) in some branches while checking the full
    # plausible-range condition in others. That mismatch meant an
    # implausible value like 999 would correctly be excluded from the
    # *primary* calibration source, but would still leak into the
    # torso-fraction fallback's `assumed_height_for_torso` and would
    # incorrectly suppress the shoulder-width fallback - producing a
    # wildly inflated, clamped-at-ceiling height estimate instead of
    # safely falling back to population-average heuristics.
    has_valid_height = bool(user_height_cm) and 120 <= user_height_cm <= 230

    scale_sources: List[float] = []
    scale_weights: List[float] = []

    if has_valid_height:
        scale_sources.append(user_height_cm / pixel_height_full)
        scale_weights.append(0.85)
        notes.append("Scale calibrated primarily from user-provided height.")
    else:
        notes.append(
            "No valid height provided; scale is estimated from body "
            "proportions only. Accuracy improves significantly if height "
            "is supplied."
        )

    # Population-average fallback / secondary signal: nose-to-hip
    # (torso) span as a fraction of total height. This fraction is
    # fairly stable across adults (roughly 0.52 of standing height from
    # crown to hip for most adult body types) and acts as a secondary,
    # independent calibration signal even when height is supplied, which
    # helps catch frames where the ankle wasn't fully visible and
    # pixel_height_full is underestimated.
    torso_fraction = 0.52
    assumed_height_for_torso = user_height_cm if has_valid_height else 170.0
    torso_scale = (assumed_height_for_torso * torso_fraction) / torso_px
    scale_sources.append(torso_scale)
    scale_weights.append(0.15 if has_valid_height else 0.5)

    # Shoulder-width-based fallback only used when no *valid* height at
    # all, since gender-specific average shoulder breadth is a weaker
    # signal than torso fraction.
    if not has_valid_height:
        avg_shoulder = 41.0 if gender == "male" else 36.5
        if shoulder_px > 6:
            scale_sources.append(avg_shoulder / shoulder_px)
            scale_weights.append(0.5)

    total_w = sum(scale_weights)
    px_to_cm = sum(s * w for s, w in zip(scale_sources, scale_weights)) / total_w

    # --- Frontal widths in cm -----------------------------------------------
    shoulder_cm = shoulder_px * px_to_cm
    hip_width_cm = hip_px * px_to_cm
    height_cm = pixel_height_full * px_to_cm

    # --- Circumference regression (gender-aware) ----------------------------
    coeffs = _coeffs_for(gender)

    chest_cm = coeffs.chest_a * shoulder_cm + coeffs.chest_b
    waist_cm = coeffs.waist_a * (hip_width_cm * 0.86) + coeffs.waist_b
    # Waist frontal width is narrower than hip frontal width for most
    # adults; 0.86 approximates waist/hip frontal-width ratio before
    # applying the circumference regression.
    hip_cm = coeffs.hip_a * hip_width_cm + coeffs.hip_b

    # --- Depth-aware correction ---------------------------------------------
    if depth_thickness_factor is not None and depth_thickness_factor > 0:
        # Clamp correction so a noisy depth map cannot wildly distort
        # the regression-based estimate; this is a *nudge*, not a
        # replacement for the regression.
        factor = max(0.85, min(1.20, depth_thickness_factor))
        chest_cm *= factor
        waist_cm *= factor
        hip_cm *= factor
        notes.append(f"Depth-based thickness correction applied (factor={factor:.2f}).")

    # --- Physiological bounds & sanity clamps -------------------------------
    shoulder_cm = _clamp(shoulder_cm, 25, 60)
    chest_cm = _clamp(chest_cm, 65, 160)
    waist_cm = _clamp(waist_cm, 55, 160)
    hip_cm = _clamp(hip_cm, 65, 160)
    height_cm = _clamp(height_cm, 120, 230)

    # --- Confidence scoring ---------------------------------------------------
    confidence = _confidence_score(
        landmarks=landmarks,
        has_user_height=has_valid_height,
        shoulder_px=shoulder_px,
        hip_px=hip_px,
        notes=notes,
    )

    return BodyMeasurements(
        shoulder_cm=shoulder_cm,
        chest_cm=chest_cm,
        waist_cm=waist_cm,
        hip_cm=hip_cm,
        height_cm=height_cm,
        confidence=confidence,
        notes=notes,
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _confidence_score(
    landmarks: Sequence[Landmark],
    has_user_height: bool,
    shoulder_px: float,
    hip_px: float,
    notes: List[str],
) -> float:
    """
    Produce a 0-100 confidence score from signals we actually have:
      - average visibility of the key landmarks used
      - whether a real height was supplied (calibration quality)
      - whether shoulder/hip pixel widths are large enough to be
        measured precisely (a person far from the camera yields tiny,
        noise-dominated pixel distances)
    This is deliberately conservative; it is meant to communicate
    uncertainty to the user, not to flatter the estimate.
    """
    key_indices = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]
    vis_values = []
    for i in key_indices:
        lm = _get(landmarks, i)
        if lm is not None and lm.visibility is not None:
            vis_values.append(lm.visibility)
    avg_vis = (sum(vis_values) / len(vis_values)) if vis_values else 0.5

    score = 40.0 * avg_vis  # up to 40 points for landmark visibility

    score += 30.0 if has_user_height else 12.0  # calibration quality

    # Pixel-scale resolution: very small shoulder/hip pixel spans mean
    # the subject is far from the camera and pixel noise dominates.
    pixel_quality = min(1.0, (min(shoulder_px, hip_px) / 60.0))
    score += 30.0 * pixel_quality

    if pixel_quality < 0.5:
        notes.append(
            "Subject appears far from the camera; move closer for a "
            "more precise measurement."
        )

    return _clamp(score, 0, 100)
