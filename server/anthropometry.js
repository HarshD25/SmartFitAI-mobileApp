// server/anthropometry.js
//
// Fallback measurement logic used ONLY when the Python measurement
// service (measure-backend/measure_server.py) is unreachable. The
// Python service is the primary, more accurate engine (gender-aware
// regression + optional depth correction - see
// measure-backend/anthropometry/model.py for the full rationale).
//
// This module intentionally mirrors the *spirit* of that model
// (gender-aware coefficients, height-prioritized calibration,
// confidence scoring) using the same conceptual approach, so that
// degraded-mode results aren't wildly inconsistent with the primary
// engine's results. It does not replicate the Python module
// line-for-line since they run in different languages, but the
// regression coefficients and calibration priority are intentionally
// kept in sync - if you tune one, tune the other.

const COEFFICIENTS = {
  male: { chestA: 2.55, chestB: 4.0, waistA: 2.65, waistB: 2.0, hipA: 2.45, hipB: 6.0 },
  female: { chestA: 2.50, chestB: 6.0, waistA: 2.80, waistB: 3.0, hipA: 2.60, hipB: 8.0 },
};

function coeffsFor(gender) {
  const key = (gender || 'male').toLowerCase();
  return COEFFICIENTS[key] || COEFFICIENTS.male;
}

function pixelDist(a, b, w, h) {
  const dx = (a.x - b.x) * w;
  const dy = (a.y - b.y) * h;
  return Math.hypot(dx, dy);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function visible(lm, threshold = 0.45) {
  return !!lm && (lm.visibility === undefined || lm.visibility === null || lm.visibility >= threshold);
}

/**
 * Estimate body measurements from normalized pose landmarks.
 * Mirrors measure-backend/anthropometry/model.py's approach:
 *  - prioritizes user-supplied height for pixel->cm calibration
 *  - applies gender-aware width-to-circumference regression instead
 *    of a single fixed ratio for every body type
 *  - returns a confidence score driven by landmark visibility and
 *    pixel-scale quality, not just brand-match closeness
 */
function estimateMeasures(landmarks, width = 720, height = 1280, userHeight = null, gender = 'male') {
  if (!landmarks || landmarks.length < 29) return null;
  const p = (i) => landmarks[i] || { x: 0.5, y: 0.5, visibility: 0 };

  const nose = p(0), ls = p(11), rs = p(12), lh = p(23), rh = p(24), la = p(27), ra = p(28);

  const required = [nose, ls, rs, lh, rh];
  if (required.some((lm) => !visible(lm))) return null;
  if (!visible(la) && !visible(ra)) return null;

  const shoulderPx = pixelDist(ls, rs, width, height);
  const hipPx = pixelDist(lh, rh, width, height);
  const ankleYNorm = Math.max(visible(la) ? la.y : 0, visible(ra) ? ra.y : 0) || Math.max(lh.y, rh.y);
  const pixelHeightFull = Math.max(5, (ankleYNorm - nose.y) * height);

  const midHipY = (lh.y + rh.y) / 2;
  const torsoPx = Math.max(5, (midHipY - nose.y) * height);

  // --- Calibration: user height dominates when plausible ---
  const scaleSources = [];
  const scaleWeights = [];
  const notes = [];

  const validHeight = userHeight && userHeight >= 120 && userHeight <= 230;
  if (validHeight) {
    scaleSources.push(userHeight / pixelHeightFull);
    scaleWeights.push(0.85);
    notes.push('Scale calibrated primarily from user-provided height.');
  } else {
    notes.push('No valid height provided; using population-average heuristics (less precise).');
  }

  const torsoFraction = 0.52;
  const assumedHeightForTorso = validHeight ? userHeight : 170.0;
  const torsoScale = (assumedHeightForTorso * torsoFraction) / torsoPx;
  scaleSources.push(torsoScale);
  scaleWeights.push(validHeight ? 0.15 : 0.5);

  if (!validHeight) {
    const avgShoulder = gender === 'female' ? 36.5 : 41.0;
    if (shoulderPx > 6) {
      scaleSources.push(avgShoulder / shoulderPx);
      scaleWeights.push(0.5);
    }
  }

  const totalW = scaleWeights.reduce((a, b) => a + b, 0);
  const pxToCm = scaleSources.reduce((sum, s, i) => sum + s * scaleWeights[i], 0) / totalW;

  const shoulderCm = shoulderPx * pxToCm;
  const hipWidthCm = hipPx * pxToCm;
  const heightCm = pixelHeightFull * pxToCm;

  const coeffs = coeffsFor(gender);
  let chestCm = coeffs.chestA * shoulderCm + coeffs.chestB;
  let waistCm = coeffs.waistA * (hipWidthCm * 0.86) + coeffs.waistB;
  let hipCm = coeffs.hipA * hipWidthCm + coeffs.hipB;

  chestCm = clamp(chestCm, 65, 160);
  waistCm = clamp(waistCm, 55, 160);
  hipCm = clamp(hipCm, 65, 160);
  const finalShoulderCm = clamp(shoulderCm, 25, 60);
  const finalHeightCm = clamp(heightCm, 120, 230);

  const confidence = confidenceScore(landmarks, validHeight, shoulderPx, hipPx, notes);

  return {
    shoulderCm: Math.round(finalShoulderCm),
    chestCm: Math.round(chestCm),
    waistCm: Math.round(waistCm),
    hipCm: Math.round(hipCm),
    heightCm: Math.round(finalHeightCm),
    confidence: Math.round(confidence * 10) / 10,
    notes,
  };
}

function confidenceScore(landmarks, hasUserHeight, shoulderPx, hipPx, notes) {
  const keyIndices = [0, 11, 12, 23, 24, 27, 28];
  const visValues = keyIndices
    .map((i) => landmarks[i])
    .filter((lm) => lm && typeof lm.visibility === 'number')
    .map((lm) => lm.visibility);
  const avgVis = visValues.length ? visValues.reduce((a, b) => a + b, 0) / visValues.length : 0.5;

  let score = 40 * avgVis;
  score += hasUserHeight ? 30 : 12;

  const pixelQuality = Math.min(1, Math.min(shoulderPx, hipPx) / 60);
  score += 30 * pixelQuality;

  if (pixelQuality < 0.5) {
    notes.push('Subject appears far from the camera; move closer for a more precise measurement.');
  }

  return clamp(score, 0, 100);
}

export { estimateMeasures };
