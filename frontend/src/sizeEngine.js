// src/sizeEngine.js
//
// Client-side helpers around sizing results and pose quality.
//
// NOTE: the actual size-matching *decision* (which brand size a set of
// measurements maps to) intentionally lives server-side only, in
// server/sizeMatch.js (Node fallback) and
// measure-backend/anthropometry/size_match.py (primary engine), so
// there is exactly one place per backend to update brand tables and
// matching rules. Duplicating that logic here in a third place was a
// real risk in this codebase (this file used to be an empty stub,
// presumably reserved for exactly that, which is why we keep it out).
//
// What belongs here instead: formatting helpers and a *pre-flight*
// pose-quality check that runs before we even send a frame to the
// backend, so users get fast feedback ("step back", "stand still")
// without round-tripping to a server first.

/**
 * Quick client-side pose-quality heuristic, run on each frame before
 * we consider locking in a capture. This intentionally mirrors a
 * subset of the server's confidence scoring (landmark visibility +
 * pixel-scale quality) so the UI can guide the user in real time,
 * without duplicating the actual measurement math itself.
 *
 * @param {Array<{x:number,y:number,visibility?:number}>} landmarks
 * @param {number} frameWidth
 * @param {number} frameHeight
 * @returns {{ ok: boolean, reason?: string, score: number }}
 */
export function assessPoseQuality(landmarks, frameWidth, frameHeight) {
  if (!landmarks || landmarks.length < 29) {
    return { ok: false, reason: 'No pose detected yet.', score: 0 };
  }

  const requiredIdx = [0, 11, 12, 23, 24];
  const required = requiredIdx.map((i) => landmarks[i]);
  const missing = required.some((lm) => !lm || (lm.visibility ?? 1) < 0.45);
  if (missing) {
    return { ok: false, reason: 'Move into frame so your shoulders and hips are visible.', score: 10 };
  }

  const hasAnkle =
    (landmarks[27] && (landmarks[27].visibility ?? 1) >= 0.45) ||
    (landmarks[28] && (landmarks[28].visibility ?? 1) >= 0.45);
  if (!hasAnkle) {
    return { ok: false, reason: 'Step back so your full body, including feet, is visible.', score: 20 };
  }

  const ls = landmarks[11], rs = landmarks[12], lh = landmarks[23], rh = landmarks[24];
  const shoulderPx = Math.hypot((ls.x - rs.x) * frameWidth, (ls.y - rs.y) * frameHeight);
  const hipPx = Math.hypot((lh.x - rh.x) * frameWidth, (lh.y - rh.y) * frameHeight);

  if (Math.min(shoulderPx, hipPx) < 30) {
    return { ok: false, reason: 'Move a little closer to the camera for a more precise scan.', score: 45 };
  }

  const avgVis =
    required.reduce((sum, lm) => sum + (lm.visibility ?? 1), 0) / required.length;
  const score = Math.round(60 + 40 * avgVis);

  return { ok: true, score: Math.min(100, score) };
}

/**
 * Format a confidence score (0-100) into a short, user-facing label.
 * Keeping this in one place avoids inconsistent wording across the
 * results screen and any future screens that show confidence.
 */
export function confidenceLabel(score) {
  if (score === null || score === undefined || Number.isNaN(score)) {
    return { label: 'Unknown', className: 'conf-unknown' };
  }
  if (score >= 80) return { label: 'High confidence', className: 'conf-high' };
  if (score >= 55) return { label: 'Moderate confidence', className: 'conf-medium' };
  return { label: 'Low confidence', className: 'conf-low' };
}

/**
 * Build the human-readable measurement summary line shown on the
 * results screen. Centralized so app.js stays focused on flow control
 * rather than string formatting.
 */
export function formatMeasurementSummary(measures) {
  if (!measures) return '';
  const parts = [
    ['Shoulder', measures.shoulderCm],
    ['Chest', measures.chestCm],
    ['Waist', measures.waistCm],
    ['Hip', measures.hipCm],
  ];
  return parts
    .map(([label, value]) => `${label}: ${value ?? '—'} cm`)
    .join(' • ');
}
