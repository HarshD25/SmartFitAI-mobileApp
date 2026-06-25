// server/test/fixtures.js
//
// Synthetic, geometrically plausible "standing person" pose landmarks
// for deterministic testing without needing a real camera/MediaPipe
// runtime.

export function makeStandingPose({
  shoulderWidthNorm = 0.38,
  hipWidthNorm = 0.30,
  headYNorm = 0.08,
  hipYNorm = 0.52,
  ankleYNorm = 0.95,
  centerX = 0.5,
  visibility = 0.95,
} = {}) {
  const lm = Array.from({ length: 33 }, () => ({ x: centerX, y: 0.5, visibility }));

  lm[0] = { x: centerX, y: headYNorm, visibility }; // nose
  lm[11] = { x: centerX - shoulderWidthNorm / 2, y: headYNorm + 0.07, visibility }; // L shoulder
  lm[12] = { x: centerX + shoulderWidthNorm / 2, y: headYNorm + 0.07, visibility }; // R shoulder
  lm[23] = { x: centerX - hipWidthNorm / 2, y: hipYNorm, visibility }; // L hip
  lm[24] = { x: centerX + hipWidthNorm / 2, y: hipYNorm, visibility }; // R hip
  lm[27] = { x: centerX - 0.04, y: ankleYNorm, visibility }; // L ankle
  lm[28] = { x: centerX + 0.04, y: ankleYNorm, visibility }; // R ankle

  return lm;
}
