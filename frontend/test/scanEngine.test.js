// frontend/test/scanEngine.test.js
//
// Tests for the pure, side-effect-free logic in scanEngine.js: pose
// gating (fullBodyVisible, isStable), the median-of-buffer smoothing
// used before capture, and the skeleton-drawing data (colors,
// landmark index references). Anything that touches the camera,
// canvas, or MediaPipe's Pose class is excluded here since those
// require a real browser - this suite is about catching logic
// regressions in code that doesn't.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  colorForVisibility,
  fullBodyVisible,
  isStable,
  medianLandmarks,
  SKELETON_BONES,
  JOINT_INDICES,
} from '../src/scanEngine.js';

function fullPose(overrides = {}) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.9 }));
  lm[0] = { x: 0.5, y: 0.1, visibility: 0.9 };   // nose
  lm[11] = { x: 0.35, y: 0.2, visibility: 0.9 }; // L shoulder
  lm[12] = { x: 0.65, y: 0.2, visibility: 0.9 }; // R shoulder
  lm[23] = { x: 0.4, y: 0.5, visibility: 0.9 };  // L hip
  lm[24] = { x: 0.6, y: 0.5, visibility: 0.9 };  // R hip
  lm[27] = { x: 0.42, y: 0.95, visibility: 0.9 }; // L ankle
  lm[28] = { x: 0.58, y: 0.95, visibility: 0.9 }; // R ankle
  return Object.assign(lm, overrides);
}

test('colorForVisibility: high visibility maps to mint tones', () => {
  const c = colorForVisibility(0.9);
  assert.match(c.joint, /rgba\(79,175,124/);
});

test('colorForVisibility: mid visibility maps to marigold tones', () => {
  const c = colorForVisibility(0.5);
  assert.match(c.joint, /rgba\(235,177,61/);
});

test('colorForVisibility: low visibility maps to coral tones', () => {
  const c = colorForVisibility(0.2);
  assert.match(c.joint, /rgba\(232,90,69/);
});

test('colorForVisibility: thresholds align with the 0.45/0.7 gates used elsewhere', () => {
  // Boundary checks - these specific values matter because the rest
  // of scanEngine.js gates pose acceptance at 0.45 visibility, and the
  // skeleton color should tell the same story, not a different one.
  assert.match(colorForVisibility(0.45).joint, /235,177,61/);   // marigold
  assert.match(colorForVisibility(0.449).joint, /232,90,69/);   // coral
  assert.match(colorForVisibility(0.7).joint, /79,175,124/);    // mint
  assert.match(colorForVisibility(0.699).joint, /235,177,61/);  // marigold
});

test('fullBodyVisible: returns false for null/undefined', () => {
  assert.equal(fullBodyVisible(null), false);
  assert.equal(fullBodyVisible(undefined), false);
});

test('fullBodyVisible: true for a well-formed full-body pose', () => {
  assert.equal(fullBodyVisible(fullPose()), true);
});

test('fullBodyVisible: false when shoulders are missing/low-visibility', () => {
  const pose = fullPose();
  pose[11] = { x: 0.35, y: 0.2, visibility: 0.1 };
  assert.equal(fullBodyVisible(pose), false);
});

test('fullBodyVisible: false when neither ankle is visible', () => {
  const pose = fullPose();
  pose[27] = { x: 0.42, y: 0.95, visibility: 0.1 };
  pose[28] = { x: 0.58, y: 0.95, visibility: 0.1 };
  assert.equal(fullBodyVisible(pose), false);
});

test('fullBodyVisible: false when the pose is too compressed vertically (not full body in frame)', () => {
  const pose = fullPose();
  // Squash everything into a small vertical span, as if only the
  // upper body were visible (e.g. cropped or too close to camera).
  for (const p of pose) p.y = 0.5 + (p.y - 0.5) * 0.1;
  assert.equal(fullBodyVisible(pose), false);
});

test('isStable: true when no previous frame to compare against', () => {
  assert.equal(isStable(fullPose(), null), true);
});

test('isStable: true for two nearly identical frames', () => {
  const a = fullPose();
  const b = fullPose();
  assert.equal(isStable(a, b), true);
});

test('isStable: false when landmarks have moved significantly', () => {
  const a = fullPose();
  const b = fullPose();
  for (const p of b) p.x += 0.2; // large shift across the whole pose
  assert.equal(isStable(a, b), false);
});

test('medianLandmarks: returns the middle value across a buffer of frames', () => {
  const buf = [
    [{ x: 0.1, y: 0.1, visibility: 0.5 }],
    [{ x: 0.5, y: 0.5, visibility: 0.9 }],
    [{ x: 0.9, y: 0.9, visibility: 1.0 }],
  ];
  const med = medianLandmarks(buf);
  assert.equal(med[0].x, 0.5);
  assert.equal(med[0].y, 0.5);
  assert.equal(med[0].visibility, 0.9);
});

test('medianLandmarks: is robust to a single outlier frame', () => {
  const buf = [
    [{ x: 0.50, y: 0.50, visibility: 0.9 }],
    [{ x: 0.51, y: 0.49, visibility: 0.9 }],
    [{ x: 0.99, y: 0.01, visibility: 0.9 }], // one wild outlier (e.g. a misdetection)
  ];
  const med = medianLandmarks(buf);
  // Median should land near the two consistent frames, not be dragged
  // toward the outlier the way a mean would be.
  assert.ok(Math.abs(med[0].x - 0.51) < 0.02);
});

test('SKELETON_BONES: every referenced landmark index is within MediaPipe Pose range (0-32)', () => {
  for (const [a, b] of SKELETON_BONES) {
    assert.ok(a >= 0 && a <= 32, `bone index ${a} out of range`);
    assert.ok(b >= 0 && b <= 32, `bone index ${b} out of range`);
  }
});

test('JOINT_INDICES: every index is within MediaPipe Pose range and unique', () => {
  for (const i of JOINT_INDICES) {
    assert.ok(i >= 0 && i <= 32, `joint index ${i} out of range`);
  }
  assert.equal(JOINT_INDICES.length, new Set(JOINT_INDICES).size, 'duplicate joint indices found');
});

test('SKELETON_BONES: every bone endpoint is also drawn as a joint', () => {
  // Drawing a bone whose endpoint isn't in JOINT_INDICES would render
  // a line with no dot at one end - a visual inconsistency rather
  // than a crash, which is exactly the kind of thing easy to miss by
  // eye but easy to catch with a test.
  const jointSet = new Set(JOINT_INDICES);
  for (const [a, b] of SKELETON_BONES) {
    assert.ok(jointSet.has(a), `bone endpoint ${a} has no matching joint`);
    assert.ok(jointSet.has(b), `bone endpoint ${b} has no matching joint`);
  }
});
