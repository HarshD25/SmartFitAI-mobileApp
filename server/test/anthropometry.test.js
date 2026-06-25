// server/test/anthropometry.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { estimateMeasures } from '../anthropometry.js';
import { makeStandingPose } from './fixtures.js';

test('returns null for missing landmarks', () => {
  assert.equal(estimateMeasures(null, 720, 1280), null);
  assert.equal(estimateMeasures([], 720, 1280), null);
});

test('returns null for too few landmarks', () => {
  const pose = makeStandingPose().slice(0, 10);
  assert.equal(estimateMeasures(pose, 720, 1280), null);
});

test('returns plausible measurements for a valid pose', () => {
  const pose = makeStandingPose();
  const result = estimateMeasures(pose, 720, 1280, 175, 'male');
  assert.ok(result);
  assert.ok(result.heightCm >= 130 && result.heightCm <= 220);
  assert.ok(result.chestCm >= 60 && result.chestCm <= 150);
  assert.ok(result.confidence >= 0 && result.confidence <= 100);
});

test('low visibility landmarks are rejected', () => {
  const pose = makeStandingPose({ visibility: 0.1 });
  assert.equal(estimateMeasures(pose, 720, 1280, 175, 'male'), null);
});

test('user height dominates calibration over population heuristics', () => {
  const pose = makeStandingPose();
  const short = estimateMeasures(pose, 720, 1280, 150, 'male');
  const tall = estimateMeasures(pose, 720, 1280, 200, 'male');
  assert.ok(tall.heightCm > short.heightCm);
  const ratio = tall.heightCm / short.heightCm;
  assert.ok(ratio >= 1.15 && ratio <= 1.45, `ratio was ${ratio}`);
});

test('gender changes circumference estimates for identical geometry', () => {
  const pose = makeStandingPose();
  const male = estimateMeasures(pose, 720, 1280, 170, 'male');
  const female = estimateMeasures(pose, 720, 1280, 170, 'female');
  assert.notDeepEqual(
    [male.chestCm, male.waistCm, male.hipCm],
    [female.chestCm, female.waistCm, female.hipCm]
  );
});

test('confidence is higher when user height is supplied', () => {
  const pose = makeStandingPose();
  const withHeight = estimateMeasures(pose, 720, 1280, 170, 'male');
  const withoutHeight = estimateMeasures(pose, 720, 1280, null, 'male');
  assert.ok(withHeight.confidence > withoutHeight.confidence);
});

test('implausible height is treated identically to no height at all', () => {
  // Regression test for a real calibration bug found during
  // development (see measure-backend/anthropometry/model.py for the
  // full writeup): an implausible height value must never partially
  // leak into the fallback calibration path.
  const pose = makeStandingPose();
  const bad = estimateMeasures(pose, 720, 1280, 999, 'male');
  const negative = estimateMeasures(pose, 720, 1280, -5, 'male');
  const none = estimateMeasures(pose, 720, 1280, null, 'male');
  assert.equal(bad.heightCm, none.heightCm);
  assert.equal(bad.chestCm, none.chestCm);
  assert.equal(negative.heightCm, none.heightCm);
});

test('realistic adult proportions yield realistic output', () => {
  const pose = makeStandingPose();
  const result = estimateMeasures(pose, 720, 1280, 178, 'male');
  assert.ok(result.heightCm >= 165 && result.heightCm <= 192);
  assert.ok(result.shoulderCm >= 36 && result.shoulderCm <= 50);
  assert.ok(result.chestCm >= 90 && result.chestCm <= 120);
  assert.ok(result.waistCm >= 70 && result.waistCm <= 100);
  assert.ok(result.hipCm >= 85 && result.hipCm <= 115);
});

test('confidence is lower for a distant subject (small pixel spans)', () => {
  const close = makeStandingPose({ shoulderWidthNorm: 0.38, hipWidthNorm: 0.30 });
  const far = makeStandingPose({ shoulderWidthNorm: 0.04, hipWidthNorm: 0.03 });
  const closeResult = estimateMeasures(close, 720, 1280, 170, 'male');
  const farResult = estimateMeasures(far, 720, 1280, 170, 'male');
  assert.ok(closeResult.confidence > farResult.confidence);
});
