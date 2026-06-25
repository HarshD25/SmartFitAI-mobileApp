// frontend/test/resultsViz.test.js
//
// Uses jsdom for a minimal `document` so the SVG/div construction in
// resultsViz.js can be exercised directly - no native `canvas`
// package needed since nothing here does canvas drawing (that's
// scanEngine.js's job, tested separately).

import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

let renderResultsViz;
let rangeToPercent;

before(async () => {
  const dom = new JSDOM('<!DOCTYPE html><div id="root"></div>');
  global.document = dom.window.document;
  global.window = dom.window;
  ({ renderResultsViz, rangeToPercent } = await import('../src/resultsViz.js'));
});

function freshContainer() {
  const el = document.createElement('div');
  document.body.appendChild(el);
  return el;
}

const SAMPLE_RANGE = {
  key: 'chest',
  userValue: 101,
  min: 96,
  max: 104,
  sizes: [
    { alpha: '38', numeric: 38, value: 96 },
    { alpha: '40', numeric: 40, value: 100 },
    { alpha: '42', numeric: 42, value: 104 },
  ],
};

test('rangeToPercent: midpoint of the range lands near 50%', () => {
  const pct = rangeToPercent(100, 96, 104);
  assert.ok(Math.abs(pct - 50) < 1);
});

test('rangeToPercent: value at min is not flush against 0% (padding applied)', () => {
  const pct = rangeToPercent(96, 96, 104);
  assert.ok(pct > 0 && pct < 15);
});

test('rangeToPercent: value at max is not flush against 100% (padding applied)', () => {
  const pct = rangeToPercent(104, 96, 104);
  assert.ok(pct < 100 && pct > 85);
});

test('rangeToPercent: value below min clamps to 0, not negative', () => {
  const pct = rangeToPercent(50, 96, 104);
  assert.equal(pct, 0);
});

test('rangeToPercent: value above max clamps to 100', () => {
  const pct = rangeToPercent(500, 96, 104);
  assert.equal(pct, 100);
});

test('rangeToPercent: zero-width range (min === max) does not divide by zero', () => {
  const pct = rangeToPercent(100, 100, 100);
  assert.ok(Number.isFinite(pct));
});

test('renderResultsViz: does not throw with a full measures + range payload', () => {
  const container = freshContainer();
  assert.doesNotThrow(() => {
    renderResultsViz(
      container,
      { shoulderCm: 44, chestCm: 116, waistCm: 81, hipCm: 92, heightCm: 181 },
      SAMPLE_RANGE,
      'male'
    );
  });
  assert.ok(container.querySelector('svg.viz-silhouette-svg'));
  assert.ok(container.querySelector('.viz-range'));
});

test('renderResultsViz: omits the range bar when range is null', () => {
  const container = freshContainer();
  renderResultsViz(container, { chestCm: 100 }, null, 'male');
  assert.ok(container.querySelector('svg.viz-silhouette-svg'));
  assert.equal(container.querySelector('.viz-range'), null);
});

test('renderResultsViz: omits the range bar when there is only one size to compare', () => {
  const container = freshContainer();
  const singleSizeRange = { ...SAMPLE_RANGE, sizes: [SAMPLE_RANGE.sizes[0]] };
  renderResultsViz(container, { chestCm: 100 }, singleSizeRange, 'male');
  assert.equal(container.querySelector('.viz-range'), null);
});

test('renderResultsViz: omits callouts for measurements that are missing', () => {
  const container = freshContainer();
  renderResultsViz(container, { chestCm: 100 }, null, 'male'); // no shoulder/waist/hip
  const text = container.textContent;
  assert.match(text, /Chest/);
  assert.doesNotMatch(text, /Shoulder/);
  assert.doesNotMatch(text, /Waist/);
  assert.doesNotMatch(text, /Hip/);
});

test('renderResultsViz: clears previous content on re-render (no stacking on Scan Again)', () => {
  const container = freshContainer();
  renderResultsViz(container, { chestCm: 100 }, SAMPLE_RANGE, 'male');
  renderResultsViz(container, { chestCm: 110 }, SAMPLE_RANGE, 'female');
  assert.equal(container.querySelectorAll('svg.viz-silhouette-svg').length, 1);
});

test('renderResultsViz: does not throw when container is null', () => {
  assert.doesNotThrow(() => renderResultsViz(null, { chestCm: 100 }, SAMPLE_RANGE, 'male'));
});

test('renderResultsViz: range bar renders a tick per size plus a user marker', () => {
  const container = freshContainer();
  renderResultsViz(container, { chestCm: 100 }, SAMPLE_RANGE, 'male');
  const ticks = container.querySelectorAll('.viz-range-tick');
  const marker = container.querySelector('.viz-range-marker');
  assert.equal(ticks.length, SAMPLE_RANGE.sizes.length);
  assert.ok(marker);
});
