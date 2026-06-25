// server/test/sizeMatch.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { matchSize } from '../sizeMatch.js';

const SAMPLE_DB = {
  male: {
    topwear: {
      shirt: [
        { alpha: '38', numeric: 38, chest: 96, waist: 82 },
        { alpha: '40', numeric: 40, chest: 100, waist: 86 },
        { alpha: '42', numeric: 42, chest: 104, waist: 90 },
      ],
    },
    bottomwear: {
      jeans: [
        { alpha: '30', numeric: 30, waist: 76, hip: 94 },
        { alpha: '32', numeric: 32, waist: 81, hip: 98 },
      ],
      chinos: [
        { alpha: '30', numeric: 30, waist: 77, hip: 96 },
        { alpha: '32', numeric: 32, waist: 82, hip: 100 },
      ],
    },
  },
};

test('picks closest chest match from brand table', () => {
  const result = matchSize(SAMPLE_DB, { chestCm: 101, waistCm: 87 }, 'male', 'shirt', '');
  assert.equal(result.alpha, '40');
  assert.equal(result.source, 'brand_table');
});

test('picks closest waist match for jeans', () => {
  const result = matchSize(SAMPLE_DB, { waistCm: 80, hipCm: 96 }, 'male', 'jeans', '');
  assert.equal(result.alpha, '32');
});

test('exact match returns exact size with zero score', () => {
  const result = matchSize(SAMPLE_DB, { chestCm: 96, waistCm: 82 }, 'male', 'shirt', '');
  assert.equal(result.alpha, '38');
  assert.equal(result.score, 0);
});

// Regression coverage for a real bug found when adding new bottomwear
// items (chinos, tights): the waist-vs-chest item detection list is a
// hardcoded array that has to stay in sync whenever a new
// waist-measured item is added to sizes_db.json. Missing an item from
// that list silently routes it to the generic fallback instead of the
// brand table that's actually right there.

test('chinos is treated as a waist item, not a chest item', () => {
  const result = matchSize(SAMPLE_DB, { waistCm: 80, hipCm: 97 }, 'male', 'chinos', '');
  assert.equal(result.source, 'brand_table');
  assert.equal(result.range.key, 'waist');
});

test('unknown item falls back to generic bands', () => {
  const result = matchSize(SAMPLE_DB, { chestCm: 100 }, 'male', 'totally_unknown', '');
  assert.equal(result.source, 'generic_fallback');
  assert.ok(['XS', 'S', 'M', 'L', 'XL'].includes(result.alpha));
});

test('empty DB does not throw', () => {
  const result = matchSize({}, { chestCm: 100 }, 'male', 'shirt', '');
  assert.equal(result.source, 'generic_fallback');
  assert.ok(result.alpha);
});

test('fallback bands are monotonic with chest size', () => {
  const order = ['XS', 'S', 'M', 'L', 'XL'];
  let prevIdx = -1;
  for (const chest of [75, 85, 93, 101, 109, 120]) {
    const result = matchSize({}, { chestCm: chest }, 'male', 'unknown', '');
    const idx = order.indexOf(result.alpha);
    assert.ok(idx >= prevIdx);
    prevIdx = idx;
  }
});

// Range data powers the results-screen "where do you land in this
// brand's range" visualization. Mirrors
// measure-backend/tests/test_size_match.py::TestSizeRangeForVisualization.

test('brand table range spans all rows', () => {
  const result = matchSize(SAMPLE_DB, { chestCm: 101, waistCm: 87 }, 'male', 'shirt', '');
  assert.equal(result.range.key, 'chest');
  assert.equal(result.range.min, 96);
  assert.equal(result.range.max, 104);
  assert.deepEqual(result.range.sizes.map((s) => s.alpha), ['38', '40', '42']);
});

test('brand table range uses waist for waist items', () => {
  const result = matchSize(SAMPLE_DB, { waistCm: 80, hipCm: 96 }, 'male', 'jeans', '');
  assert.equal(result.range.key, 'waist');
  assert.equal(result.range.min, 76);
  assert.equal(result.range.max, 81);
});

test('brand table range reports user value', () => {
  const result = matchSize(SAMPLE_DB, { chestCm: 101, waistCm: 87 }, 'male', 'shirt', '');
  assert.equal(result.range.userValue, 101);
});

test('generic fallback still includes a usable range', () => {
  const result = matchSize({}, { chestCm: 100 }, 'male', 'unknown', '');
  assert.ok(result.range);
  assert.equal(result.range.key, 'chest');
  assert.ok(result.range.min < result.range.max);
  assert.ok(result.range.sizes.length >= 4);
});

test('range sizes are in ascending order', () => {
  const result = matchSize(SAMPLE_DB, { chestCm: 101 }, 'male', 'shirt', '');
  const values = result.range.sizes.map((s) => s.value);
  const sorted = [...values].sort((a, b) => a - b);
  assert.deepEqual(values, sorted);
});
