// server/sizeMatch.js
//
// Brand size-chart matching. Mirrors measure-backend/anthropometry/size_match.py
// so the Node fallback path and the primary Python service produce
// consistent results when run against the same sizes_db.json.

function matchSize(sizesDb, measures, gender = 'male', item = 'shirt', brand = '') {
  const genderKey = (gender || 'male').toLowerCase();
  const itemKey = (item || 'shirt').toLowerCase();

  try {
    const genderTable = sizesDb[genderKey] || {};
    const itemTable =
      (genderTable.topwear && genderTable.topwear[itemKey]) ||
      (genderTable.bottomwear && genderTable.bottomwear[itemKey]) ||
      null;

    if (itemTable && itemTable.length) {
      const isWaistItem = ['jeans', 'pant', 'pants', 'shorts', 'skirt', 'chinos', 'cargopants', 'tights'].includes(itemKey);
      const key = isWaistItem ? 'waist' : 'chest';
      const targetValue = isWaistItem ? measures.waistCm : measures.chestCm;

      let best = null;
      let bestScore = Number.POSITIVE_INFINITY;
      for (const row of itemTable) {
        const target = row[key] ?? row.chest ?? row.waist ?? row.hip;
        if (target === undefined || targetValue === undefined || targetValue === null) continue;
        const score = Math.abs(target - targetValue);
        if (score < bestScore) {
          bestScore = score;
          best = row;
        }
      }

      if (best) {
        return {
          alpha: best.alpha,
          numeric: best.numeric ?? null,
          fit: 'Brand match',
          score: Math.round(bestScore * 10) / 10,
          source: 'brand_table',
          range: buildRange(itemTable, key, targetValue),
        };
      }
    }
  } catch (e) {
    console.warn('matchSize error', e);
  }

  return genericFallback(measures, genderKey);
}

// Additive size-range summary for the results-screen "where do you
// land in this brand's range" visualization. Mirrors
// measure-backend/anthropometry/size_match.py's _build_range -
// callers that only need the matched size can ignore this field.
function buildRange(itemTable, key, userValue) {
  const sizes = itemTable
    .map((row) => ({
      alpha: row.alpha,
      numeric: row.numeric ?? null,
      value: row[key] ?? row.chest ?? row.waist ?? row.hip,
    }))
    .filter((s) => s.value !== undefined && s.value !== null);

  if (!sizes.length) return null;

  const values = sizes.map((s) => s.value);
  return {
    key,
    userValue: userValue ?? null,
    min: Math.min(...values),
    max: Math.max(...values),
    sizes,
  };
}

function genericFallback(measures, genderKey) {
  const c = measures.chestCm || 96;
  const bands =
    genderKey === 'female'
      ? [
          [82, 'XS', 32],
          [90, 'S', 34],
          [98, 'M', 36],
          [106, 'L', 38],
        ]
      : [
          [88, 'XS', 34],
          [96, 'S', 36],
          [104, 'M', 38],
          [112, 'L', 40],
        ];

  // Approximate a representative chest value per alpha size (4cm
  // below its cutoff) purely for the range visualization, matching
  // how the match itself is scored below.
  const bandSizes = bands.map(([limit, alpha, numeric]) => ({ alpha, numeric, value: limit - 4 }));
  bandSizes.push({ alpha: 'XL', numeric: genderKey === 'female' ? 40 : 42, value: 116 });
  const genericRange = {
    key: 'chest',
    userValue: c,
    min: bandSizes[0].value,
    max: bandSizes[bandSizes.length - 1].value,
    sizes: bandSizes,
  };

  for (const [limit, alpha, numeric] of bands) {
    if (c < limit) {
      return {
        alpha,
        numeric,
        fit: fitFor(alpha),
        score: Math.round(Math.abs(c - (limit - 4)) * 10) / 10,
        source: 'generic_fallback',
        range: genericRange,
      };
    }
  }

  const xl = genderKey === 'female' ? { alpha: 'XL', numeric: 40 } : { alpha: 'XL', numeric: 42 };
  return {
    ...xl,
    fit: 'Loose',
    score: Math.round(Math.abs(c - 116) * 10) / 10,
    source: 'generic_fallback',
    range: genericRange,
  };
}

function fitFor(alpha) {
  return { XS: 'Slim', S: 'Slim', M: 'Regular', L: 'Relaxed', XL: 'Loose' }[alpha] || 'Regular';
}

export { matchSize };
