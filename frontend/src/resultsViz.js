// src/resultsViz.js
//
// Renders the results-screen visualization: a body silhouette
// annotated with the estimated measurements, plus a horizontal bar
// showing where the user's measurement lands within the brand's full
// size range (when that data is available - see
// measure-backend/anthropometry/size_match.py's `range` field).
//
// Kept as its own module rather than inline in app.js because this is
// presentation logic with real layout math in it (silhouette
// proportions, marker positioning along a range), not flow control.

/**
 * @param {HTMLElement} container
 * @param {{shoulderCm?, chestCm?, waistCm?, hipCm?, heightCm?}} measures
 * @param {{key, userValue, min, max, sizes}|null} range
 * @param {'male'|'female'|string} gender
 */
export function renderResultsViz(container, measures, range, gender) {
  if (!container) return;
  container.innerHTML = '';
  container.appendChild(buildSilhouette(measures, gender));
  if (range && range.sizes && range.sizes.length > 1) {
    container.appendChild(buildRangeBar(range));
  }
}

// ---------------------------------------------------------------------------
// Body silhouette with measurement callouts
// ---------------------------------------------------------------------------
function buildSilhouette(measures, gender) {
  const wrap = document.createElement('div');
  wrap.className = 'viz-silhouette';

  // Reuses the same body outline path already used for the camera
  // screen's pose-alignment ghost guide, so the two screens share a
  // visual language instead of introducing a third unrelated shape.
  const bodyPath =
    'M100 35 c-12 0 -20 10 -20 22 v20 c-18 12 -24 30 -24 48 v70 c0 40 20 90 44 120 ' +
    'c24 -30 44 -80 44 -120 v-70 c0 -18 -6 -36 -24 -48 v-20 c0 -12 -8 -22 -20 -22z';

  const rows = [
    { label: 'Shoulder', value: measures.shoulderCm, y: 95, anchor: 'left' },
    { label: 'Chest', value: measures.chestCm, y: 150, anchor: 'right' },
    { label: 'Waist', value: measures.waistCm, y: 215, anchor: 'left' },
    { label: 'Hip', value: measures.hipCm, y: 260, anchor: 'right' },
  ].filter((r) => r.value !== undefined && r.value !== null);

  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  // Wide enough on both sides for a full "Shoulder 44 cm"-length
  // label to render without being clipped by the viewBox edge - the
  // body sits in the central ~200px band, with ~95px of label space
  // on each side (see buildCalloutRow's anchor math below, which this
  // viewBox width must stay in sync with).
  svg.setAttribute('viewBox', '0 0 380 340');
  svg.setAttribute('class', 'viz-silhouette-svg');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Body diagram with estimated measurements');

  const body = document.createElementNS(svgNS, 'path');
  body.setAttribute('d', bodyPath);
  body.setAttribute('transform', 'translate(90, 0)');
  body.setAttribute('fill', 'none');
  body.setAttribute('class', 'viz-body-outline');
  body.setAttribute('stroke-width', '2');
  body.setAttribute('opacity', '0.85');
  svg.appendChild(body);

  for (const row of rows) {
    svg.appendChild(buildCalloutRow(svgNS, row));
  }

  wrap.appendChild(svg);
  return wrap;
}

function buildCalloutRow(svgNS, { label, value, y, anchor }) {
  const group = document.createElementNS(svgNS, 'g');
  const bodyCenterX = 190; // matches translate(90,0) applied to a ~200-wide body path
  // Label text sits in the ~85px margin between the body silhouette's
  // edge and the viewBox boundary on each side (viewBox width 380,
  // body spans roughly x=110 to x=270 after the translate above).
  // text-anchor is set to grow *away* from the edge (start on the
  // left side, end on the right side) so longer labels like
  // "Shoulder 44 cm" extend inward toward the body rather than
  // outward past the viewBox edge, which is what clipped them before.
  const lineEndX = anchor === 'left' ? 92 : 288;
  const textX = anchor === 'left' ? 8 : 372;
  const bodyEdgeX = anchor === 'left' ? bodyCenterX - 44 : bodyCenterX + 44;

  const line = document.createElementNS(svgNS, 'line');
  line.setAttribute('x1', bodyEdgeX);
  line.setAttribute('y1', y);
  line.setAttribute('x2', lineEndX);
  line.setAttribute('y2', y);
  line.setAttribute('class', 'viz-callout-line');
  line.setAttribute('stroke-width', '1');
  line.setAttribute('stroke-dasharray', '3,3');
  group.appendChild(line);

  const dot = document.createElementNS(svgNS, 'circle');
  dot.setAttribute('cx', bodyEdgeX);
  dot.setAttribute('cy', y);
  dot.setAttribute('r', '3');
  dot.setAttribute('class', 'viz-callout-dot');
  group.appendChild(dot);

  const text = document.createElementNS(svgNS, 'text');
  text.setAttribute('x', textX);
  text.setAttribute('y', y);
  text.setAttribute('text-anchor', anchor === 'left' ? 'start' : 'end');
  text.setAttribute('dominant-baseline', 'middle');
  text.setAttribute('class', 'viz-callout-text');

  const labelSpan = document.createElementNS(svgNS, 'tspan');
  labelSpan.setAttribute('class', 'viz-callout-label');
  labelSpan.textContent = `${label} `;
  text.appendChild(labelSpan);

  const valueSpan = document.createElementNS(svgNS, 'tspan');
  valueSpan.setAttribute('class', 'viz-callout-value');
  valueSpan.textContent = `${value} cm`;
  text.appendChild(valueSpan);

  group.appendChild(text);
  return group;
}

// ---------------------------------------------------------------------------
// Size-range bar
// ---------------------------------------------------------------------------
/**
 * Maps a value within a (possibly padded) range to a 0-100 percentage
 * for horizontal positioning along the range bar. Padding keeps an
 * endpoint-exact value from rendering flush against the track edge,
 * and clamping keeps an out-of-range user measurement from rendering
 * off the visible track entirely.
 *
 * Pulled out as a standalone pure function (rather than inlined as a
 * closure inside buildRangeBar) specifically so it has unit test
 * coverage independent of any DOM/browser environment - the rest of
 * this module's DOM-construction code needs a real browser to
 * exercise meaningfully, but this is the one piece of actual layout
 * math worth getting right on its own.
 */
export function rangeToPercent(value, min, max, paddingFraction = 0.12) {
  const span = Math.max(1, max - min);
  const padding = span * paddingFraction;
  const lo = min - padding;
  const hi = max + padding;
  return Math.min(100, Math.max(0, ((value - lo) / (hi - lo)) * 100));
}

function buildRangeBar(range) {
  const wrap = document.createElement('div');
  wrap.className = 'viz-range';

  const heading = document.createElement('div');
  heading.className = 'viz-range-heading';
  const keyLabel = range.key === 'waist' ? 'Waist' : 'Chest';
  heading.textContent = `Where you land in this brand's ${keyLabel.toLowerCase()} range`;
  wrap.appendChild(heading);

  const track = document.createElement('div');
  track.className = 'viz-range-track';

  const pct = (v) => rangeToPercent(v, range.min, range.max);

  for (const size of range.sizes) {
    const tick = document.createElement('div');
    tick.className = 'viz-range-tick';
    tick.style.left = `${pct(size.value)}%`;

    const tickLabel = document.createElement('span');
    tickLabel.className = 'viz-range-tick-label';
    tickLabel.textContent = size.alpha ?? size.numeric ?? '';
    tick.appendChild(tickLabel);

    track.appendChild(tick);
  }

  if (range.userValue !== null && range.userValue !== undefined) {
    const marker = document.createElement('div');
    marker.className = 'viz-range-marker';
    marker.style.left = `${pct(range.userValue)}%`;
    marker.title = `You: ${range.userValue} cm`;
    track.appendChild(marker);
  }

  wrap.appendChild(track);
  return wrap;
}
