// src/app.js
import { Voice } from './audioGuide.js';
import { showStep, setStatus } from './ui.js';
import * as ScanEngine from './scanEngine.js';
import { confidenceLabel, formatMeasurementSummary } from './sizeEngine.js';
import { renderResultsViz } from './resultsViz.js';

// Same-origin by default (server.js serves the frontend and exposes
// /api/measure on the same port), so this works whether the app is
// opened via localhost during development or deployed elsewhere.
// Override by setting window.SMARTFIT_API_BASE before this script
// loads, e.g. for pointing a static frontend at a remote API host.
const API_BASE = window.SMARTFIT_API_BASE || '';

const heightEl = document.getElementById('height');
const genderEl = document.getElementById('gender');
const categoryEl = document.getElementById('category');
const itemEl = document.getElementById('item');
const brandEl = document.getElementById('brand');

const getStartedBtn = document.getElementById('getStartedBtn');
const nextToCameraBtn = document.getElementById('nextToCamera');
const cancelScanBtn = document.getElementById('cancelScan');
const flipCameraBtn = document.getElementById('flipCamera');
const scanAgainBtn = document.getElementById('scanAgain');
const goHomeBtn = document.getElementById('goHome');

const cameraEl = document.getElementById('camera');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');

const resBrand = document.getElementById('resBrand');
const resItem = document.getElementById('resItem');
const resSize = document.getElementById('resSize');
const resFit = document.getElementById('resFit');
const measSummary = document.getElementById('measSummary');
const resultsViz = document.getElementById('resultsViz');

const ITEMS_MAP = {
  topwear: ['Shirt','T-Shirt','Polo','Henley','Kurta','Top'],
  bottomwear: ['Jeans','Pant','Chinos','Cargo Pants','Shorts','Skirt'],
  ethnic: ['Kurta','Kurti','Saree','Lehenga'],
  outerwear: ['Jacket','Blazer','Coat','Hoodie'],
  activewear: ['Track Jacket','Tights','Sports Bra'],
  others: ['Dress','Sweater','Tracksuit']
};

function normalizeItem(raw) {
  if (!raw) return 'shirt';
  const key = raw.toLowerCase().replace(/\s+/g,'').replace(/-/g,'');
  const map = {
    "tshirt":"tshirt","t-shirt":"tshirt","polo":"tshirt","henley":"tshirt",
    "shirt":"shirt","kurta":"shirt","kurti":"shirt","top":"shirt",
    "jeans":"jeans","pant":"jeans","pants":"jeans","shorts":"jeans","skirt":"jeans",
    "chinos":"chinos","cargopants":"chinos",
    "saree":"saree","lehenga":"lehenga",
    "jacket":"jacket","coat":"jacket","hoodie":"jacket",
    "blazer":"blazer",
    "trackjacket":"trackjacket","tights":"tights","sportsbra":"sportsbra",
    "dress":"dress","sweater":"sweater","tracksuit":"tracksuit"
  };
  return map[key] || key || 'shirt';
}

categoryEl.addEventListener('change', ()=>{
  itemEl.innerHTML = '<option value="">What to size?</option>';
  (ITEMS_MAP[categoryEl.value]||[]).forEach(it=>{
    const o = document.createElement('option');
    o.value = it.toLowerCase();
    o.textContent = it;
    itemEl.appendChild(o);
  });
});

function updateGhostGuide() {
  const ghostMale = document.getElementById('ghostMale');
  const ghostFemale = document.getElementById('ghostFemale');
  if (!ghostMale || !ghostFemale) return;
  const isFemale = genderEl.value === 'female';
  ghostMale.style.display = isFemale ? 'none' : '';
  ghostFemale.style.display = isFemale ? '' : 'none';
}

getStartedBtn.addEventListener('click', ()=>{
  document.getElementById('welcomeScreen').classList.add('hide');
  document.getElementById('appCard').classList.add('show');
  setTimeout(()=> document.getElementById('welcomeScreen').style.display = 'none', 650);
  showStep(1);
});

nextToCameraBtn.addEventListener('click', async ()=>{
  if (!genderEl.value || !categoryEl.value || !itemEl.value || !brandEl.value) {
    alert('Almost there — just fill in gender, category, item and brand first!');
    return;
  }
  const heightRaw = heightEl.value.trim();
  if (heightRaw) {
    const h = parseFloat(heightRaw);
    if (Number.isNaN(h) || h < 120 || h > 230) {
      alert('That height doesn\'t look right — enter your height in cm (120–230), or leave it blank.');
      heightEl.focus();
      return;
    }
  }
  updateGhostGuide();
  showStep(2);
  await ScanEngine.initScan();
  await ScanEngine.startCamera(cameraEl, overlayCtx, onCaptureReady);
});

cancelScanBtn.addEventListener('click', ()=>{
  ScanEngine.stopCamera(cameraEl);
  showStep(1);
});

flipCameraBtn.addEventListener('click', async ()=>{
  await ScanEngine.switchCamera();
});

scanAgainBtn.addEventListener('click', async ()=>{
  ScanEngine.stopCamera(cameraEl);
  updateGhostGuide();
  showStep(2);
  await ScanEngine.initScan();
  await ScanEngine.startCamera(cameraEl, overlayCtx, onCaptureReady);
});

goHomeBtn.addEventListener('click', ()=>{
  ScanEngine.stopCamera(cameraEl);
  showStep(1);
});

async function onCaptureReady(payload) {
  try {
    setStatus('Sending data for measurement', 'Please wait...');
    Voice.speak('Sending data for measurement');

    const body = {
      image: payload.image,
      landmarks: payload.landmarks,
      height: parseFloat(heightEl.value) || null,
      gender: genderEl.value,
      item: normalizeItem(itemEl.value),
      brand: brandEl.value
    };

    const resp = await fetch(`${API_BASE}/api/measure`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });

    if (!resp.ok) {
      const txt = await resp.text();
      console.error('measure backend error', txt);
      alert("Something went wrong on our end — mind trying that again?");
      showStep(1);
      return;
    }

    const json = await resp.json();
    if (!json || !json.ok) {
      alert("Couldn't quite get a clean read — try again with a bit more light and your full body in frame.");
      showStep(1);
      return;
    }

    const measures = json.measures || {};
    const size = json.size || {};
    const confidence = json.confidence ?? measures.confidence ?? null;
    const conf = confidenceLabel(confidence);

    resBrand.textContent = brandEl.value || '—';
    resItem.textContent = itemEl.options[itemEl.selectedIndex]?.text || '—';
    resSize.textContent = (size.numeric ? `${size.numeric} — ${size.alpha}` : (size.alpha || '—'));
    resFit.textContent = `${size.fit || 'Recommended'} • Est. Height: ${measures.heightCm || '—'} cm`;

    renderResultsViz(resultsViz, measures, size.range || null, genderEl.value);

    const summaryParts = [
      formatMeasurementSummary(measures),
      confidence !== null ? `${conf.label} (${Math.round(confidence)}%)` : null,
      json.engine === 'fallback' ? 'Measured using local fallback engine — for best accuracy, ensure the measurement service is running.' : null,
    ].filter(Boolean);
    measSummary.textContent = summaryParts.join(' • ');
    measSummary.className = conf.className;

    Voice.speak(`Scan complete. Recommended size ${size.alpha || size.numeric || 'M'}`);
    showStep(3);
  } catch (err) {
    console.error('onCaptureReady error', err);
    alert("We couldn't reach the measurement service — make sure it's running and try again.");
    showStep(1);
  }
}

/* warm up */
(async function warmUp() {
  try { await ScanEngine.initScan(); } catch(e) { console.warn('warmup fail', e); }
})();
