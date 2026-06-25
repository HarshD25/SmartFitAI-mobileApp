// server/server.js
//
// Responsibilities:
//   1. Serve the static frontend.
//   2. Expose POST /api/measure, which:
//        - Primarily proxies to the Python measurement service
//          (measure-backend/measure_server.py), which runs the
//          gender-aware anthropometric regression model plus optional
//          MiDaS depth correction - see that service for details.
//        - Falls back to a local, lighter-weight JS implementation of
//          the same model (server/anthropometry.js) if the Python
//          service is unreachable, so the app still works in a
//          Node-only environment (e.g. quick demos without a Python
//          environment set up) - clearly flagged in the response via
//          `engine: 'fallback'` so callers know accuracy is reduced.
//
// This file previously contained its own ad hoc measurement math
// (fixed shoulder*1.45 chest ratio, hip*0.95 waist ratio, applied
// identically regardless of declared gender). That logic has been
// replaced by the shared anthropometry module; see server/anthropometry.js
// for the rationale.

import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import bodyParser from 'body-parser';
import { estimateMeasures } from './anthropometry.js';
import { matchSize } from './sizeMatch.js';

const __dirname = path.resolve();
const app = express();
const PORT = process.env.PORT || 3000;
const MEASURE_BACKEND_URL = process.env.MEASURE_BACKEND_URL || 'http://localhost:5000';
const MEASURE_BACKEND_TIMEOUT_MS = Number(process.env.MEASURE_BACKEND_TIMEOUT_MS || 6000);

app.use(cors());
app.use(bodyParser.json({ limit: '15mb' }));

// Serve frontend (project root) statically
app.use('/', express.static(path.join(__dirname, '..', 'frontend')));

let SIZES_DB = {};
try {
  const p = path.join(__dirname, 'sizes_db.json');
  SIZES_DB = JSON.parse(fs.readFileSync(p, 'utf8'));
  console.log('Loaded sizes DB entries');
} catch (e) {
  console.warn('Could not load sizes_db.json', e);
}

function validateMeasureBody(body) {
  const { landmarks, height, gender } = body;
  if (!landmarks || !Array.isArray(landmarks) || landmarks.length < 29) {
    return 'landmarks must be an array of at least 29 pose points';
  }
  if (height !== null && height !== undefined) {
    const h = Number(height);
    if (Number.isNaN(h) || h <= 0 || h >= 300) {
      return 'height out of plausible range';
    }
  }
  if (gender && !['male', 'female'].includes(String(gender).toLowerCase())) {
    return "gender must be 'male' or 'female'";
  }
  return null;
}

async function tryPrimaryBackend(body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), MEASURE_BACKEND_TIMEOUT_MS);
  try {
    const resp = await fetch(`${MEASURE_BACKEND_URL}/measure`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => '');
      throw new Error(`primary backend responded ${resp.status}: ${text}`);
    }
    const json = await resp.json();
    return { ...json, engine: 'primary' };
  } finally {
    clearTimeout(timeout);
  }
}

function runFallback(body) {
  const { landmarks, height, gender, item, brand } = body;
  const measures = estimateMeasures(landmarks, 720, 1280, height, gender);
  if (!measures) {
    return { ok: false, error: 'failed to estimate measures (fallback engine)', engine: 'fallback' };
  }
  const size = matchSize(SIZES_DB, measures, gender, item, brand);
  return { ok: true, measures, size, confidence: measures.confidence, engine: 'fallback' };
}

app.post('/api/measure', async (req, res) => {
  try {
    const body = req.body || {};
    const validationError = validateMeasureBody(body);
    if (validationError) {
      return res.status(400).json({ ok: false, error: validationError });
    }

    try {
      const result = await tryPrimaryBackend(body);
      return res.json(result);
    } catch (primaryErr) {
      console.warn('Primary measurement backend unavailable, using fallback:', primaryErr.message);
      const fallbackResult = runFallback(body);
      const status = fallbackResult.ok ? 200 : 500;
      return res.status(status).json(fallbackResult);
    }
  } catch (e) {
    console.error('server /api/measure error', e);
    return res.status(500).json({ ok: false, error: e.toString() });
  }
});

app.get('/api/health', async (_req, res) => {
  let primaryAvailable = false;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    const resp = await fetch(`${MEASURE_BACKEND_URL}/health`, { signal: controller.signal });
    clearTimeout(timeout);
    primaryAvailable = resp.ok;
  } catch {
    primaryAvailable = false;
  }
  res.json({ ok: true, primaryBackend: primaryAvailable, fallbackAvailable: true });
});

app.listen(PORT, () => console.log(`SmartFitAI server running at http://localhost:${PORT}`));
