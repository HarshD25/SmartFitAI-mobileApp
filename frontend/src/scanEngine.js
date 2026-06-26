import { Voice } from './audioGuide.js';
import { assessPoseQuality } from './sizeEngine.js';

const STABLE_REQUIRED = 8;
const MEDIAN_FRAMES = 16;
const STABILITY_JITTER = 0.035;
const AUTO_FORCE_AFTER = 36;

let pose = null;
let camUtil = null;
let frameBuffer = [];
let stableFrames = 0;
let attempts = 0;
let scanning = false;
let lastState = '';

function speakState(key, message) {
  if (key === lastState) return;
  lastState = key;
  Voice.speak(message);
}

// MediaPipe Pose landmark connections we draw as skeleton "bones".
// Limited to the upper/lower body segments relevant to body
// measurement (we don't draw the face mesh points at all) so the
// overlay reads as a clean body skeleton rather than a noisy point
// cloud - matches what people expect from a "real" CV product.
export const SKELETON_BONES = [
  // Shoulders / torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left arm
  [11, 13], [13, 15],
  // Right arm
  [12, 14], [14, 16],
  // Left leg
  [23, 25], [25, 27], [27, 29], [27, 31],
  // Right leg
  [24, 26], [26, 28], [28, 30], [28, 32],
];

// Landmarks worth drawing as joints - same set the bones above touch,
// plus the nose so the head position is visible.
export const JOINT_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

// Pulled from the same coral/marigold/mint/ink palette as styles.css
// (see :root there) rather than generic neon CV-demo colors - canvas
// 2D can't read CSS variables directly, so these are the literal
// values for --mint, --marigold-dim, and --coral-dim, boosted slightly
// in opacity since they need to stay legible against arbitrary live
// video, not just the cream card background the rest of the UI sits on.
const SKELETON_PALETTE = {
  good: { joint: 'rgba(79,175,124,0.95)', bone: 'rgba(79,175,124,0.85)' },        // mint
  borderline: { joint: 'rgba(235,177,61,0.95)', bone: 'rgba(235,177,61,0.85)' },  // marigold-dim
  poor: { joint: 'rgba(232,90,69,0.92)', bone: 'rgba(232,90,69,0.8)' },           // coral-dim
};

export function colorForVisibility(vis) {
  // Mint = confidently tracked, marigold = borderline, coral = unreliable.
  // Thresholds match the 0.45 visibility gate used elsewhere in this
  // file, so the skeleton's color tells the same story as the actual
  // pass/fail logic instead of just looking pretty.
  if (vis >= 0.7) return SKELETON_PALETTE.good;
  if (vis >= 0.45) return SKELETON_PALETTE.borderline;
  return SKELETON_PALETTE.poor;
}

function drawProgressRing(overlayCtx, w, h, progress) {
  // progress: 0..1, or null to hide entirely (no body / unstable /
  // not yet scanning). Drawn around the torso center so it reads as
  // "the system is accumulating a confident capture" rather than a
  // generic loading spinner disconnected from the body.
  if (progress === null || progress <= 0) return;
  const cx = w / 2;
  const cy = h * 0.34; // roughly chest height within the camera guide
  const radius = Math.min(w, h) * 0.16;

  overlayCtx.save();
  overlayCtx.lineWidth = 5;
  overlayCtx.lineCap = 'round';
  overlayCtx.strokeStyle = 'rgba(255,251,245,0.3)'; // --cream, translucent track
  overlayCtx.beginPath();
  overlayCtx.arc(cx, cy, radius, 0, Math.PI * 2);
  overlayCtx.stroke();

  overlayCtx.strokeStyle = progress >= 1 ? 'rgba(79,175,124,0.95)' : 'rgba(255,111,89,0.9)'; // mint : coral
  overlayCtx.beginPath();
  overlayCtx.arc(cx, cy, radius, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, progress));
  overlayCtx.stroke();
  overlayCtx.restore();
}

function drawLandmarks(lm, videoEl, overlayCtx, progress = null) {
  if (!overlayCtx || !videoEl) return;
  const dpr = window.devicePixelRatio || 1;
  const w = videoEl.clientWidth;
  const h = videoEl.clientHeight;
  const targetW = Math.round(w * dpr);
  const targetH = Math.round(h * dpr);
  if (overlayCtx.canvas.width !== targetW) overlayCtx.canvas.width = targetW;
  if (overlayCtx.canvas.height !== targetH) overlayCtx.canvas.height = targetH;
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  overlayCtx.clearRect(0, 0, w, h);

  drawProgressRing(overlayCtx, w, h, progress);

  if (!lm || !lm.length) return;

  const px = (p) => p.x * w;
  const py = (p) => p.y * h;
  const vis = (p) => p?.visibility ?? 1;

  // Bones first so joints render on top of the lines that meet them.
  overlayCtx.lineWidth = 3;
  overlayCtx.lineCap = 'round';
  for (const [a, b] of SKELETON_BONES) {
    const pa = lm[a], pb = lm[b];
    if (!pa || !pb) continue;
    // A bone is only as trustworthy as its weaker endpoint - color it
    // accordingly rather than always drawing it confident-green.
    const boneVis = Math.min(vis(pa), vis(pb));
    overlayCtx.strokeStyle = colorForVisibility(boneVis).bone;
    overlayCtx.beginPath();
    overlayCtx.moveTo(px(pa), py(pa));
    overlayCtx.lineTo(px(pb), py(pb));
    overlayCtx.stroke();
  }

  // Joints on top, each with a thin ink outline so the mark reads
  // clearly against arbitrary video - keeps the dots looking like
  // deliberate UI rather than blending into skin tones or clothing.
  for (const i of JOINT_INDICES) {
    const p = lm[i];
    if (!p) continue;
    const x = px(p), y = py(p), r = i === 0 ? 6 : 4.5;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, r, 0, Math.PI * 2);
    overlayCtx.fillStyle = colorForVisibility(vis(p)).joint;
    overlayCtx.fill();
    overlayCtx.lineWidth = 1;
    overlayCtx.strokeStyle = 'rgba(45,42,110,0.5)'; // --ink, translucent
    overlayCtx.stroke();
  }
}

export function fullBodyVisible(lm) {
  if (!lm) return false;
  const vis = i => (lm[i] && (lm[i].visibility ?? 1) > 0.45);
  if (!vis(0) || !vis(11) || !vis(12) || !vis(23) || !vis(24)) return false;
  if (!(vis(27) || vis(28))) return false;
  const ys = lm.map(p => p?.y ?? 0);
  const span = Math.max(...ys) - Math.min(...ys);
  return span > 0.46;
}

export function isStable(newLm, lastLm) {
  if (!lastLm) return true;
  let sum = 0;
  for (let i=0;i<newLm.length;i++) {
    sum += Math.abs(newLm[i].x - lastLm[i].x) + Math.abs(newLm[i].y - lastLm[i].y);
  }
  const avg = sum / newLm.length;
  return avg < STABILITY_JITTER;
}

export function medianLandmarks(buf) {
  const L = buf[0].length;
  const med = [];
  for (let i=0;i<L;i++) {
    const xs = buf.map(b => b[i].x).sort((a,b)=>a-b);
    const ys = buf.map(b => b[i].y).sort((a,b)=>a-b);
    const vs = buf.map(b => (b[i].visibility ?? 1)).sort((a,b)=>a-b);
    med.push({ x: xs[Math.floor(xs.length/2)], y: ys[Math.floor(ys.length/2)], visibility: vs[Math.floor(vs.length/2)] });
  }
  return med;
}

function autoForceIfNeeded(onCapture) {
  attempts++;
  if (attempts >= AUTO_FORCE_AFTER && frameBuffer.length >= 3) {
    Voice.speak('I will try to scan with the current pose');
    const med = medianLandmarks(frameBuffer);
    onCapture({ image: null, landmarks: med, autoForced: true });
    attempts = 0;
  }
}

export async function initScan() {
  if (pose) return;
  pose = new Pose({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${file}`
  });
  pose.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.55, minTrackingConfidence: 0.55 });
}

let currentFacingMode = 'environment';
let currentOnCapture = null;
let currentVideoEl = null;
let currentOverlayCtx = null;

export async function startCamera(videoEl, overlayCtx, onCapture, facingMode = currentFacingMode) {
  currentFacingMode = facingMode;
  currentOnCapture = onCapture;
  currentVideoEl = videoEl;
  currentOverlayCtx = overlayCtx;

  if (!pose) await initScan();
  pose.onResults(results => {
    try {
      const lm = results.poseLandmarks;

      if (!scanning) {
        drawLandmarks(lm || [], videoEl, overlayCtx, null);
        return;
      }

      if (!lm) {
        stableFrames = 0; frameBuffer = [];
        drawLandmarks([], videoEl, overlayCtx, null);
        setStatusAndSpeak('Hop into frame!', '', 'no_body');
        return;
      }

      if (!fullBodyVisible(lm)) {
        stableFrames = 0; frameBuffer = [];
        drawLandmarks(lm, videoEl, overlayCtx, null);
        setStatusAndSpeak('Show your whole self', 'Head to feet inside the dashed outline', 'partial');
        return;
      }

      const last = frameBuffer.length ? frameBuffer[frameBuffer.length-1] : null;
      if (!isStable(lm, last || lm)) {
        stableFrames = 0; frameBuffer = [];
        drawLandmarks(lm, videoEl, overlayCtx, 0);
        setStatusAndSpeak('Freeze!', 'Just a moment of stillness', 'hold');
        return;
      }

      const snap = lm.map(p => ({ x: p.x, y: p.y, visibility: p.visibility ?? 1 }));
      frameBuffer.push(snap);
      if (frameBuffer.length > MEDIAN_FRAMES) frameBuffer.shift();

      stableFrames++;
      drawLandmarks(lm, videoEl, overlayCtx, stableFrames / STABLE_REQUIRED);
      const quality = assessPoseQuality(snap, videoEl.videoWidth || videoEl.clientWidth, videoEl.videoHeight || videoEl.clientHeight);
      setStatusAndSpeak(`Nice and still (${stableFrames}/${STABLE_REQUIRED}) • ${quality.score}%`, 'Almost there, keep holding', 'capturing');

      if (stableFrames >= STABLE_REQUIRED && frameBuffer.length >= Math.min(MEDIAN_FRAMES, STABLE_REQUIRED)) {
        setStatusAndSpeak('Got it! ✨', '', 'scanning');
        Voice.speak('Got it, scanning now');
        const med = medianLandmarks(frameBuffer);
        try {
          const tmp = document.createElement('canvas');
          tmp.width = videoEl.videoWidth || videoEl.clientWidth;
          tmp.height = videoEl.videoHeight || videoEl.clientHeight;
          const c = tmp.getContext('2d');
          c.drawImage(videoEl, 0, 0, tmp.width, tmp.height);
          const jpg = tmp.toDataURL('image/jpeg', 0.92);
          onCapture({ image: jpg, landmarks: med, autoForced: false });
        } catch (e) {
          onCapture({ image: null, landmarks: med, autoForced: false });
        }
        stopCamera(videoEl);
        return;
      }

      autoForceIfNeeded(onCapture);
    } catch (e) {
      console.error('pose handler error', e);
    }
  });

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 720, height: 1280, facingMode: currentFacingMode }, audio: false });
    videoEl.srcObject = stream;
    camUtil = new Camera(videoEl, { onFrame: async () => { if (pose) await pose.send({ image: videoEl }); }, width: 720, height: 1280 });
    await camUtil.start();
    scanning = true; stableFrames = 0; frameBuffer = []; attempts = 0; lastState = '';
    setStatusAndSpeak('Looking for you…', 'Stand back so your full body is in the outline', 'start');
    Voice.speak('Starting camera. Please stand facing the camera and keep your full body inside the guide.');
  } catch (err) {
    console.error('camera start error', err);
    setStatusAndSpeak('Can\'t reach the camera', 'Check your browser permissions and try again', 'camera_error');
    Voice.speak('Please allow camera access in your browser.');
  }
}

function setStatusAndSpeak(status, subtitle, key) {
  const statusEl = document.getElementById('status');
  const subtitleEl = document.getElementById('subtitle');
  if (statusEl) statusEl.textContent = status;
  if (subtitleEl) subtitleEl.textContent = subtitle || '';
  speakState(key, `${status} ${subtitle}`.trim());
}

export function stopCamera(videoEl) {
  try { if (videoEl && videoEl.srcObject) videoEl.srcObject.getTracks().forEach(t => t.stop()); } catch(e){}
  if (camUtil && camUtil.stop) camUtil.stop();
  camUtil = null;
  scanning = false;
  stableFrames = 0;
  frameBuffer = [];
  attempts = 0;
}

/**
 * Switch between front ('user') and back ('environment') camera
 * mid-scan. The back camera is the default (better quality, lets a
 * friend hold the phone and frame a full-body shot properly), but the
 * front camera is genuinely useful for a solo scan since it's the
 * only way to see the ghost guide and skeleton overlay on yourself
 * while positioning. Re-running startCamera with the stored callback
 * and elements re-establishes the pose handler and stream cleanly,
 * rather than trying to mutate the existing stream's track in place
 * (switching facingMode on a live track isn't reliably supported
 * across browsers, where a fresh getUserMedia call is).
 */
export async function switchCamera() {
  if (!currentVideoEl || !currentOnCapture) return;
  const nextMode = currentFacingMode === 'environment' ? 'user' : 'environment';
  stopCamera(currentVideoEl);
  await startCamera(currentVideoEl, currentOverlayCtx, currentOnCapture, nextMode);
}
