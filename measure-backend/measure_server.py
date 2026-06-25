"""
measure_server.py

Flask backend - the primary measurement engine for SmartFitAI.

Pipeline:
 1. Accept a base64 JPEG frame + median pose landmarks from the frontend.
 2. Run MiDaS (monocular depth estimation) to derive a relative body
    "thickness" signal used to correct frontal-width-based circumference
    estimates (see anthropometry/model.py for why this matters).
 3. Run the gender-aware anthropometric regression model to compute
    shoulder/chest/waist/hip/height estimates with a confidence score.
 4. Match measurements to brand size tables (server/sizes_db.json).
 5. Return JSON: { ok, measures, size, confidence }

This server previously existed in the codebase but was never called by
the frontend - the live app only used the much weaker Node.js fixed-ratio
heuristic. It is now the primary measurement endpoint; see
server/server.js for the lightweight Node fallback used only when this
service is unreachable.
"""

import os
import io
import json
import base64
import logging
from typing import Any, Dict, List, Optional

from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch

from anthropometry import Landmark, estimate as estimate_measures
from anthropometry.size_match import match_size

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("measure_server")

# --- Config ---
PORT = int(os.environ.get("MEASURE_PORT", 5000))
SIZES_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server", "sizes_db.json"))
MAX_IMAGE_DIM = 1600  # guard against absurdly large uploads slowing down depth inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIDAS = None
MIDAS_TRANSFORM = None

if os.path.exists(SIZES_DB_PATH):
    with open(SIZES_DB_PATH, "r", encoding="utf-8") as f:
        SIZES_DB = json.load(f)
    logger.info("Loaded sizes DB from %s", SIZES_DB_PATH)
else:
    logger.warning("sizes_db.json not found; brand matching will use generic rules.")
    SIZES_DB = {}

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def decode_base64_image(data_uri: str) -> Optional[np.ndarray]:
    """Decode a base64 data URI (or raw base64) into a BGR numpy array."""
    if not data_uri:
        return None
    try:
        _, encoded = data_uri.split(",", 1)
    except ValueError:
        encoded = data_uri
    try:
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        logger.exception("Failed to decode incoming image")
        return None

    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR

    h, w = arr.shape[:2]
    if max(h, w) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(h, w)
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)))

    return arr


def _pre_trust_torch_hub_repos(owners_and_names):
    """
    torch.hub's `trust_repo` kwarg only applies to the *direct* repo
    being loaded - it does not propagate to repos that the loaded
    hubconf.py itself calls torch.hub.load() on internally. MiDaS_small
    is a real-world example of this: its hubconf.py fetches an
    EfficientNet backbone from a second repo
    (rwightman/gen-efficientnet-pytorch), and that nested call has no
    trust_repo argument at all, so on a fresh cache it hits an
    interactive y/N prompt. In a non-interactive server process that
    raises an unhandled EOFError and the model never loads.

    torch.hub actually exposes a more durable mechanism for this: a
    `trusted_list` file under its hub cache directory. Pre-populating
    it with the owner_repo names we know we need avoids the prompt
    entirely, for both the top-level and any nested repos a model's
    hubconf.py might reach for.
    """
    try:
        hub_dir = torch.hub.get_dir()
        os.makedirs(hub_dir, exist_ok=True)
        trusted_list_path = os.path.join(hub_dir, "trusted_list")
        existing = set()
        if os.path.exists(trusted_list_path):
            with open(trusted_list_path) as f:
                existing = {line.strip() for line in f if line.strip()}
        to_add = set(owners_and_names) - existing
        if to_add:
            with open(trusted_list_path, "a") as f:
                for entry in to_add:
                    f.write(entry + "\n")
    except Exception:
        logger.warning("Could not pre-populate torch.hub trusted_list; MiDaS load may prompt or fail.")


def _vendored_midas_path() -> Optional[str]:
    """
    Returns the path to a local MiDaS clone if setup_midas.py has been
    run, else None.

    Why this exists: torch.hub.load("intel-isl/MiDaS", ...) makes a
    GitHub API call (_validate_not_a_forked_repo) purely to check the
    repo isn't a malicious fork, *before* it even looks at the trusted
    list. On networks that block or rate-limit the GitHub API (common
    on corporate networks, CI runners, and some sandboxed
    environments), that call returns a non-200 response, and a bug in
    torch's own error handling (it tries to delete a header that was
    never set) turns that into an opaque KeyError instead of a clear
    network error - see https://github.com/pytorch/pytorch/issues for
    reports of the same failure with other torch.hub models.

    Loading from a local clone via source="local" skips that GitHub
    API call entirely, since there's no remote repo to validate.
    `setup_midas.py` sets this up by cloning both MiDaS and its nested
    EfficientNet dependency directly via git (a plain `git clone`,
    which only needs to reach github.com over HTTPS - notably *not*
    the GitHub REST API - so it succeeds in environments where
    torch.hub's own API call fails).
    """
    vendor_dir = os.path.join(os.path.dirname(__file__), "vendor", "MiDaS")
    hubconf_path = os.path.join(vendor_dir, "hubconf.py")
    # Check for hubconf.py specifically, not just the directory - an
    # interrupted or partially-failed `git clone` in setup_midas.py
    # can leave an empty/incomplete vendor/MiDaS folder behind. Trusting
    # isdir() alone there previously caused torch.hub.load() to fail
    # with a confusing FileNotFoundError for hubconf.py deep inside
    # torch's own loader, rather than this function correctly reporting
    # "no usable vendored clone" up front.
    return vendor_dir if os.path.isfile(hubconf_path) else None


def load_midas() -> None:
    """Load MiDaS small model for depth estimation (idempotent)."""
    global MIDAS, MIDAS_TRANSFORM
    if MIDAS is not None:
        return

    _pre_trust_torch_hub_repos(["intel-isl_MiDaS", "rwightman_gen-efficientnet-pytorch"])

    vendored = _vendored_midas_path()
    try:
        if vendored:
            logger.info("Loading MiDaS from local vendor clone at %s ...", vendored)
            MIDAS = torch.hub.load(vendored, "MiDaS_small", source="local", trust_repo=True)
            transforms = torch.hub.load(vendored, "transforms", source="local", trust_repo=True)
        else:
            logger.info(
                "No vendored MiDaS clone found - loading via torch.hub from GitHub directly. "
                "If this hangs or fails with a network/KeyError, run "
                "`python setup_midas.py` once to vendor it locally and avoid the GitHub "
                "API dependency entirely (see that script for details)."
            )
            MIDAS = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

        MIDAS.to(DEVICE)
        MIDAS.eval()
        MIDAS_TRANSFORM = transforms.small_transform
        logger.info("MiDaS loaded on device: %s", DEVICE)
    except Exception:
        logger.exception(
            "Failed to load MiDaS - depth correction will be skipped and the app will run "
            "in width-only estimation mode. Run `python setup_midas.py` for a more reliable "
            "load path that avoids torch.hub's GitHub API dependency."
        )
        MIDAS = None
        MIDAS_TRANSFORM = None


def estimate_depth(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Estimate a normalized (0..1) depth map for the given BGR image.
    Returns None if MiDaS isn't available, so callers must handle that
    gracefully rather than assuming depth is always present.
    """
    if MIDAS is None or MIDAS_TRANSFORM is None:
        return None
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = MIDAS_TRANSFORM(img_rgb).to(DEVICE)
        with torch.no_grad():
            prediction = MIDAS(input_tensor.unsqueeze(0))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy()
        dmin, dmax = float(np.nanmin(depth)), float(np.nanmax(depth))
        denom = (dmax - dmin) if (dmax - dmin) > 1e-6 else 1.0
        return (depth - dmin) / denom
    except Exception:
        logger.exception("Depth estimation failed; continuing without depth correction.")
        return None


def thickness_factor_from_depth(
    depth_map: Optional[np.ndarray],
    landmarks: List[Dict[str, Any]],
    img_shape,
) -> Optional[float]:
    """
    Derive a rough relative "torso thickness" correction factor from the
    depth map by comparing depth variance across the torso region to a
    expected baseline. This is intentionally a light-touch signal (see
    anthropometry/model.py's clamp on this factor) - monocular depth
    estimates are relative, not metric, so we use them only to nudge,
    never to replace, the width-based regression.
    """
    if depth_map is None or not landmarks or len(landmarks) < 25:
        return None
    try:
        h, w = img_shape[:2]
        xs = [int(landmarks[i]["x"] * w) for i in (11, 12, 23, 24) if i < len(landmarks)]
        ys = [int(landmarks[i]["y"] * h) for i in (11, 12, 23, 24) if i < len(landmarks)]
        if not xs or not ys:
            return None
        x0, x1 = max(0, min(xs)), min(w - 1, max(xs))
        y0, y1 = max(0, min(ys)), min(h - 1, max(ys))
        if x1 <= x0 or y1 <= y0:
            return None
        region = depth_map[y0:y1, x0:x1]
        if region.size == 0:
            return None
        # Higher depth variance in the torso region suggests the body
        # has more pronounced front-to-back structure relative to a
        # flat/distant baseline; this is a heuristic nudge, not a
        # calibrated metric measurement.
        variance = float(np.var(region))
        baseline_variance = 0.015  # empirically reasonable midpoint for a centered torso
        factor = 1.0 + (variance - baseline_variance) * 4.0
        return factor
    except Exception:
        logger.exception("Failed to derive thickness factor from depth map.")
        return None


def landmarks_from_payload(raw: List[Dict[str, Any]]) -> List[Landmark]:
    return [
        Landmark(
            x=float(p.get("x", 0.5)),
            y=float(p.get("y", 0.5)),
            visibility=float(p.get("visibility", 1.0)),
        )
        for p in raw
    ]


def validate_request(data: Dict[str, Any]) -> Optional[str]:
    """Return an error message if the payload is invalid, else None."""
    landmarks = data.get("landmarks")
    if not landmarks or not isinstance(landmarks, list) or len(landmarks) < 29:
        return "landmarks must be an array of at least 29 pose points"

    height = data.get("height")
    if height is not None:
        try:
            height_f = float(height)
        except (TypeError, ValueError):
            return "height must be numeric"
        if not (0 < height_f < 300):
            return "height out of plausible range"

    gender = (data.get("gender") or "").lower()
    if gender and gender not in ("male", "female"):
        return "gender must be 'male' or 'female'"

    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "device": str(DEVICE), "midas_loaded": bool(MIDAS is not None)})


@app.route("/measure", methods=["POST"])
def measure_endpoint():
    """
    Expected payload:
    {
      "image": "data:image/jpeg;base64,....",   // optional - depth correction skipped if absent
      "landmarks": [ {x, y, visibility}, ... ],   // required, >= 29 points
      "height": 170,                              // optional, cm
      "gender": "male" | "female",                // required for accurate sizing
      "item": "jeans",
      "brand": "Levi's"
    }
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"ok": False, "error": "invalid JSON body"}), 400

    validation_error = validate_request(data)
    if validation_error:
        return jsonify({"ok": False, "error": validation_error}), 400

    raw_landmarks = data["landmarks"]
    image_b64 = data.get("image")
    user_height = data.get("height")
    gender = (data.get("gender") or "male").lower()
    item = data.get("item") or "shirt"
    brand = data.get("brand") or ""

    landmarks = landmarks_from_payload(raw_landmarks)

    depth_thickness = None
    image_width, image_height = 720, 1280

    if image_b64:
        img = decode_base64_image(image_b64)
        if img is not None:
            image_height, image_width = img.shape[:2]
            load_midas()
            depth_map = estimate_depth(img)
            depth_thickness = thickness_factor_from_depth(depth_map, raw_landmarks, img.shape)

    result = estimate_measures(
        landmarks=landmarks,
        image_width=image_width,
        image_height=image_height,
        gender=gender,
        user_height_cm=float(user_height) if user_height is not None else None,
        depth_thickness_factor=depth_thickness,
    )

    if result is None:
        return jsonify({
            "ok": False,
            "error": "Could not compute measures - required landmarks missing or low-confidence. "
                     "Ensure full body (head to feet) is visible and well lit.",
        }), 422

    size = match_size(SIZES_DB, result.to_dict(), gender, item, brand)

    response = {
        "ok": True,
        "measures": result.to_dict(),
        "size": size,
        "confidence": result.confidence,
    }
    logger.info("measure result: size=%s confidence=%.1f", size, result.confidence)
    return jsonify(response)


if __name__ == "__main__":
    try:
        load_midas()
    except Exception:
        logger.warning("MiDaS warmup failed; depth correction will be unavailable until it loads.")
    app.run(host="0.0.0.0", port=PORT, debug=False)
