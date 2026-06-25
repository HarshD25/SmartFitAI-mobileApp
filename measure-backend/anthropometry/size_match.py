"""
anthropometry/size_match.py

Brand size-chart matching, factored out as its own module so it is not
duplicated (and silently drifting out of sync) between the Node.js and
Python backends. The Python backend is the source of truth for size
matching; server/server.js delegates to this same sizes_db.json file
and intentionally mirrors this logic for its fallback-mode use case
(see server/server.js comments).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def match_size(
    sizes_db: Dict[str, Any],
    measures: Dict[str, float],
    gender: str = "male",
    item: str = "shirt",
    brand: str = "",
) -> Dict[str, Any]:
    """
    Match measurements against a brand-specific size table when one is
    available in sizes_db, otherwise fall back to a generic chest/waist
    based banding.

    Returns a dict: {alpha, numeric, fit, score, source, range}

    `range` is additive (kept for callers that only care about the
    matched size) and describes where the user's measurement sits
    within the full set of sizes considered, for UIs that want to show
    a "you're here on the size spectrum" visualization rather than
    just the single best match:
        {
          "key": "chest" | "waist",
          "userValue": <measured cm>,
          "min": <smallest size's value>,
          "max": <largest size's value>,
          "sizes": [{"alpha", "numeric", "value"}, ...]  # in table order
        }
    """
    gender_key = (gender or "male").strip().lower()
    item_key = (item or "shirt").strip().lower()

    try:
        gender_table = sizes_db.get(gender_key, {})
        item_table = (
            gender_table.get("topwear", {}).get(item_key)
            or gender_table.get("bottomwear", {}).get(item_key)
        )

        if item_table:
            is_waist_item = item_key in ("jeans", "pant", "pants", "shorts", "skirt", "chinos", "cargopants", "tights")
            key = "waist" if is_waist_item else "chest"
            target_value = measures.get("waistCm") if is_waist_item else measures.get("chestCm")

            best = None
            best_score = float("inf")
            for row in item_table:
                target = row.get(key, row.get("chest", row.get("waist", row.get("hip"))))
                if target is None or target_value is None:
                    continue
                score = abs(target - target_value)
                if score < best_score:
                    best_score = score
                    best = row

            if best:
                return {
                    "alpha": best.get("alpha"),
                    "numeric": best.get("numeric"),
                    "fit": "Brand match",
                    "score": round(best_score, 1),
                    "source": "brand_table",
                    "range": _build_range(item_table, key, target_value),
                }
    except Exception:
        # Fall through to generic mapping below; a malformed/missing
        # brand table should never crash the request.
        pass

    return _generic_fallback(measures, gender_key)


def _build_range(item_table, key: str, user_value: Optional[float]) -> Optional[Dict[str, Any]]:
    sizes = []
    for row in item_table:
        value = row.get(key, row.get("chest", row.get("waist", row.get("hip"))))
        if value is None:
            continue
        sizes.append({"alpha": row.get("alpha"), "numeric": row.get("numeric"), "value": value})

    if not sizes:
        return None

    values = [s["value"] for s in sizes]
    return {
        "key": key,
        "userValue": user_value,
        "min": min(values),
        "max": max(values),
        "sizes": sizes,
    }


def _generic_fallback(measures: Dict[str, float], gender_key: str) -> Dict[str, Any]:
    c = measures.get("chestCm", 96) or 96
    if gender_key == "female":
        bands = [
            (82, "XS", 32), (90, "S", 34), (98, "M", 36), (106, "L", 38),
        ]
    else:
        bands = [
            (88, "XS", 34), (96, "S", 36), (104, "M", 38), (112, "L", 40),
        ]
    # The generic bands above are upper-bound cutoffs, not a fitted
    # midpoint per size - approximate a representative chest value per
    # alpha size (4cm below its cutoff) purely for the range
    # visualization, so it lines up with how the match itself is
    # scored a few lines below.
    band_sizes = [
        {"alpha": alpha, "numeric": numeric, "value": limit - 4} for limit, alpha, numeric in bands
    ]
    xl_alpha, xl_numeric = ("XL", 42) if gender_key != "female" else ("XL", 40)
    band_sizes.append({"alpha": xl_alpha, "numeric": xl_numeric, "value": 116})
    generic_range = {
        "key": "chest",
        "userValue": c,
        "min": band_sizes[0]["value"],
        "max": band_sizes[-1]["value"],
        "sizes": band_sizes,
    }

    for limit, alpha, numeric in bands:
        if c < limit:
            return {
                "alpha": alpha,
                "numeric": numeric,
                "fit": _fit_for(alpha),
                "score": round(abs(c - (limit - 4)), 1),
                "source": "generic_fallback",
                "range": generic_range,
            }
    alpha, numeric = ("XL", 42) if gender_key != "female" else ("XL", 40)
    return {
        "alpha": alpha,
        "numeric": numeric,
        "fit": "Loose",
        "score": round(abs(c - 116), 1),
        "source": "generic_fallback",
        "range": generic_range,
    }


def _fit_for(alpha: str) -> str:
    return {
        "XS": "Slim",
        "S": "Slim",
        "M": "Regular",
        "L": "Relaxed",
        "XL": "Loose",
    }.get(alpha, "Regular")
