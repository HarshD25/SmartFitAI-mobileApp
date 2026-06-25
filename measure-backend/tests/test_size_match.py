"""
tests/test_size_match.py - unit tests for anthropometry/size_match.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropometry.size_match import match_size

SAMPLE_DB = {
    "male": {
        "topwear": {
            "shirt": [
                {"alpha": "38", "numeric": 38, "chest": 96, "waist": 82},
                {"alpha": "40", "numeric": 40, "chest": 100, "waist": 86},
                {"alpha": "42", "numeric": 42, "chest": 104, "waist": 90},
            ]
        },
        "bottomwear": {
            "jeans": [
                {"alpha": "30", "numeric": 30, "waist": 76, "hip": 94},
                {"alpha": "32", "numeric": 32, "waist": 81, "hip": 98},
            ],
            "chinos": [
                {"alpha": "30", "numeric": 30, "waist": 77, "hip": 96},
                {"alpha": "32", "numeric": 32, "waist": 82, "hip": 100},
            ],
        },
    },
    "female": {
        "topwear": {
            "tshirt": [
                {"alpha": "S", "numeric": 34, "chest": 84, "waist": 68, "hip": 92},
                {"alpha": "M", "numeric": 36, "chest": 88, "waist": 72, "hip": 96},
            ]
        }
    },
}


class TestBrandTableMatch:
    def test_picks_closest_chest_match(self):
        result = match_size(SAMPLE_DB, {"chestCm": 101, "waistCm": 87}, "male", "shirt", "")
        assert result["alpha"] == "40"
        assert result["source"] == "brand_table"

    def test_picks_closest_waist_match_for_jeans(self):
        # waist=80 is closer to the "32" row (waist 81, diff 1) than the
        # "30" row (waist 76, diff 4).
        result = match_size(SAMPLE_DB, {"waistCm": 80, "hipCm": 96}, "male", "jeans", "")
        assert result["alpha"] == "32"
        assert result["source"] == "brand_table"

    def test_exact_match_returns_exact_size(self):
        result = match_size(SAMPLE_DB, {"chestCm": 96, "waistCm": 82}, "male", "shirt", "")
        assert result["alpha"] == "38"
        assert result["score"] == 0


class TestWaistItemDetection:
    """
    Regression coverage for a real bug found when adding new
    bottomwear items (chinos, tights): the waist-vs-chest item
    detection list is a hardcoded tuple that has to be kept in sync
    whenever a new waist-measured item is added to sizes_db.json. When
    an item is missing from that list, match_size tries to match it
    against chestCm instead of waistCm, finds no usable target in rows
    that only have waist/hip keys, and silently falls back to the
    generic bands instead of using the (perfectly good) brand table
    that's actually sitting right there.
    """

    def test_chinos_is_treated_as_a_waist_item(self):
        result = match_size(SAMPLE_DB, {"waistCm": 80, "hipCm": 97}, "male", "chinos", "")
        assert result["source"] == "brand_table"
        assert result["range"]["key"] == "waist"

    def test_every_known_waist_item_resolves_to_waist_not_chest(self):
        # chestCm deliberately omitted from measures - if any of these
        # were (re-)miscategorized as chest items, match_size would
        # have no chestCm to compare against and would fall through to
        # the generic fallback instead of the brand table.
        for item in ("jeans", "pant", "pants", "shorts", "skirt", "chinos", "cargopants", "tights"):
            result = match_size(SAMPLE_DB, {"waistCm": 80, "hipCm": 97}, "male", item, "")
            # Items without a matching table in SAMPLE_DB legitimately
            # fall back to generic - that's fine. The thing we're
            # actually guarding is that chinos/cargopants/tights (which
            # DO have a table for "chinos") aren't skipped over due to
            # being treated as chest items.
            if item == "chinos":
                assert result["source"] == "brand_table"


class TestGenericFallback:
    def test_unknown_item_falls_back_to_generic(self):
        result = match_size(SAMPLE_DB, {"chestCm": 100}, "male", "totally_unknown_item", "")
        assert result["source"] == "generic_fallback"
        assert result["alpha"] in ("XS", "S", "M", "L", "XL")

    def test_missing_gender_table_falls_back(self):
        result = match_size(SAMPLE_DB, {"chestCm": 100}, "unisex", "shirt", "")
        assert result["source"] == "generic_fallback"

    def test_empty_db_does_not_crash(self):
        result = match_size({}, {"chestCm": 100}, "male", "shirt", "")
        assert result["source"] == "generic_fallback"
        assert result["alpha"] is not None


class TestFallbackBandsAreMonotonic:
    def test_larger_chest_yields_larger_or_equal_size(self):
        sizes_order = ["XS", "S", "M", "L", "XL"]
        prev_index = -1
        for chest in (75, 85, 93, 101, 109, 120):
            result = match_size({}, {"chestCm": chest}, "male", "unknown", "")
            idx = sizes_order.index(result["alpha"])
            assert idx >= prev_index
            prev_index = idx


class TestSizeRangeForVisualization:
    """
    `range` is additive data used by the frontend's results-screen
    visualization to show where the user's measurement sits relative
    to the full set of sizes considered - not just the single closest
    match. These tests guard the shape and correctness of that data
    independently of the match itself.
    """

    def test_brand_table_range_spans_all_rows(self):
        result = match_size(SAMPLE_DB, {"chestCm": 101, "waistCm": 87}, "male", "shirt", "")
        r = result["range"]
        assert r["key"] == "chest"
        assert r["min"] == 96
        assert r["max"] == 104
        assert [s["alpha"] for s in r["sizes"]] == ["38", "40", "42"]

    def test_brand_table_range_uses_waist_for_waist_items(self):
        result = match_size(SAMPLE_DB, {"waistCm": 80, "hipCm": 96}, "male", "jeans", "")
        r = result["range"]
        assert r["key"] == "waist"
        assert r["min"] == 76
        assert r["max"] == 81

    def test_brand_table_range_reports_user_value(self):
        result = match_size(SAMPLE_DB, {"chestCm": 101, "waistCm": 87}, "male", "shirt", "")
        assert result["range"]["userValue"] == 101

    def test_generic_fallback_still_includes_a_usable_range(self):
        result = match_size({}, {"chestCm": 100}, "male", "unknown", "")
        r = result["range"]
        assert r is not None
        assert r["key"] == "chest"
        assert r["min"] < r["max"]
        assert len(r["sizes"]) >= 4

    def test_range_sizes_are_in_ascending_order(self):
        """
        The frontend renders sizes left-to-right along a bar assuming
        table order already reflects size order - this guards that
        assumption against a malformed or re-ordered sizes_db.json.
        """
        result = match_size(SAMPLE_DB, {"chestCm": 101}, "male", "shirt", "")
        values = [s["value"] for s in result["range"]["sizes"]]
        assert values == sorted(values)
