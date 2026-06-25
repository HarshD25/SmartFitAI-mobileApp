"""
tests/test_model.py - unit tests for anthropometry/model.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropometry import estimate


class TestBasicEstimation:
    def test_returns_none_for_missing_landmarks(self):
        assert estimate([], 720, 1280, gender="male") is None
        assert estimate(None, 720, 1280, gender="male") is None

    def test_returns_none_for_too_few_landmarks(self, standing_pose_factory):
        pose = standing_pose_factory()[:10]
        assert estimate(pose, 720, 1280, gender="male") is None

    def test_returns_measurements_for_valid_pose(self, standing_pose_factory):
        pose = standing_pose_factory()
        result = estimate(pose, 720, 1280, gender="male", user_height_cm=175)
        assert result is not None
        assert 130 <= result.height_cm <= 220
        assert 60 <= result.chest_cm <= 150
        assert 50 <= result.waist_cm <= 150
        assert 60 <= result.hip_cm <= 150
        assert 0 <= result.confidence <= 100

    def test_low_visibility_landmarks_rejected(self, standing_pose_factory):
        pose = standing_pose_factory(visibility=0.1)
        result = estimate(pose, 720, 1280, gender="male")
        assert result is None


class TestHeightCalibration:
    def test_user_height_dominates_scale(self, standing_pose_factory):
        """
        Same pose geometry, two different user-supplied heights, should
        produce noticeably different cm outputs roughly proportional to
        the height ratio - this confirms user height is being used as
        the primary calibration signal rather than being washed out by
        the population-average fallback (the original bug: only a 25%
        weight was given to user-supplied height).
        """
        pose = standing_pose_factory()
        short = estimate(pose, 720, 1280, gender="male", user_height_cm=150)
        tall = estimate(pose, 720, 1280, gender="male", user_height_cm=200)

        assert short is not None and tall is not None
        assert tall.height_cm > short.height_cm
        assert tall.chest_cm > short.chest_cm
        # Roughly proportional: 200/150 ~= 1.33x; allow generous tolerance
        # since other (smaller-weighted) calibration signals are blended in.
        ratio = tall.height_cm / short.height_cm
        assert 1.15 <= ratio <= 1.45

    def test_implausible_height_ignored(self, standing_pose_factory):
        """A height outside plausible human range should not be used
        as the calibration source (it should fall back to population
        heuristics rather than producing a wildly scaled-up estimate)."""
        pose = standing_pose_factory()
        result_bad = estimate(pose, 720, 1280, gender="male", user_height_cm=999)
        result_none = estimate(pose, 720, 1280, gender="male", user_height_cm=None)
        assert result_bad is not None
        # Should land in the same ballpark as the no-height fallback,
        # not anywhere near 999cm-scaled values.
        assert abs(result_bad.height_cm - result_none.height_cm) < 40

    def test_implausible_height_matches_no_height_exactly(self, standing_pose_factory):
        """
        Regression test for a real bug found during development: an
        implausible height (e.g. 999) was correctly excluded from the
        *primary* calibration source, but an inconsistent truthiness
        check elsewhere in the calibration logic let it leak into the
        torso-fraction fallback's assumed height AND suppressed the
        shoulder-width fallback (which only activates when there is no
        valid height at all) - producing a result far closer to a
        valid-height calculation than to the true "no usable height"
        fallback. An implausible height must be treated identically to
        no height being supplied at all, everywhere in the function.
        """
        pose = standing_pose_factory()
        result_bad = estimate(pose, 720, 1280, gender="male", user_height_cm=999)
        result_negative = estimate(pose, 720, 1280, gender="male", user_height_cm=-5)
        result_none = estimate(pose, 720, 1280, gender="male", user_height_cm=None)

        assert result_bad.height_cm == result_none.height_cm
        assert result_bad.chest_cm == result_none.chest_cm
        assert result_negative.height_cm == result_none.height_cm

    def test_realistic_adult_proportions_yield_realistic_output(self, standing_pose_factory):
        """
        Sanity-check the model end-to-end against a normally-proportioned
        adult pose (not just bounds-checking): with a valid height
        supplied, chest/waist/hip should land within a realistic
        clothing-size range, not at the floor/ceiling of the bounds
        clamps. This guards against silent regressions where the model
        technically "returns a result" but the result is nonsense
        (which is exactly what the aspect-ratio mismatch bug above
        looked like before it was caught).
        """
        pose = standing_pose_factory()
        result = estimate(pose, 720, 1280, gender="male", user_height_cm=178)
        assert result is not None
        assert 165 <= result.height_cm <= 192
        assert 36 <= result.shoulder_cm <= 50
        assert 90 <= result.chest_cm <= 120
        assert 70 <= result.waist_cm <= 100
        assert 85 <= result.hip_cm <= 115


class TestGenderAwareness:
    def test_male_and_female_coefficients_differ(self, standing_pose_factory):
        """
        Identical pose geometry should yield different chest/waist/hip
        estimates depending on declared gender, because the regression
        coefficients differ - this is the fix for the original
        single-fixed-ratio-for-everyone bug.
        """
        pose = standing_pose_factory()
        male = estimate(pose, 720, 1280, gender="male", user_height_cm=170)
        female = estimate(pose, 720, 1280, gender="female", user_height_cm=170)

        assert male is not None and female is not None
        assert male.chest_cm != female.chest_cm or male.waist_cm != female.waist_cm

    def test_unknown_gender_defaults_safely(self, standing_pose_factory):
        pose = standing_pose_factory()
        result = estimate(pose, 720, 1280, gender="nonbinary-typo", user_height_cm=170)
        # Should not crash; falls back to default coefficients.
        assert result is not None


class TestDepthCorrection:
    def test_thickness_factor_changes_circumferences(self, standing_pose_factory):
        pose = standing_pose_factory()
        baseline = estimate(pose, 720, 1280, gender="male", user_height_cm=170)
        thicker = estimate(
            pose, 720, 1280, gender="male", user_height_cm=170, depth_thickness_factor=1.15
        )
        assert baseline is not None and thicker is not None
        assert thicker.chest_cm > baseline.chest_cm

    def test_thickness_factor_is_clamped(self, standing_pose_factory):
        """An extreme/noisy depth signal should be clamped, not applied
        verbatim, to avoid wild distortions from a single noisy frame."""
        pose = standing_pose_factory()
        baseline = estimate(pose, 720, 1280, gender="male", user_height_cm=170)
        extreme = estimate(
            pose, 720, 1280, gender="male", user_height_cm=170, depth_thickness_factor=5.0
        )
        assert baseline is not None and extreme is not None
        # Clamped to max 1.20x, so the ratio should never approach 5x.
        assert extreme.chest_cm / baseline.chest_cm <= 1.25


class TestConfidenceScoring:
    def test_confidence_higher_with_user_height(self, standing_pose_factory):
        pose = standing_pose_factory()
        with_height = estimate(pose, 720, 1280, gender="male", user_height_cm=170)
        without_height = estimate(pose, 720, 1280, gender="male", user_height_cm=None)
        assert with_height.confidence > without_height.confidence

    def test_confidence_lower_for_distant_subject(self, standing_pose_factory):
        """A subject far from the camera produces small shoulder/hip
        pixel spans; confidence should reflect that reduced precision."""
        close_pose = standing_pose_factory(shoulder_width_norm=0.38, hip_width_norm=0.30)
        far_pose = standing_pose_factory(shoulder_width_norm=0.04, hip_width_norm=0.03)

        close_result = estimate(close_pose, 720, 1280, gender="male", user_height_cm=170)
        far_result = estimate(far_pose, 720, 1280, gender="male", user_height_cm=170)

        assert close_result is not None and far_result is not None
        assert close_result.confidence > far_result.confidence
