"""
tests/conftest.py - shared fixtures for synthetic pose landmarks.

We build a synthetic, geometrically plausible "standing person" pose
so the anthropometric model can be tested deterministically without
needing real camera captures or a MediaPipe runtime in CI.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from anthropometry import Landmark


def make_standing_pose(
    image_width=720,
    image_height=1280,
    shoulder_width_norm=0.38,
    hip_width_norm=0.30,
    head_y_norm=0.08,
    hip_y_norm=0.52,
    ankle_y_norm=0.95,
    center_x=0.5,
    visibility=0.95,
):
    """
    Build a 33-point MediaPipe-pose-shaped landmark list representing a
    person standing facing the camera. Only the indices our model
    actually reads are positioned meaningfully; the rest are filled
    with plausible placeholder values so list-length checks pass.

    Default width norms are deliberately aspect-ratio-aware: pixel
    distances are computed as (x_norm * image_width) for horizontal
    spans and (y_norm * image_height) for vertical spans (see
    anthropometry/model.py:_pixel_dist). For a portrait 720x1280 frame,
    a realistic adult shoulder width (~24% of standing height) and hip
    width (~19% of standing height) translate to *larger* normalized
    x-fractions than you'd naively guess, precisely because width and
    height use different pixel scales. Getting this wrong silently
    produces undersized synthetic bodies that don't reflect real
    capture data - the original version of this fixture used
    shoulder_width_norm=0.22 / hip_width_norm=0.18, which looked
    plausible as raw fractions but corresponded to a shoulder width of
    only ~14% of height once the aspect ratio was accounted for.
    """
    lm = [Landmark(x=center_x, y=0.5, visibility=visibility) for _ in range(33)]

    lm[0] = Landmark(x=center_x, y=head_y_norm, visibility=visibility)  # nose
    lm[11] = Landmark(x=center_x - shoulder_width_norm / 2, y=head_y_norm + 0.07, visibility=visibility)  # L shoulder
    lm[12] = Landmark(x=center_x + shoulder_width_norm / 2, y=head_y_norm + 0.07, visibility=visibility)  # R shoulder
    lm[23] = Landmark(x=center_x - hip_width_norm / 2, y=hip_y_norm, visibility=visibility)  # L hip
    lm[24] = Landmark(x=center_x + hip_width_norm / 2, y=hip_y_norm, visibility=visibility)  # R hip
    lm[27] = Landmark(x=center_x - 0.04, y=ankle_y_norm, visibility=visibility)  # L ankle
    lm[28] = Landmark(x=center_x + 0.04, y=ankle_y_norm, visibility=visibility)  # R ankle

    return lm


@pytest.fixture
def standing_pose_factory():
    return make_standing_pose
