"""Unit tests for geometry and homography operations."""
import numpy as np
import pytest
from shapely.geometry import Polygon

from pitchlab.core.geometry import compute_footprint, project_points, validate_homography
from pitchlab.core.pitch_template import PitchTemplate


def test_project_points_identity():
    H = np.eye(3, dtype=np.float64)
    pts = np.array([
        [0.0, 0.0],
        [10.0, 20.0],
        [-5.0, 15.0]
    ])
    proj, valid = project_points(H, pts)
    assert np.all(valid)
    np.testing.assert_allclose(proj, pts)


def test_project_points_behind_camera():
    # Homography where w is negative
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    pts = np.array([[10.0, 10.0]])
    proj, valid = project_points(H, pts)
    assert not valid[0]


def test_validate_homography_invalid_cases():
    pitch = PitchTemplate(105.0, 68.0)
    w, h = 1920, 1080

    # None
    valid, reason = validate_homography(None, pitch, w, h)
    assert not valid

    # Degenerate (zeros)
    H_zero = np.zeros((3, 3))
    valid, reason = validate_homography(H_zero, pitch, w, h)
    assert not valid

    # NaN
    H_nan = np.eye(3)
    H_nan[0, 0] = np.nan
    valid, reason = validate_homography(H_nan, pitch, w, h)
    assert not valid


def test_compute_footprint_clipped():
    pitch = PitchTemplate(105.0, 68.0)
    w, h = 1920, 1080

    # A simple homography that scales 1920x1080 to pitch coordinates roughly
    # (u, v) -> (x, y)
    # let's construct affine H: x = (u - 960)/20, y = (540 - v)/20
    H = np.array([
        [1.0 / 20.0, 0.0, -960.0 / 20.0],
        [0.0, -1.0 / 20.0, 540.0 / 20.0],
        [0.0, 0.0, 1.0],
    ])

    footprint = compute_footprint(H, w, h, pitch)
    assert footprint is not None
    assert len(footprint) >= 3

    # Every point of the footprint must be inside the pitch boundaries
    for pt in footprint:
        assert pitch.is_inside(pt[0], pt[1], margin_m=1e-4)
