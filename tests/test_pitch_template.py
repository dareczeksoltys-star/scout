"""Unit tests for PitchTemplate.

Acceptance criteria for M0:
- All keypoints must be within pitch boundaries.
- Pitch geometry must be symmetric across both X and Y axes.
- Standard FIFA fixed dimensions must match regulations regardless of pitch length/width.
"""
import pytest
from pitchlab.core.pitch_template import PitchTemplate


def test_pitch_dimensions_and_bounds():
    pitch = PitchTemplate(length_m=105.0, width_m=68.0)
    assert pitch.length == 105.0
    assert pitch.width == 68.0
    assert pitch.x_min == -52.5
    assert pitch.x_max == 52.5
    assert pitch.y_min == -34.0
    assert pitch.y_max == 34.0

    assert pitch.is_inside(0.0, 0.0)
    assert pitch.is_inside(-52.5, -34.0)
    assert pitch.is_inside(52.5, 34.0)
    assert not pitch.is_inside(53.0, 0.0)
    assert not pitch.is_inside(0.0, -35.0)


def test_keypoints_inside_pitch():
    pitch = PitchTemplate(length_m=105.0, width_m=68.0)
    kps = pitch.get_keypoints()
    assert len(kps) > 20

    for name, (x, y) in kps.items():
        assert pitch.is_inside(x, y, margin_m=1e-5), f"Keypoint {name} at ({x}, {y}) outside pitch!"


def test_pitch_symmetry():
    pitch = PitchTemplate(length_m=105.0, width_m=68.0)
    kps = pitch.get_keypoints()

    # X-axis reflection (Left <-> Right)
    assert kps["penalty_spot_left"][0] == pytest.approx(-kps["penalty_spot_right"][0])
    assert kps["penalty_spot_left"][1] == pytest.approx(kps["penalty_spot_right"][1])

    assert kps["corner_top_left"][0] == pytest.approx(-kps["corner_top_right"][0])
    assert kps["corner_bottom_left"][0] == pytest.approx(-kps["corner_bottom_right"][0])

    assert kps["penalty_area_left_tr"][0] == pytest.approx(-kps["penalty_area_right_tl"][0])
    assert kps["goal_area_left_tr"][0] == pytest.approx(-kps["goal_area_right_tl"][0])

    # Y-axis reflection (Top <-> Bottom)
    assert kps["corner_top_left"][1] == pytest.approx(-kps["corner_bottom_left"][1])
    assert kps["corner_top_right"][1] == pytest.approx(-kps["corner_bottom_right"][1])
    assert kps["halfway_top"][1] == pytest.approx(-kps["halfway_bottom"][1])


def test_fixed_fifa_dimensions():
    # Test with standard 105x68
    p1 = PitchTemplate(length_m=105.0, width_m=68.0)
    # Test with non-standard 100x64
    p2 = PitchTemplate(length_m=100.0, width_m=64.0)

    for p in [p1, p2]:
        kps = p.get_keypoints()
        # Penalty spot distance from goal line is exactly 11.0m
        left_spot_dist = kps["penalty_spot_left"][0] - p.x_min
        right_spot_dist = p.x_max - kps["penalty_spot_right"][0]
        assert left_spot_dist == pytest.approx(11.0)
        assert right_spot_dist == pytest.approx(11.0)

        # Penalty box length is 16.5m
        box_len = kps["penalty_area_left_tr"][0] - p.x_min
        assert box_len == pytest.approx(16.5)

        # Penalty box width is 40.32m
        box_width = kps["penalty_area_left_tl"][1] - kps["penalty_area_left_bl"][1]
        assert box_width == pytest.approx(40.32)

        # Goal area length is 5.5m
        goal_box_len = kps["goal_area_left_tr"][0] - p.x_min
        assert goal_box_len == pytest.approx(5.5)

        # Goal area width is 18.32m
        goal_box_width = kps["goal_area_left_tl"][1] - kps["goal_area_left_bl"][1]
        assert goal_box_width == pytest.approx(18.32)

        # Center circle radius is 9.15m
        assert p.center_circle_radius == 9.15


def test_line_segments_and_arcs():
    pitch = PitchTemplate(length_m=105.0, width_m=68.0)
    lines = pitch.get_lines()
    assert len(lines) >= 11

    for line in lines:
        x1, y1 = line.p1
        x2, y2 = line.p2
        assert pitch.is_inside(x1, y1, margin_m=1e-5), f"Line {line.name} p1 outside"
        assert pitch.is_inside(x2, y2, margin_m=1e-5), f"Line {line.name} p2 outside"

    arcs = pitch.get_arcs()
    assert len(arcs) == 3
    center_circle = next(a for a in arcs if a.name == "center_circle")
    assert center_circle.radius == 9.15
    assert center_circle.center == (0.0, 0.0)
