import numpy as np
import pytest
from pitchlab.core.pitch_model import PitchModel, PITCH_LENGTH, PITCH_WIDTH


def test_pitch_model_dimensions():
    model = PitchModel()
    assert model.length == 105.0
    assert model.width == 68.0


def test_pitch_model_keypoints():
    model = PitchModel()
    kp = model.get_keypoints()
    assert "TOP_LEFT_CORNER" in kp
    assert kp["TOP_LEFT_CORNER"] == (0.0, 0.0)
    assert kp["BOTTOM_RIGHT_CORNER"] == (105.0, 68.0)
    assert kp["CENTER_MARK"] == (52.5, 34.0)
    assert kp["L_PENALTY_MARK"] == (11.0, 34.0)
    assert kp["R_PENALTY_MARK"] == (94.0, 34.0)


def test_pitch_model_symmetry():
    model = PitchModel()
    kp = model.get_keypoints()
    # Left and right goal boxes should mirror horizontally
    l_box_depth = kp["L_GOAL_BOX_TR"][0]
    r_box_depth = 105.0 - kp["R_GOAL_BOX_TL"][0]
    assert pytest.approx(l_box_depth, 1e-4) == r_box_depth

    # Left and right penalty areas should mirror horizontally
    l_pen_depth = kp["L_PENALTY_BOX_TR"][0]
    r_pen_depth = 105.0 - kp["R_PENALTY_BOX_TL"][0]
    assert pytest.approx(l_pen_depth, 1e-4) == r_pen_depth


def test_pitch_model_bounds():
    model = PitchModel()
    assert model.is_inside_pitch(52.5, 34.0)
    assert model.is_inside_pitch(0.0, 0.0)
    assert model.is_inside_pitch(105.0, 68.0)
    assert not model.is_inside_pitch(-1.0, 34.0, margin=0.0)
    assert model.is_inside_pitch(-1.0, 34.0, margin=2.0)
    assert not model.is_inside_pitch(108.0, 34.0, margin=2.0)


def test_pitch_model_sampling_and_projection():
    model = PitchModel()
    pts = model.sample_line_points(density=1.0)
    assert pts.ndim == 2
    assert pts.shape[0] > 500
    assert pts.shape[1] == 2

    # Identity homography projection
    H_eye = np.eye(3)
    projected = model.project_ground_points(pts[:10], H_eye)
    assert np.allclose(pts[:10], projected)
