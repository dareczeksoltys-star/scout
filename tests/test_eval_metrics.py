"""Unit tests for M5 evaluation metrics and benchmarking."""
import numpy as np
import pytest

from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.m2_calib.backend_roboflow import RoboflowCalibrator
from pitchlab.m5_eval.benchmark_gen import generate_benchmark_dataset
from pitchlab.m5_eval.dataset import EvalDataset, EvalSample
from pitchlab.m5_eval.metrics import (
    compute_mask_iou,
    compute_pitch_image_mask,
    compute_reprojection_error,
    evaluate_calibration,
)


def test_pitch_mask_generation():
    pitch = PitchTemplate(105.0, 68.0)
    w, h = 1280, 720

    # Test None
    mask_none = compute_pitch_image_mask(None, w, h, pitch)
    assert mask_none.shape == (h, w)
    assert np.all(mask_none == 0)

    # Valid H
    H = np.array([
        [0.1, 0.0, -64.0],
        [0.0, -0.1, 36.0],
        [0.0, 0.0, 1.0],
    ])
    mask = compute_pitch_image_mask(H, w, h, pitch)
    assert mask.shape == (h, w)
    assert np.sum(mask > 0) > 1000


def test_compute_mask_iou_identical():
    pitch = PitchTemplate(105.0, 68.0)
    w, h = 1280, 720
    H = np.array([
        [0.1, 0.0, -64.0],
        [0.0, -0.1, 36.0],
        [0.0, 0.0, 1.0],
    ])

    iou = compute_mask_iou(H, H, w, h, pitch)
    assert iou == pytest.approx(1.0, abs=1e-4)

    iou_none = compute_mask_iou(H, None, w, h, pitch)
    assert iou_none == 0.0


def test_compute_reprojection_error_exact():
    pitch = PitchTemplate(105.0, 68.0)
    H = np.array([
        [0.1, 0.0, -64.0],
        [0.0, -0.1, 36.0],
        [0.0, 0.0, 1.0],
    ])
    H_inv = np.linalg.inv(H)
    kps = pitch.get_keypoints()

    # Generate exact keypoint image positions
    keypoints_px = {}
    for name in ["center_mark", "halfway_bottom", "penalty_spot_left"]:
        x, y = kps[name]
        u = H_inv[0, 0] * x + H_inv[0, 2]
        v = H_inv[1, 1] * y + H_inv[1, 2]
        keypoints_px[name] = (float(u), float(v))

    mean_err, med_err = compute_reprojection_error(H, keypoints_px, pitch)
    assert mean_err == pytest.approx(0.0, abs=1e-3)
    assert med_err == pytest.approx(0.0, abs=1e-3)


def test_benchmark_dataset_generation():
    pitch = PitchTemplate(105.0, 68.0)
    ds = generate_benchmark_dataset(pitch=pitch)

    assert len(ds) == 36
    main_samples = ds.filter_by_view("main")
    assert len(main_samples) == 32
    closeup_samples = ds.filter_by_view("closeup")
    assert len(closeup_samples) == 2
    graphic_samples = ds.filter_by_view("crowd_graphic")
    assert len(graphic_samples) == 2


def test_evaluate_calibration_with_oracle():
    pitch = PitchTemplate(105.0, 68.0)
    ds = generate_benchmark_dataset(pitch=pitch)

    # Oracle detector returns visible keypoints directly from sample
    current_sample = [None]

    def mock_detector(frame):
        if current_sample[0] is not None:
            return current_sample[0].keypoints_px
        return {}

    calib = RoboflowCalibrator(keypoint_detector=mock_detector)

    # We evaluate directly using evaluate_calibration, injecting sample
    # To test pipeline, let's evaluate on 4 samples
    mini_ds = EvalDataset(ds.samples[:4])
    for s in mini_ds:
        current_sample[0] = s

    metrics = evaluate_calibration(calib, mini_ds, pitch)
    assert metrics.total_samples == 4
    assert metrics.backend_name == "roboflow"
