"""Unit and contract tests for M2 (Calibration) and M4 (Footprint & Minimap).

Acceptance criteria:
- Calibrator backend implements Calibrator protocol.
- FrameCalib contract matches Section 6 of docs/PLAN.md.
- Keypoint-based homography recovers known pitch projection.
- 2D pitch minimap renders correct dimensions and handles footprint overlays.
- Control video and JSON metadata are generated properly.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import pytest

from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.core.types import FrameCalib
from pitchlab.m2_calib.base import Calibrator, get_calibrator
from pitchlab.m2_calib.backend_roboflow import RoboflowCalibrator
from pitchlab.m4_footprint.pitch_minimap import MinimapRenderer
from pitchlab.m4_footprint.video_overlay import render_control_video


def test_calibrator_protocol_conformance():
    calib = get_calibrator("roboflow")
    assert isinstance(calib, Calibrator)
    assert calib.name == "roboflow"


def test_calibrator_no_pitch():
    pitch = PitchTemplate(105.0, 68.0)
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    calib = RoboflowCalibrator()
    result = calib.calibrate(blank_frame, pitch, frame_idx=0, t=0.0)

    assert isinstance(result, FrameCalib)
    assert result.status == "no_pitch"
    assert result.H_img2pitch is None
    assert result.footprint_m is None
    assert result.conf == 0.0


def test_calibrator_with_synthetic_keypoints():
    pitch = PitchTemplate(105.0, 68.0)
    img_w, img_h = 1280, 720
    frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Construct a known ground-truth affine homography for testing:
    # x_m = (u - 640) / 10.0, y_m = (360 - v) / 10.0
    # H_img2pitch:
    # [ 0.1   0.0  -64.0 ]
    # [ 0.0  -0.1   36.0 ]
    # [ 0.0   0.0    1.0 ]
    H_gt = np.array([
        [0.1, 0.0, -64.0],
        [0.0, -0.1, 36.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    H_inv = np.linalg.inv(H_gt)

    # Generate synthetic detections for 8 keypoints using H_inv
    pitch_kps = pitch.get_keypoints()
    selected_kps = [
        "corner_bottom_left", "corner_bottom_right",
        "halfway_bottom", "halfway_top", "center_mark",
        "penalty_spot_left", "penalty_spot_right",
        "center_circle_left", "center_circle_right"
    ]

    mock_detections = {}
    for name in selected_kps:
        x, y = pitch_kps[name]
        pt_homo = H_inv @ np.array([x, y, 1.0])
        u, v = pt_homo[0] / pt_homo[2], pt_homo[1] / pt_homo[2]
        mock_detections[name] = (float(u), float(v))

    calib = RoboflowCalibrator(keypoint_detector=lambda f: mock_detections)
    res = calib.calibrate(frame, pitch, frame_idx=10, t=0.4)

    assert res.status == "ok"
    assert res.H_img2pitch is not None
    assert res.conf > 0.9
    assert res.footprint_m is not None
    assert len(res.footprint_m) >= 3

    # Recovered homography should closely match ground truth
    # Normalize scale by H[2, 2]
    H_recovered = res.H_img2pitch / res.H_img2pitch[2, 2]
    np.testing.assert_allclose(H_recovered, H_gt, atol=1e-3)


def test_minimap_renderer():
    pitch = PitchTemplate(105.0, 68.0)
    renderer = MinimapRenderer(pitch, width_px=400, height_px=260)

    # Base pitch image check
    assert renderer.base_pitch_img.shape == (260, 400, 3)

    # Footprint overlay check
    dummy_footprint = np.array([
        [-20.0, -10.0],
        [20.0, -10.0],
        [15.0, 15.0],
        [-15.0, 15.0],
    ])
    minimap = renderer.draw_minimap(dummy_footprint, status="ok")
    assert minimap.shape == (260, 400, 3)
    assert not np.array_equal(minimap, renderer.base_pitch_img)


def test_control_video_and_json_export(tmp_path: Path):
    pitch = PitchTemplate(105.0, 68.0)
    w, h, fps = 640, 360, 10.0
    num_frames = 15

    # 1. Create a dummy test video
    test_video_path = tmp_path / "test_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(test_video_path), fourcc, fps, (w, h))
    for i in range(num_frames):
        img = np.full((h, w, 3), (35, 120, 45), dtype=np.uint8)
        cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(img)
    writer.release()

    # 2. Mock calib results
    H_dummy = np.array([
        [0.1, 0.0, -32.0],
        [0.0, -0.1, 18.0],
        [0.0, 0.0, 1.0],
    ])
    footprint = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])

    calib_results = [
        FrameCalib(
            frame_idx=i,
            t=i / fps,
            status="ok" if i >= 3 else "no_pitch",
            H_img2pitch=H_dummy if i >= 3 else None,
            conf=0.95 if i >= 3 else 0.0,
            footprint_m=footprint if i >= 3 else None,
            source="keypoints",
        )
        for i in range(num_frames)
    ]

    out_video = tmp_path / "control_preview.mp4"
    out_json = tmp_path / "calib_summary.json"

    # 3. Render control video
    render_control_video(
        video_path=test_video_path,
        calib_results=calib_results,
        output_video_path=out_video,
        output_json_path=out_json,
        pitch=pitch,
    )

    assert out_video.exists()
    assert out_video.stat().st_size > 1000

    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["schema_version"] == "1.0.0"
    assert len(meta["frames"]) == num_frames
    assert meta["frames"][5]["status"] == "ok"
