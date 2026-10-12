from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pitchlab.core.config import (
    ByteTrackConfig,
    KalmanBallConfig,
    M6Config,
    SAHIConfig,
    YoloConfig,
    load_config,
)
from pitchlab.core.io import load_tracks_parquet, save_tracks_parquet
from pitchlab.core.types import EntityClass, Track2D
from pitchlab.m6_detect_track.ball_tracker import BallKalmanTracker
from pitchlab.m6_detect_track.detector import YOLOSAHIDetector
from pitchlab.m6_detect_track.interpolation import interpolate_ball_gaps
from pitchlab.m6_detect_track.pipeline import run_m6_pipeline
from pitchlab.m6_detect_track.tracker import ByteTrackWrapper


def test_config_loader(tmp_path: Path):
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text(
        """
m6:
  yolo:
    weights: "custom.pt"
    conf_thresh: 0.3
    iou_thresh: 0.5
    imgsz: 640
    device: "cpu"
  sahi:
    enabled: true
    slice_height: 320
    slice_width: 320
    overlap_height_ratio: 0.1
    overlap_width_ratio: 0.1
  bytetrack:
    track_thresh: 0.6
    track_buffer: 20
    match_thresh: 0.7
    frame_rate: 30
  kalman_ball:
    process_noise: 0.5
    measurement_noise: 5.0
    max_dist_px: 100.0
    max_missing_frames: 3
"""
    )
    loaded = load_config(cfg_file)
    assert loaded.m6.yolo.weights == "custom.pt"
    assert loaded.m6.yolo.conf_thresh == 0.3
    assert loaded.m6.sahi.enabled is True
    assert loaded.m6.bytetrack.frame_rate == 30
    assert loaded.m6.kalman_ball.max_missing_frames == 3


def test_tracks_parquet_io(tmp_path: Path):
    parquet_path = tmp_path / "tracks.parquet"
    tracks = [
        Track2D(
            track_id=1,
            frame_idx=0,
            bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
            conf=0.95,
            cls=EntityClass.PLAYER,
        ),
        Track2D(
            track_id=0,
            frame_idx=0,
            bbox_xyxy=(50.0, 50.0, 55.0, 55.0),
            conf=0.88,
            cls=EntityClass.BALL,
        ),
    ]

    save_tracks_parquet(tracks, parquet_path)
    assert parquet_path.exists()

    loaded = load_tracks_parquet(parquet_path)
    assert len(loaded) == 2
    assert loaded[0].track_id == 1
    assert loaded[0].cls == EntityClass.PLAYER
    assert loaded[0].bbox_xyxy == (10.0, 20.0, 30.0, 40.0)
    assert loaded[1].track_id == 0
    assert loaded[1].cls == EntityClass.BALL
    assert loaded[1].bbox_xyxy == (50.0, 50.0, 55.0, 55.0)


def test_ball_kalman_tracker_smooth_and_predict():
    cfg = KalmanBallConfig(
        process_noise=1.0,
        measurement_noise=5.0,
        max_dist_px=100.0,
        max_missing_frames=2,
    )
    tracker = BallKalmanTracker(cfg=cfg, ball_id=0)

    # Frame 0: first detection
    d0 = [
        Track2D(
            track_id=-1,
            frame_idx=0,
            bbox_xyxy=(100.0, 100.0, 110.0, 110.0),
            conf=0.9,
            cls=EntityClass.BALL,
        )
    ]
    t0 = tracker.update(d0, frame_idx=0)
    assert t0 is not None
    assert t0.track_id == 0
    assert t0.frame_idx == 0

    # Frame 1: moved slightly
    d1 = [
        Track2D(
            track_id=-1,
            frame_idx=1,
            bbox_xyxy=(105.0, 105.0, 115.0, 115.0),
            conf=0.9,
            cls=EntityClass.BALL,
        )
    ]
    t1 = tracker.update(d1, frame_idx=1)
    assert t1 is not None
    assert t1.frame_idx == 1

    # Frame 2: missing detection (should use prediction)
    t2 = tracker.update([], frame_idx=2)
    assert t2 is not None
    assert t2.frame_idx == 2

    # Frame 3: missing detection
    t3 = tracker.update([], frame_idx=3)
    assert t3 is not None

    # Frame 4: exceeds max_missing_frames -> returns None
    t4 = tracker.update([], frame_idx=4)
    assert t4 is None


def test_ball_gap_interpolation():
    tracks = [
        # Frame 0
        Track2D(
            track_id=0,
            frame_idx=0,
            bbox_xyxy=(10.0, 10.0, 20.0, 20.0),
            conf=0.8,
            cls=EntityClass.BALL,
        ),
        # Gap: missing frame 1, 2
        # Frame 3
        Track2D(
            track_id=0,
            frame_idx=3,
            bbox_xyxy=(40.0, 40.0, 50.0, 50.0),
            conf=0.8,
            cls=EntityClass.BALL,
        ),
        # Player untouched
        Track2D(
            track_id=5,
            frame_idx=0,
            bbox_xyxy=(100.0, 100.0, 150.0, 200.0),
            conf=0.9,
            cls=EntityClass.PLAYER,
        ),
    ]

    interpolated = interpolate_ball_gaps(tracks, max_gap_frames=3)
    ball_frames = [t.frame_idx for t in interpolated if t.cls == EntityClass.BALL]
    assert ball_frames == [0, 1, 2, 3]

    # Verify frame 1 interpolation midpoint
    f1_ball = next(t for t in interpolated if t.cls == EntityClass.BALL and t.frame_idx == 1)
    assert f1_ball.bbox_xyxy[0] == pytest.approx(20.0, abs=1e-3)
    assert f1_ball.bbox_xyxy[1] == pytest.approx(20.0, abs=1e-3)


@patch("ultralytics.YOLO")
def test_yolo_sahi_detector_mocked(mock_yolo_cls):
    # Setup mock YOLO prediction
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_boxes = MagicMock()
    import torch

    mock_boxes.xyxy = torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 50.0, 60.0, 60.0]])
    mock_boxes.conf = torch.tensor([0.9, 0.8])
    mock_boxes.cls = torch.tensor([0, 32])  # player, ball

    mock_pred = MagicMock()
    mock_pred.boxes = mock_boxes
    mock_model.predict.return_value = [mock_pred]

    yolo_cfg = YoloConfig(weights="mock.pt", device="cpu")
    sahi_cfg = SAHIConfig(enabled=False)
    detector = YOLOSAHIDetector(yolo_cfg, sahi_cfg)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    results = detector.detect(dummy_frame, frame_idx=0)

    assert len(results) == 2
    assert results[0].cls == EntityClass.PLAYER
    assert results[0].bbox_xyxy == (10.0, 20.0, 30.0, 40.0)
    assert results[1].cls == EntityClass.BALL
    assert results[1].bbox_xyxy == (50.0, 50.0, 60.0, 60.0)


@patch("supervision.ByteTrack", new_callable=MagicMock)
def test_bytetrack_wrapper_mocked(mock_bytetrack_cls):
    mock_tracker = MagicMock()
    mock_bytetrack_cls.return_value = mock_tracker

    # Mock supervision Detections output
    mock_tracked = MagicMock()
    mock_tracked.__len__.return_value = 1
    mock_tracked.xyxy = np.array([[10.0, 20.0, 30.0, 40.0]])
    mock_tracked.confidence = np.array([0.9])
    mock_tracked.tracker_id = np.array([1])
    mock_tracker.update_with_detections.return_value = mock_tracked

    cfg = ByteTrackConfig()
    wrapper = ByteTrackWrapper(cfg)

    dets = [
        Track2D(
            track_id=-1,
            frame_idx=0,
            bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
            conf=0.9,
            cls=EntityClass.PLAYER,
        )
    ]
    tracks = wrapper.update(dets, frame_idx=0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].cls == EntityClass.PLAYER


@patch("pitchlab.m6_detect_track.pipeline.VideoReader")
@patch("pitchlab.m6_detect_track.pipeline.YOLOSAHIDetector")
@patch("pitchlab.m6_detect_track.pipeline.ByteTrackWrapper")
@patch("pitchlab.m6_detect_track.pipeline.BallKalmanTracker")
def test_pipeline_integration_mocked(
    mock_ball_cls,
    mock_tracker_cls,
    mock_detector_cls,
    mock_reader_cls,
    tmp_path: Path,
):
    # Mock VideoReader generator yielding 2 frames
    mock_reader = MagicMock()
    mock_reader.__enter__.return_value = iter(
        [(0, np.zeros((10, 10, 3), dtype=np.uint8)), (1, np.zeros((10, 10, 3), dtype=np.uint8))]
    )
    mock_reader_cls.return_value = mock_reader

    # Mock Detector
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    mock_detector.detect.return_value = []

    # Mock Tracker
    mock_tracker = MagicMock()
    mock_tracker_cls.return_value = mock_tracker
    mock_tracker.update.return_value = [
        Track2D(
            track_id=1,
            frame_idx=0,
            bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
            conf=0.9,
            cls=EntityClass.PLAYER,
        )
    ]

    # Mock Ball Tracker
    mock_ball = MagicMock()
    mock_ball_cls.return_value = mock_ball
    mock_ball.update.return_value = None

    cfg = M6Config(
        yolo=YoloConfig(),
        sahi=SAHIConfig(),
        bytetrack=ByteTrackConfig(),
        kalman_ball=KalmanBallConfig(),
    )

    out_file = tmp_path / "out.parquet"
    tracks = run_m6_pipeline(
        video_path="dummy.mp4",
        cfg=cfg,
        output_path=out_file,
    )

    assert len(tracks) == 2
    assert out_file.exists()
