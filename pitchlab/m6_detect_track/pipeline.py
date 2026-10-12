from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import cv2
import numpy as np

from pitchlab.core.config import M6Config
from pitchlab.core.io import VideoReader, save_tracks_parquet
from pitchlab.core.types import Track2D
from pitchlab.m6_detect_track.ball_tracker import BallKalmanTracker
from pitchlab.m6_detect_track.detector import YOLOSAHIDetector
from pitchlab.m6_detect_track.interpolation import interpolate_ball_gaps
from pitchlab.m6_detect_track.tracker import ByteTrackWrapper

logger = logging.getLogger(__name__)


def run_m6_pipeline(
    video_path: str | Path,
    cfg: M6Config,
    output_path: str | Path | None = None,
    max_frames: int | None = None,
) -> list[Track2D]:
    """Runs end-to-end detection, multi-person tracking, ball tracking and interpolation.

    Args:
        video_path: Path to source video file.
        cfg: Module 6 configuration dataclass.
        output_path: Optional path to save Parquet output.
        max_frames: Optional max frames to process (useful for testing).

    Returns:
        List of Track2D records.
    """
    video_path = Path(video_path)
    logger.info("Starting M6 pipeline on video: %s", video_path)

    detector = YOLOSAHIDetector(yolo_cfg=cfg.yolo, sahi_cfg=cfg.sahi)
    tracker = ByteTrackWrapper(cfg=cfg.bytetrack)
    ball_tracker = BallKalmanTracker(cfg=cfg.kalman_ball)

    all_tracks: list[Track2D] = []

    with VideoReader(video_path) as reader:
        for frame_idx, frame in reader:
            if max_frames is not None and frame_idx >= max_frames:
                logger.info("Reached maximum frames limit (%d)", max_frames)
                break

            # 1. Detect entities
            detections = detector.detect(frame, frame_idx=frame_idx)

            # 2. Track players via ByteTrack
            player_tracks = tracker.update(detections, frame_idx=frame_idx)

            # 3. Track ball via Kalman Filter
            ball_track = ball_tracker.update(detections, frame_idx=frame_idx)

            all_tracks.extend(player_tracks)
            if ball_track is not None:
                all_tracks.append(ball_track)

    logger.info("Finished tracking loop. Total raw tracks collected: %d", len(all_tracks))

    # 4. Interpolate missing ball frames
    processed_tracks = interpolate_ball_gaps(
        all_tracks, max_gap_frames=cfg.kalman_ball.max_missing_frames
    )
    logger.info("Finished ball interpolation. Final tracks count: %d", len(processed_tracks))

    # 5. Persist to Parquet if output path specified
    if output_path is not None:
        save_tracks_parquet(processed_tracks, output_path)
        logger.info("Saved tracks to Parquet: %s", output_path)

    return processed_tracks


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
MAX_ANALYSIS_FRAMES = 150


def _cfg_value(cfg: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(cfg, name, None)
        if value is not None:
            return value
    return default


class _SegmentDetector:
    """Ultralytics adapter exposing detect(frame, frame_idx).

    If your existing M6 detector already exposes that interface, replace this
    adapter in _make_segment_components().
    """

    def __init__(self, cfg: Any) -> None:
        import torch
        from ultralytics import YOLO

        device = _cfg_value(cfg, "device", default=None)
        device_text = str(device).lower()

        requests_cuda = (
            device_text.startswith("cuda")
            or device_text.isdigit()
            or device_text == "gpu"
        )
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif requests_cuda and not torch.cuda.is_available():
            device = "cpu"

        # Adapt these attribute names to your M6Config if necessary.
        weights = _cfg_value(
            cfg,
            "model_path",
            "weights",
            "model_weights",
            default="yolov8n.pt",
        )
        self.model = YOLO(str(weights))
        self.device = device
        self.confidence = float(
            _cfg_value(cfg, "conf_threshold", "confidence", default=0.20)
        )
        self.image_size = int(_cfg_value(cfg, "imgsz", default=640))

    def detect(
        self, frame: np.ndarray, frame_idx: int
    ) -> list[dict[str, Any]]:
        result = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )[0]

        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for bbox, confidence, class_id in zip(boxes, confidences, classes):
            label = str(result.names[int(class_id)]).lower()
            if label in {"person", "player", "referee"}:
                kind = "player"
            elif label in {"sports ball", "ball", "football", "soccer ball"}:
                kind = "ball"
            else:
                continue

            detections.append(
                {
                    "bbox": [float(v) for v in bbox],
                    "conf": float(confidence),
                    "kind": kind,
                    "label": label,
                }
            )
        return detections


class _SegmentTracker:
    """Lightweight centroid association.

    Replace with your existing M6 player/ball trackers for production-quality
    identity preservation through occlusions.
    """

    def __init__(self, max_missing: int = 8) -> None:
        self.next_id = 1
        self.tracks: dict[int, dict[str, Any]] = {}
        self.max_missing = max_missing

    @staticmethod
    def _center(detection: dict[str, Any]) -> tuple[float, float]:
        x1, y1, x2, y2 = detection["bbox"]
        return (x1 + x2) / 2, (y1 + y2) / 2

    def update(
        self,
        detections: list[dict[str, Any]],
        frame_idx: int,
        image_shape: tuple[int, ...],
    ) -> list[dict[str, Any]]:
        height, width = image_shape[:2]
        max_distance = 0.15 * math.hypot(width, height)

        for state in self.tracks.values():
            state["missing"] += 1

        candidates: list[tuple[float, int, int]] = []
        for track_id, state in self.tracks.items():
            old_x, old_y = self._center(state["detection"])
            for detection_idx, detection in enumerate(detections):
                new_x, new_y = self._center(detection)
                distance = math.hypot(new_x - old_x, new_y - old_y)
                if distance <= max_distance:
                    candidates.append((distance, track_id, detection_idx))

        assigned_tracks: set[int] = set()
        assigned_detections: dict[int, int] = {}

        for _, track_id, detection_idx in sorted(candidates):
            if (
                track_id in assigned_tracks
                or detection_idx in assigned_detections
            ):
                continue
            assigned_tracks.add(track_id)
            assigned_detections[detection_idx] = track_id

        output = []
        for detection_idx, detection in enumerate(detections):
            track_id = assigned_detections.get(detection_idx)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1

            self.tracks[track_id] = {
                "detection": detection,
                "missing": 0,
            }
            output.append({**detection, "track_id": track_id})

        self.tracks = {
            track_id: state
            for track_id, state in self.tracks.items()
            if state["missing"] <= self.max_missing
        }
        return output


def _make_segment_components(cfg: Any) -> tuple[Any, Any, Any]:
    """Integration point for your existing M6 detector and trackers."""
    return _SegmentDetector(cfg), _SegmentTracker(), _SegmentTracker()


def _pitch_point(
    x: float, y: float, width: int, height: int
) -> tuple[float, float]:
    return (
        round(float(np.clip(x / width, 0, 1)) * PITCH_LENGTH, 3),
        round(float(np.clip(y / height, 0, 1)) * PITCH_WIDTH, 3),
    )


def _pixel_bbox(
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    ball: bool = False,
) -> list[float]:
    """Create a demo bbox whose feet/center maps to (x, y) on the pitch."""
    px = x / PITCH_LENGTH * width
    py = y / PITCH_WIDTH * height

    if ball:
        radius = max(2.0, width * 0.004)
        bbox = [px - radius, py - radius, px + radius, py + radius]
    else:
        box_width = width * 0.018
        box_height = height * 0.075
        bbox = [
            px - box_width / 2,
            py - box_height,
            px + box_width / 2,
            py,
        ]

    return [
        round(float(np.clip(value, 0, limit)), 2)
        for value, limit in zip(bbox, [width, height, width, height])
    ]


def _synthetic_frame(
    frame_idx: int,
    fps: float,
    start_time: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Smooth, deterministic demo movement: 22 players, referee, and passes."""
    elapsed = frame_idx / fps
    players = []

    # Formation positions are mirrored for the opposing team.
    formation = [
        (7, 34),
        (23, 10), (23, 26), (23, 42), (23, 58),
        (40, 13), (40, 34), (40, 55),
        (55, 15), (59, 34), (55, 53),
    ]

    for team_idx, team in enumerate(("A", "B")):
        for player_idx, (base_x, base_y) in enumerate(formation):
            track_id = team_idx * 11 + player_idx + 1
            if team_idx:
                base_x = PITCH_LENGTH - base_x

            phase = track_id * 0.61
            x = float(np.clip(
                base_x + 2.8 * math.sin(elapsed * 0.45 + phase),
                1, 104,
            ))
            y = float(np.clip(
                base_y + 1.8 * math.sin(elapsed * 0.38 + phase),
                1, 67,
            ))
            players.append(
                {
                    "track_id": track_id,
                    "team": team,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "bbox": _pixel_bbox(x, y, width, height),
                    # Zero confidence: this is simulated, not detected.
                    "conf": 0.0,
                }
            )

    ref_x = 52.5 + 5 * math.sin(elapsed * 0.25)
    ref_y = 34 + 4 * math.cos(elapsed * 0.30)
    players.append(
        {
            "track_id": 23,
            "team": "referee",
            "x": round(ref_x, 3),
            "y": round(ref_y, 3),
            "bbox": _pixel_bbox(ref_x, ref_y, width, height),
            "conf": 0.0,
        }
    )

    # Pass every 1.6 seconds between nearby players on team A.
    receivers = [5, 6, 7, 10, 9, 8]
    pass_number = int(elapsed / 1.6)
    progress = (elapsed % 1.6) / 1.6
    source = players[receivers[pass_number % len(receivers)]]
    target = players[receivers[(pass_number + 1) % len(receivers)]]

    ball_x = (1 - progress) * source["x"] + progress * target["x"]
    ball_y = (1 - progress) * source["y"] + progress * target["y"]
    ball_y += 0.8 * math.sin(math.pi * progress)

    return {
        "frame_idx": frame_idx,
        "timestamp": round(start_time + elapsed, 2),
        "players": players,
        "ball": {
            "x": round(ball_x, 3),
            "y": round(ball_y, 3),
            "bbox": _pixel_bbox(
                ball_x, ball_y, width, height, ball=True
            ),
            "conf": 0.0,
        },
    }


def analyze_video_segment(
    video_path: str | Path,
    start_time: float,
    end_time: float,
    cfg: M6Config | None = None,
    output_dir: str | Path | None = None,
    segment_id: str | None = None,
) -> dict[str, Any]:
    """Analyze a half-open video interval [start_time, end_time).

    frame_idx is relative to int(start_time * fps), not the sampled-frame
    sequence number. total_frames is the number of processed samples.

    output_dir is reserved for M6 artifacts; JSON persistence is handled by
    the server.
    """
    start_time = float(start_time)
    end_time = float(end_time)
    if (
        not math.isfinite(start_time)
        or not math.isfinite(end_time)
        or start_time < 0
        or end_time <= start_time
    ):
        raise ValueError("Require finite times with 0 <= start_time < end_time")

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if cfg is None:
        try:
            # If your project has a default-config loader, call it here.
            cfg = M6Config()
        except Exception:
            logger.warning("Default M6 config unavailable; using demo defaults")
            cfg = SimpleNamespace(
                model_path="yolov8n.pt",
                device="cpu",
                conf_threshold=0.20,
                imgsz=640,
            )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Video could not be opened")

    frames: list[dict[str, Any]] = []
    sample_metadata: list[tuple[int, int, int]] = []
    synthetic = False
    fallback_reason: str | None = None

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps_estimated = not math.isfinite(fps) or fps <= 0
        if fps_estimated:
            fps = 25.0

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        if end_frame <= start_frame:
            raise ValueError("Segment contains no complete frame interval")

        raw_frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if math.isfinite(raw_frame_count) and raw_frame_count > 0:
            end_frame = min(end_frame, int(raw_frame_count))
        if start_frame >= end_frame:
            raise ValueError("Segment starts beyond the available video")

        # Every other frame at >=25fps; increase stride for longer segments.
        sample_rate = max(
            2 if fps >= 25 else 1,
            math.ceil((end_frame - start_frame) / MAX_ANALYSIS_FRAMES),
        )

        detector = tracker = ball_tracker = None

        def enable_fallback(exc: Exception) -> None:
            nonlocal synthetic, fallback_reason, frames
            synthetic = True
            fallback_reason = type(exc).__name__
            logger.warning(
                "M6 unavailable; using synthetic segment analysis",
                exc_info=True,
            )
            # Regenerate previous samples to avoid mixing real and fake IDs.
            frames = [
                _synthetic_frame(idx, fps, start_time, width, height)
                for idx, width, height in sample_metadata
            ]

        # Seek once; grab skipped frames without copying them into Python.
        if start_frame:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
                raise ValueError("Video backend could not seek to segment")

        absolute_idx = start_frame
        while absolute_idx < end_frame and len(frames) < MAX_ANALYSIS_FRAMES:
            relative_idx = absolute_idx - start_frame

            if relative_idx % sample_rate:
                if not cap.grab():
                    break
                absolute_idx += 1
                continue

            ok, frame = cap.read()
            if not ok:
                break

            height, width = frame.shape[:2]
            sample_metadata.append((relative_idx, width, height))

            if synthetic:
                frames.append(
                    _synthetic_frame(
                        relative_idx, fps, start_time, width, height
                    )
                )
            else:
                try:
                    # Lazy loading avoids downloading weights for empty input.
                    if detector is None:
                        detector, tracker, ball_tracker = (
                            _make_segment_components(cfg)
                        )

                    detections = detector.detect(frame, relative_idx)
                    player_tracks = tracker.update(
                        [d for d in detections if d["kind"] == "player"],
                        relative_idx,
                        frame.shape,
                    )

                    ball_detections = [
                        d for d in detections if d["kind"] == "ball"
                    ]
                    # Keep at most one ball observation.
                    ball_detections = sorted(
                        ball_detections,
                        key=lambda d: d["conf"],
                        reverse=True,
                    )[:1]
                    ball_tracks = ball_tracker.update(
                        ball_detections, relative_idx, frame.shape
                    )

                    players = []
                    for track in player_tracks:
                        x1, y1, x2, y2 = track["bbox"]
                        pitch_x, pitch_y = _pitch_point(
                            (x1 + x2) / 2, y2, width, height
                        )
                        track_id = int(track["track_id"])
                        team = (
                            "referee"
                            if track.get("label") == "referee"
                            else ("A" if track_id % 2 else "B")
                        )
                        players.append(
                            {
                                "track_id": track_id,
                                "team": team,
                                "x": pitch_x,
                                "y": pitch_y,
                                "bbox": track["bbox"],
                                "conf": float(track["conf"]),
                            }
                        )

                    ball = None
                    if ball_tracks:
                        track = ball_tracks[0]
                        x1, y1, x2, y2 = track["bbox"]
                        pitch_x, pitch_y = _pitch_point(
                            (x1 + x2) / 2,
                            (y1 + y2) / 2,
                            width,
                            height,
                        )
                        ball = {
                            "x": pitch_x,
                            "y": pitch_y,
                            "bbox": track["bbox"],
                            "conf": float(track["conf"]),
                        }

                    frames.append(
                        {
                            "frame_idx": relative_idx,
                            "timestamp": round(
                                start_time + relative_idx / fps, 2
                            ),
                            "players": players,
                            "ball": ball,
                        }
                    )
                except Exception as exc:
                    # Includes weight loading, inference, and tracker failures.
                    # The current sample is already in sample_metadata.
                    enable_fallback(exc)

            absolute_idx += 1

    finally:
        cap.release()

    if not frames:
        raise ValueError("No readable frames in the requested segment")

    unique_players = {
        player["track_id"]
        for frame in frames
        for player in frame["players"]
        if player["team"] != "referee"
    }
    ball_frames = sum(frame["ball"] is not None for frame in frames)

    return {
        "segment_id": segment_id or "segment",
        "start_time": start_time,
        "end_time": end_time,
        "duration": round(end_time - start_time, 2),
        "fps": fps,
        "total_frames": len(frames),
        "frames": frames,
        "stats": {
            "tracked_players_count": len(unique_players),
            # Percentage in [0, 100]. Synthetic output is NOT detection.
            "ball_detection_rate": (
                0.0 if synthetic else round(100 * ball_frames / len(frames), 2)
            ),
            "ball_presence_rate": round(100 * ball_frames / len(frames), 2),
        },
        "synthetic": synthetic,
        "fallback_reason": fallback_reason,
        "sample_rate": sample_rate,
        "fps_estimated": fps_estimated,
        "coordinate_mapping": "normalized_video",
    }
