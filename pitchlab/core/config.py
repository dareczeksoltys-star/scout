from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class YoloConfig:
    weights: str = "yolov8x.pt"
    conf_thresh: float = 0.25
    iou_thresh: float = 0.45
    imgsz: int = 1280
    device: str = "cpu"


@dataclass
class SAHIConfig:
    enabled: bool = False
    slice_height: int = 640
    slice_width: int = 640
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2


@dataclass
class ByteTrackConfig:
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 25


@dataclass
class KalmanBallConfig:
    process_noise: float = 1.0
    measurement_noise: float = 10.0
    max_dist_px: float = 150.0
    max_missing_frames: int = 5


@dataclass
class M6Config:
    yolo: YoloConfig = field(default_factory=YoloConfig)
    sahi: SAHIConfig = field(default_factory=SAHIConfig)
    bytetrack: ByteTrackConfig = field(default_factory=ByteTrackConfig)
    kalman_ball: KalmanBallConfig = field(default_factory=KalmanBallConfig)


@dataclass
class PitchLabConfig:
    m6: M6Config = field(default_factory=M6Config)
    processing: dict[str, Any] = field(default_factory=dict)

    @property
    def m6_detect_track(self) -> M6Config:
        return self.m6


def _as_mapping(value: Any, section: str) -> dict[str, Any]:
    """Treat missing or null sections as empty mappings."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping.")
    return value


def load_config(path: str | Path) -> PitchLabConfig:
    """Load YAML configuration, using dataclass defaults for missing keys."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = _as_mapping(yaml.safe_load(handle), "root")

    # Prefer the canonical key if both names are present.
    m6_key = "m6" if "m6" in data else "m6_detect_track"
    m6_data = _as_mapping(data.get(m6_key), m6_key)

    m6 = M6Config(
        yolo=YoloConfig(
            **_as_mapping(m6_data.get("yolo"), f"{m6_key}.yolo")
        ),
        sahi=SAHIConfig(
            **_as_mapping(m6_data.get("sahi"), f"{m6_key}.sahi")
        ),
        bytetrack=ByteTrackConfig(
            **_as_mapping(m6_data.get("bytetrack"), f"{m6_key}.bytetrack")
        ),
        kalman_ball=KalmanBallConfig(
            **_as_mapping(m6_data.get("kalman_ball"), f"{m6_key}.kalman_ball")
        ),
    )
    processing = _as_mapping(data.get("processing"), "processing")

    return PitchLabConfig(m6=m6, processing=processing)
