"""Configuration loader for PITCHLAB.

Reads YAML configurations (e.g. configs/default.yaml) and applies environment overrides.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class PitchConfig:
    length_m: float = 105.0
    width_m: float = 68.0


@dataclass
class SamplingConfig:
    dev_fps: int = 5
    gpu_fps: int = 25


@dataclass
class CameraShotsConfig:
    min_scene_length_sec: float = 1.0
    green_hsv_lower: List[int] = field(default_factory=lambda: [35, 40, 40])
    green_hsv_upper: List[int] = field(default_factory=lambda: [85, 255, 255])
    main_view_green_ratio_min: float = 0.40


@dataclass
class CalibrationConfig:
    default_backend: str = "roboflow"
    iou_threshold_reliable: float = 0.85
    min_lines_detected: int = 4


@dataclass
class TrackingConfig:
    conf_threshold: float = 0.35
    ball_conf_threshold: float = 0.20
    track_buffer_frames: int = 30


@dataclass
class KinematicsConfig:
    max_speed_m_per_s: float = 12.0
    max_accel_m_per_s2: float = 9.0


@dataclass
class GCPConfig:
    project_id: str = ""
    storage_bucket: str = "pitchlab-data"
    firestore_collection_matches: str = "matches"
    firestore_collection_weights: str = "models/scoring_weights"


@dataclass
class AppConfig:
    pitch: PitchConfig = field(default_factory=PitchConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    camera_shots: CameraShotsConfig = field(default_factory=CameraShotsConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    gcp: GCPConfig = field(default_factory=GCPConfig)
    env_mode: str = "dev"  # "dev" or "gpu"

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "AppConfig":
        """Loads configuration from YAML file and applies environment overrides."""
        data: Dict[str, Any] = {}
        if config_path is None:
            # Default to configs/default.yaml if present
            default_p = Path("configs/default.yaml")
            if default_p.is_file():
                config_path = default_p

        if config_path and Path(config_path).is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        pitch_cfg = PitchConfig(**data.get("pitch", {}))
        sampling_cfg = SamplingConfig(**data.get("sampling", {}))
        camera_shots_cfg = CameraShotsConfig(**data.get("camera_shots", {}))
        calib_cfg = CalibrationConfig(**data.get("calibration", {}))
        track_cfg = TrackingConfig(**data.get("tracking", {}))
        kinematics_cfg = KinematicsConfig(**data.get("kinematics", {}))
        gcp_cfg = GCPConfig(**data.get("gcp", {}))

        # Detect or override env mode
        env_mode = os.getenv("PITCHLAB_ENV")
        if not env_mode:
            # Check if CUDA is available in PyTorch without hard error
            try:
                import torch
                env_mode = "gpu" if torch.cuda.is_available() else "dev"
            except Exception:
                env_mode = "dev"

        # Allow GCP environment variable overrides
        if os.getenv("GCP_PROJECT_ID"):
            gcp_cfg.project_id = os.environ["GCP_PROJECT_ID"]
        if os.getenv("GCS_BUCKET"):
            gcp_cfg.storage_bucket = os.environ["GCS_BUCKET"]

        return cls(
            pitch=pitch_cfg,
            sampling=sampling_cfg,
            camera_shots=camera_shots_cfg,
            calibration=calib_cfg,
            tracking=track_cfg,
            kinematics=kinematics_cfg,
            gcp=gcp_cfg,
            env_mode=env_mode,
        )

    def get_effective_fps(self) -> int:
        """Returns the processing FPS based on current environment mode."""
        return self.sampling.gpu_fps if self.env_mode == "gpu" else self.sampling.dev_fps
