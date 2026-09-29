"""Calibration pipeline runner.

Executes calibrator over video frames, collects FrameCalib results,
and saves outputs as Parquet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

from pitchlab.core.config import AppConfig
from pitchlab.core.io import VideoReader, save_parquet
from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.core.types import FrameCalib, SCHEMA_VERSION
from pitchlab.m2_calib.base import Calibrator, get_calibrator


def calib_to_dict(calib: FrameCalib) -> dict:
    """Converts a FrameCalib dataclass to a flat dictionary for Parquet storage."""
    h_flat = calib.H_img2pitch.flatten().tolist() if calib.H_img2pitch is not None else [None] * 9
    footprint_json = json.dumps(calib.footprint_m.tolist()) if calib.footprint_m is not None else None

    row = {
        "frame_idx": calib.frame_idx,
        "t": calib.t,
        "status": calib.status,
        "conf": calib.conf,
        "source": calib.source,
        "footprint_m": footprint_json,
        "schema_version": SCHEMA_VERSION,
    }
    for i, val in enumerate(h_flat):
        row[f"h_{i // 3}_{i % 3}"] = val
    return row


def run_calibration(
    video_path: Union[str, Path],
    output_parquet_path: Optional[Union[str, Path]] = None,
    calibrator: Optional[Calibrator] = None,
    backend_name: str = "roboflow",
    config: Optional[AppConfig] = None,
) -> List[FrameCalib]:
    """Runs pitch calibration on a video.
    
    Args:
        video_path: Path to video file (local or gs://).
        output_parquet_path: Destination path for Parquet output.
        calibrator: Optional Calibrator instance (or creates via get_calibrator).
        backend_name: Name of backend if calibrator is not provided.
        config: Optional AppConfig.
        
    Returns:
        List of FrameCalib objects.
    """
    cfg = config or AppConfig.load()
    pitch = PitchTemplate(length_m=cfg.pitch.length_m, width_m=cfg.pitch.width_m)
    calib = calibrator or get_calibrator(backend_name)

    reader = VideoReader(video_path)
    target_fps = cfg.get_effective_fps()

    results: List[FrameCalib] = []
    rows: List[dict] = []

    for frame_idx, t_sec, frame_bgr in reader.iter_frames(target_fps=target_fps):
        res = calib.calibrate(frame_bgr, pitch, frame_idx=frame_idx, t=t_sec)
        results.append(res)
        rows.append(calib_to_dict(res))

    if output_parquet_path is not None and rows:
        df = pd.DataFrame(rows)
        save_parquet(df, output_parquet_path)

    return results
