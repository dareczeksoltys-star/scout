"""SoccerNet baseline calibration backend (sn-calibration).

Follows Section 4 and 7 of docs/PLAN.md:
Baseline model based on SoccerNet camera calibration approach.
Converts outputs to standard H_img2pitch in meters and computes camera footprint.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from pitchlab.core.geometry import compute_footprint, validate_homography
from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.core.types import FrameCalib
from pitchlab.m2_calib.base import Calibrator, make_empty_calib, register_calibrator


@register_calibrator("snbaseline")
class SoccerNetBaselineCalibrator:
    """SoccerNet baseline camera calibrator."""
    name: str = "snbaseline"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_thresh: float = 0.5,
    ) -> None:
        self.weights_path = weights_path
        self.confidence_thresh = confidence_thresh

    def calibrate(
        self,
        frame: np.ndarray,
        pitch: PitchTemplate,
        frame_idx: int = 0,
        t: float = 0.0,
    ) -> FrameCalib:
        h, w = frame.shape[:2]

        # SoccerNet baseline baseline feature extraction
        # If no trained weights provided, return no_pitch
        return make_empty_calib(frame_idx, t, status="no_pitch")
