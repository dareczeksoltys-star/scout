"""Roboflow sports calibration backend.

Adapter for keypoint-based pitch calibration (e.g. YOLO pose / keypoint detector from roboflow/sports).
Matches detected 2D pixel keypoints to 3D/2D FIFA pitch coordinates and calculates homography H_img2pitch.
Validates H and computes pitch footprint in meters.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from pitchlab.core.geometry import compute_footprint, validate_homography
from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.core.types import FrameCalib
from pitchlab.m2_calib.base import Calibrator, make_empty_calib, register_calibrator


@register_calibrator("roboflow")
class RoboflowCalibrator:
    """Pitch calibrator based on keypoint landmark detection."""
    name: str = "roboflow"

    def __init__(
        self,
        model_path: Optional[str] = None,
        keypoint_detector: Optional[Callable[[np.ndarray], Dict[str, Tuple[float, float]]]] = None,
        min_keypoints: int = 4,
        ransac_threshold_m: float = 3.0,
    ) -> None:
        self.model_path = model_path
        self.detector = keypoint_detector
        self.min_keypoints = min_keypoints
        self.ransac_threshold_m = ransac_threshold_m

    def detect_keypoints(self, frame: np.ndarray, pitch: PitchTemplate) -> Dict[str, Tuple[float, float]]:
        """Detects 2D image pixel coordinates of pitch landmark keypoints.
        
        Returns mapping: landmark_name -> (u_px, v_px).
        """
        if self.detector is not None:
            return self.detector(frame)
        # Default placeholder / heuristic: if no model loaded, returns empty
        return {}

    def calibrate(
        self,
        frame: np.ndarray,
        pitch: PitchTemplate,
        frame_idx: int = 0,
        t: float = 0.0,
    ) -> FrameCalib:
        h, w = frame.shape[:2]

        # 1. Detect keypoints in image coordinates
        kps_img = self.detect_keypoints(frame, pitch)
        pitch_kps = pitch.get_keypoints()

        # 2. Match with known ground-truth coordinates on pitch
        src_pts: List[Tuple[float, float]] = []
        dst_pts: List[Tuple[float, float]] = []

        for name, img_xy in kps_img.items():
            if name in pitch_kps:
                src_pts.append(img_xy)
                dst_pts.append(pitch_kps[name])

        if len(src_pts) < self.min_keypoints:
            return make_empty_calib(frame_idx, t, status="no_pitch")

        src_arr = np.array(src_pts, dtype=np.float64)
        dst_arr = np.array(dst_pts, dtype=np.float64)

        # 3. Estimate Homography with RANSAC
        H, inliers = cv2.findHomography(
            src_arr,
            dst_arr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold_m,
        )

        if H is None or inliers is None:
            return make_empty_calib(frame_idx, t, status="unreliable")

        # 4. Physical sanity validation
        is_valid, reason = validate_homography(H, pitch, img_width=w, img_height=h)
        if not is_valid:
            return FrameCalib(
                frame_idx=frame_idx,
                t=t,
                status="unreliable",
                H_img2pitch=H,
                conf=0.0,
                footprint_m=None,
                source="keypoints",
            )

        # 5. Compute inlier ratio confidence
        conf = float(np.sum(inliers)) / float(len(inliers))

        # 6. Compute camera footprint clipped to pitch
        footprint_m = compute_footprint(H, img_width=w, img_height=h, pitch=pitch)

        return FrameCalib(
            frame_idx=frame_idx,
            t=t,
            status="ok",
            H_img2pitch=H,
            conf=conf,
            footprint_m=footprint_m,
            source="keypoints",
        )
