"""Metrics calculator for pitch calibration benchmark.

Implements requirements from Section 7 (M5) of docs/PLAN.md:
- Pitch mask IoU (Intersection-over-Union).
- Reprojection error in pixels.
- Status distribution (ok, tracked, unreliable, no_pitch).
- Feet-in-bounds sanity check (>95%).
- Gate verification: median IoU on 'main' views >= 0.90.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from pitchlab.core.geometry import project_points
from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.core.types import CalibStatus, FrameCalib
from pitchlab.m2_calib.base import Calibrator
from pitchlab.m5_eval.dataset import EvalDataset, EvalSample


@dataclass
class SampleMetricResult:
    sample_id: str
    view: str
    status: CalibStatus
    conf: float
    iou: float
    reproj_error_mean_px: float
    reproj_error_median_px: float
    feet_in_bounds_ratio: Optional[float]


@dataclass
class EvaluationMetrics:
    backend_name: str
    total_samples: int
    main_view_count: int

    # Primary Stage 1 Quality Gate Metric
    median_iou_main: float
    mean_iou_main: float
    gate_passed: bool                  # median_iou_main >= 0.90

    # Secondary Metrics
    median_reproj_error_px: float
    mean_reproj_error_px: float
    status_distribution: Dict[str, float]
    sanity_feet_in_bounds_pct: float

    sample_results: List[SampleMetricResult] = field(default_factory=list)


def compute_pitch_image_mask(
    H_img2pitch: Optional[np.ndarray],
    img_w: int,
    img_h: int,
    pitch: PitchTemplate,
    num_samples_per_edge: int = 40,
) -> np.ndarray:
    """Renders a binary mask (uint8, 0 or 255) of the pitch boundary projected onto the image."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if H_img2pitch is None:
        return mask

    try:
        H_pitch2img = np.linalg.inv(H_img2pitch)
    except np.linalg.LinAlgError:
        return mask

    # Sample boundary of pitch in meters (clockwise)
    n = num_samples_per_edge
    top = np.column_stack([np.linspace(pitch.x_min, pitch.x_max, n), np.full(n, pitch.y_max)])
    right = np.column_stack([np.full(n, pitch.x_max), np.linspace(pitch.y_max, pitch.y_min, n)])
    bottom = np.column_stack([np.linspace(pitch.x_max, pitch.x_min, n), np.full(n, pitch.y_min)])
    left = np.column_stack([np.full(n, pitch.x_min), np.linspace(pitch.y_min, pitch.y_max, n)])

    perimeter_m = np.vstack([top, right[1:], bottom[1:], left[1:-1]])
    pts_img, valid = project_points(H_pitch2img, perimeter_m)

    if not np.any(valid) or len(pts_img[valid]) < 3:
        return mask

    valid_pts = pts_img[valid].astype(np.int32)
    # Clip coordinates to a safe bounding range to avoid rasterizer overflow
    clipped_pts = np.clip(valid_pts, -1000, max(img_w, img_h) + 1000)
    cv2.fillPoly(mask, [clipped_pts], 255)
    return mask


def compute_mask_iou(
    H_pred: Optional[np.ndarray],
    H_gt: Optional[np.ndarray],
    img_w: int,
    img_h: int,
    pitch: PitchTemplate,
) -> float:
    """Computes IoU between predicted pitch mask and ground-truth pitch mask."""
    if H_pred is None or H_gt is None:
        return 0.0

    mask_pred = compute_pitch_image_mask(H_pred, img_w, img_h, pitch)
    mask_gt = compute_pitch_image_mask(H_gt, img_w, img_h, pitch)

    intersection = np.logical_and(mask_pred > 0, mask_gt > 0).sum()
    union = np.logical_or(mask_pred > 0, mask_gt > 0).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_reprojection_error(
    H_pred: Optional[np.ndarray],
    keypoints_px: Dict[str, Tuple[float, float]],
    pitch: PitchTemplate,
) -> Tuple[float, float]:
    """Computes mean and median Euclidean reprojection error in pixels."""
    if H_pred is None or not keypoints_px:
        return float("nan"), float("nan")

    try:
        H_pitch2img = np.linalg.inv(H_pred)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")

    pitch_kps = pitch.get_keypoints()
    errors: List[float] = []

    for name, (u_gt, v_gt) in keypoints_px.items():
        if name in pitch_kps:
            x_m, y_m = pitch_kps[name]
            pt_img, valid = project_points(H_pitch2img, np.array([[x_m, y_m]]))
            if valid[0]:
                u_pred, v_pred = pt_img[0]
                dist = float(np.hypot(u_pred - u_gt, v_pred - v_gt))
                errors.append(dist)

    if not errors:
        return float("nan"), float("nan")

    return float(np.mean(errors)), float(np.median(errors))


def compute_feet_sanity(
    H_pred: Optional[np.ndarray],
    feet_points_px: Optional[np.ndarray],
    pitch: PitchTemplate,
    margin_m: float = 1.0,
) -> Optional[float]:
    """Computes ratio of detected player feet landing inside pitch boundaries."""
    if H_pred is None or feet_points_px is None or len(feet_points_px) == 0:
        return None

    pts_pitch, valid = project_points(H_pred, feet_points_px)
    if not np.any(valid):
        return 0.0

    valid_feet = pts_pitch[valid]
    inside_count = sum(pitch.is_inside(x, y, margin_m=margin_m) for x, y in valid_feet)
    return float(inside_count) / float(len(valid_feet))


def evaluate_calibration(
    calibrator: Calibrator,
    dataset: EvalDataset,
    pitch: PitchTemplate,
) -> EvaluationMetrics:
    """Evaluates a calibrator backend against an EvalDataset."""
    results: List[SampleMetricResult] = []
    status_counts: Dict[str, int] = {"ok": 0, "tracked": 0, "unreliable": 0, "no_pitch": 0}
    iou_main_list: List[float] = []
    reproj_errors_list: List[float] = []
    feet_sanity_ratios: List[float] = []

    for idx, sample in enumerate(dataset):
        h, w = sample.image.shape[:2]
        calib_res = calibrator.calibrate(sample.image, pitch, frame_idx=idx, t=idx * 0.1)

        status_counts[calib_res.status] = status_counts.get(calib_res.status, 0) + 1

        # IoU
        iou = compute_mask_iou(calib_res.H_img2pitch, sample.H_gt, w, h, pitch)
        if sample.view == "main":
            iou_main_list.append(iou)

        # Reprojection error
        mean_err, med_err = compute_reprojection_error(calib_res.H_img2pitch, sample.keypoints_px, pitch)
        if not np.isnan(med_err):
            reproj_errors_list.append(med_err)

        # Sanity check
        feet_ratio = compute_feet_sanity(calib_res.H_img2pitch, sample.feet_points_px, pitch)
        if feet_ratio is not None:
            feet_sanity_ratios.append(feet_ratio)

        results.append(SampleMetricResult(
            sample_id=sample.sample_id,
            view=sample.view,
            status=calib_res.status,
            conf=calib_res.conf,
            iou=iou,
            reproj_error_mean_px=mean_err,
            reproj_error_median_px=med_err,
            feet_in_bounds_ratio=feet_ratio,
        ))

    total = len(dataset)
    status_dist = {k: (v / total * 100.0 if total > 0 else 0.0) for k, v in status_counts.items()}

    median_iou_main = float(np.median(iou_main_list)) if iou_main_list else 0.0
    mean_iou_main = float(np.mean(iou_main_list)) if iou_main_list else 0.0
    median_reproj = float(np.median(reproj_errors_list)) if reproj_errors_list else float("nan")
    mean_reproj = float(np.mean(reproj_errors_list)) if reproj_errors_list else float("nan")
    feet_pct = (float(np.mean(feet_sanity_ratios)) * 100.0) if feet_sanity_ratios else 100.0

    # Gate condition: median IoU on main view >= 0.90
    gate_passed = (median_iou_main >= 0.90)

    return EvaluationMetrics(
        backend_name=calibrator.name,
        total_samples=total,
        main_view_count=len(iou_main_list),
        median_iou_main=median_iou_main,
        mean_iou_main=mean_iou_main,
        gate_passed=gate_passed,
        median_reproj_error_px=median_reproj,
        mean_reproj_error_px=mean_reproj,
        status_distribution=status_dist,
        sanity_feet_in_bounds_pct=feet_pct,
        sample_results=results,
    )
