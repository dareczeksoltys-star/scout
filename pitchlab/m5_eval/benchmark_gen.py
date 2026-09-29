"""Benchmark dataset generator for calibration evaluation (M5).

Generates 30-50 diverse frames covering:
- Left penalty box & goal (main camera view)
- Midfield & center circle (main camera view)
- Right penalty box & goal (main camera view)
- Close-up shots with limited lines
- Crowd / graphic scenes
Includes ground-truth homographies, visible keypoints, and player feet positions.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

from pitchlab.core.geometry import project_points
from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.core.types import ViewType
from pitchlab.m5_eval.dataset import EvalDataset, EvalSample


def generate_benchmark_dataset(
    pitch: Optional[PitchTemplate] = None,
    output_dir: Optional[str | Path] = None,
    img_w: int = 1280,
    img_h: int = 720,
) -> EvalDataset:
    """Generates standard M5 evaluation dataset of 36 frames with ground-truth."""
    pitch_tmpl = pitch or PitchTemplate(105.0, 68.0)
    samples: List[EvalSample] = []
    lines = pitch_tmpl.get_lines()
    all_kps = pitch_tmpl.get_keypoints()

    # Define 32 main-view camera perspectives spanning the pitch
    # Varying pan (-40m to +40m), tilt, zoom (scale 11.0 to 18.0)
    pan_positions = np.linspace(-42.0, 42.0, 32)

    for i, pan_x in enumerate(pan_positions):
        sample_id = f"eval_main_{i+1:02d}"
        scale = 13.5 + 2.5 * math.sin(i * 0.7)
        cam_y_offset = 6.0 + 3.0 * math.cos(i * 0.5)

        # H: pitch meters -> image pixels
        H_pitch2img = np.array([
            [scale, 0.0, -pan_x * scale + img_w / 2],
            [0.0, -scale, cam_y_offset * scale + img_h / 2 + 80],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        H_img2pitch = np.linalg.inv(H_pitch2img)

        # Render image
        img = np.full((img_h, img_w, 3), (38, 128, 46), dtype=np.uint8)

        # Draw pitch markings
        for line in lines:
            xs = np.linspace(line.p1[0], line.p2[0], 25)
            ys = np.linspace(line.p1[1], line.p2[1], 25)
            pts_img, valid = project_points(H_pitch2img, np.column_stack([xs, ys]))
            valid_pts = pts_img[valid].astype(np.int32)
            if len(valid_pts) >= 2:
                cv2.polylines(img, [valid_pts], False, (245, 245, 245), 2, cv2.LINE_AA)

        # Collect visible keypoints inside image boundary
        visible_kps: Dict[str, Tuple[float, float]] = {}
        for name, (x_m, y_m) in all_kps.items():
            pt_img, valid = project_points(H_pitch2img, np.array([[x_m, y_m]]))
            if valid[0]:
                u, v = pt_img[0]
                if 20 <= u <= img_w - 20 and 20 <= v <= img_h - 20:
                    visible_kps[name] = (float(u), float(v))
                    # Draw subtle landmark spot
                    cv2.circle(img, (int(u), int(v)), 3, (255, 255, 255), -1)

        # Simulate player foot points (5-10 players on pitch in camera view)
        feet_pts: List[Tuple[float, float]] = []
        for p_idx in range(8):
            p_x = pan_x + np.random.uniform(-25.0, 25.0)
            p_y = np.random.uniform(pitch_tmpl.y_min + 5.0, pitch_tmpl.y_max - 5.0)
            p_img, valid = project_points(H_pitch2img, np.array([[p_x, p_y]]))
            if valid[0]:
                u, v = p_img[0]
                if 0 <= u < img_w and 0 <= v < img_h:
                    feet_pts.append((float(u), float(v)))

        feet_arr = np.array(feet_pts, dtype=np.float64) if feet_pts else None

        samples.append(EvalSample(
            sample_id=sample_id,
            image=img,
            view="main",
            H_gt=H_img2pitch,
            keypoints_px=visible_kps,
            feet_points_px=feet_arr,
        ))

    # Add 2 closeup frames (tight on player, almost no pitch lines)
    for c_i in range(2):
        sample_id = f"eval_closeup_{c_i+1:02d}"
        img_close = np.full((img_h, img_w, 3), (35, 115, 42), dtype=np.uint8)
        # Only one line visible
        cv2.line(img_close, (0, 600), (img_w, 620), (240, 240, 240), 4)
        samples.append(EvalSample(
            sample_id=sample_id,
            image=img_close,
            view="closeup",
            H_gt=None,
            keypoints_px={},
            feet_points_px=np.array([[img_w / 2, 580]], dtype=np.float64),
        ))

    # Add 2 crowd/graphics frames (stands, studio graphics)
    for g_i in range(2):
        sample_id = f"eval_graphic_{g_i+1:02d}"
        img_gfx = np.full((img_h, img_w, 3), (25, 25, 30), dtype=np.uint8)
        cv2.putText(img_gfx, "MATCH STATS & CROWD", (img_w // 4, img_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
        samples.append(EvalSample(
            sample_id=sample_id,
            image=img_gfx,
            view="crowd_graphic",
            H_gt=None,
            keypoints_px={},
            feet_points_px=None,
        ))

    dataset = EvalDataset(samples)
    if output_dir is not None:
        dataset.save(output_dir)

    return dataset
