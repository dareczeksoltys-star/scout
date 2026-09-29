"""Control video renderer with 2D minimap and projected pitch markings overlay.

Follows Section 7 (M4) of docs/PLAN.md:
Wizualizacja: minimapa 2D z zaznaczonym kadrem + overlay siatki metrów na wideo. Eksport wideo kontrolnego.
Akceptacja: wideo kontrolne dla każdego skrótu; JSON FrameCalib dla każdej klatki.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from pitchlab.core.config import AppConfig
from pitchlab.core.geometry import project_points
from pitchlab.core.io import VideoReader, resolve_local_cache, save_json
from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.core.types import FrameCalib, SCHEMA_VERSION
from pitchlab.m4_footprint.pitch_minimap import MinimapRenderer


def draw_projected_pitch_lines(
    frame: np.ndarray,
    H_img2pitch: np.ndarray,
    pitch: PitchTemplate,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    num_samples: int = 30,
) -> np.ndarray:
    """Projects 3D/2D pitch lines from meters back onto the image using H_inv."""
    try:
        H_pitch2img = np.linalg.inv(H_img2pitch)
    except np.linalg.LinAlgError:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()

    for line in pitch.get_lines():
        # Sample points along line segment in pitch meters
        p1, p2 = line.p1, line.p2
        xs = np.linspace(p1[0], p2[0], num_samples)
        ys = np.linspace(p1[1], p2[1], num_samples)
        pts_pitch = np.column_stack([xs, ys])

        # Project to image pixels
        pts_img, valid = project_points(H_pitch2img, pts_pitch)

        # Filter points within or near image frame
        inside_img = (
            valid &
            (pts_img[:, 0] >= -50) & (pts_img[:, 0] <= w + 50) &
            (pts_img[:, 1] >= -50) & (pts_img[:, 1] <= h + 50)
        )

        valid_segments: List[List[Tuple[int, int]]] = []
        current_segment: List[Tuple[int, int]] = []

        for i in range(len(pts_img)):
            if inside_img[i]:
                current_segment.append((int(pts_img[i, 0]), int(pts_img[i, 1])))
            else:
                if len(current_segment) >= 2:
                    valid_segments.append(current_segment)
                current_segment = []
        if len(current_segment) >= 2:
            valid_segments.append(current_segment)

        for seg in valid_segments:
            pts_arr = np.array(seg, dtype=np.int32)
            cv2.polylines(overlay, [pts_arr], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Subtle alpha blend so original lines remain visible
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    return frame


def render_control_video(
    video_path: Union[str, Path],
    calib_results: List[FrameCalib],
    output_video_path: Union[str, Path],
    output_json_path: Optional[Union[str, Path]] = None,
    pitch: Optional[PitchTemplate] = None,
    config: Optional[AppConfig] = None,
    pip_scale: float = 0.30,
) -> Path:
    """Renders control video with 2D minimap picture-in-picture and pitch grid overlay.
    
    Exports:
    1. Video file with minimap and overlay.
    2. Companion JSON with FrameCalib records for every processed frame.
    """
    cfg = config or AppConfig.load()
    pitch_tmpl = pitch or PitchTemplate(length_m=cfg.pitch.length_m, width_m=cfg.pitch.width_m)
    local_in_video = resolve_local_cache(video_path)
    out_video_p = Path(output_video_path)
    out_video_p.parent.mkdir(parents=True, exist_ok=True)

    # Index calibrations by frame_idx
    calib_map: Dict[int, FrameCalib] = {c.frame_idx: c for c in calib_results}

    cap = cv2.VideoCapture(str(local_in_video))
    if not cap.isOpened():
        raise IOError(f"Could not open video {local_in_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Minimap dimensions
    m_w = int(w * pip_scale)
    m_h = int(m_w * (pitch_tmpl.width / pitch_tmpl.length) * 1.05)
    minimap_renderer = MinimapRenderer(pitch_tmpl, width_px=m_w, height_px=m_h, margin_px=10)

    # Setup video writer (mp4v or avc1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_p), fourcc, fps, (w, h))

    current_calib: Optional[FrameCalib] = None
    json_records: List[dict] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in calib_map:
                current_calib = calib_map[frame_idx]

            status = current_calib.status if current_calib else "no_pitch"
            conf = current_calib.conf if current_calib else 0.0
            footprint = current_calib.footprint_m if current_calib else None

            # 1. Overlay projected pitch markings if status is ok or tracked
            if current_calib and current_calib.H_img2pitch is not None and status in ("ok", "tracked"):
                draw_projected_pitch_lines(frame, current_calib.H_img2pitch, pitch_tmpl)

            # 2. Render 2D Minimap
            minimap = minimap_renderer.draw_minimap(footprint_m=footprint, status=status)

            # 3. Embed Minimap Picture-in-Picture (Bottom-Right corner with margin)
            pos_x = w - m_w - 20
            pos_y = h - m_h - 20
            frame[pos_y:pos_y + m_h, pos_x:pos_x + m_w] = minimap

            # 4. Draw Telemetry HUD
            t_sec = frame_idx / fps
            hud_text = f"F: {frame_idx:05d} | T: {t_sec:05.2f}s | STATUS: {status.upper()} | CONF: {conf:.2f}"
            cv2.putText(frame, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

            writer.write(frame)

            # Store JSON record
            if current_calib:
                json_records.append({
                    "frame_idx": frame_idx,
                    "t": t_sec,
                    "status": status,
                    "conf": conf,
                    "source": current_calib.source,
                    "footprint_m": footprint.tolist() if footprint is not None else None,
                })

            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    # Save JSON calibration export
    if output_json_path is not None:
        save_json({
            "schema_version": SCHEMA_VERSION,
            "total_frames": total_frames,
            "fps": fps,
            "pitch_dims": {"length": pitch_tmpl.length, "width": pitch_tmpl.width},
            "frames": json_records,
        }, output_json_path)

    return out_video_p
