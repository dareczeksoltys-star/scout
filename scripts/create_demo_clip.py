"""Generates a synthetic broadcast video clip for testing and demonstration."""
from pathlib import Path
import cv2
import numpy as np

from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.core.geometry import project_points


def create_demo_broadcast_video(output_path: str = "data/raw/demo_highlight.mp4", duration_sec: float = 3.0, fps: float = 25.0):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    w, h = 1280, 720
    total_frames = int(duration_sec * fps)

    pitch = PitchTemplate(105.0, 68.0)
    lines = pitch.get_lines()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_idx in range(total_frames):
        # Base pitch color
        frame = np.full((h, w, 3), (40, 130, 48), dtype=np.uint8)

        # Camera slowly panning from left to center
        pan_x = -30.0 + (frame_idx / total_frames) * 25.0

        # Construct realistic camera perspective matrix
        # Pitch meters -> Image pixels
        scale = 14.0
        # (x, y) pitch -> (u, v) image
        # u = (x - pan_x) * scale + w/2
        # v = (y - 5.0) * (-scale) + h/2 + 100
        H_pitch2img = np.array([
            [scale, 0.0, -pan_x * scale + w / 2],
            [0.0, -scale, 5.0 * scale + h / 2 + 80],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Draw pitch markings on the frame
        for line in lines:
            xs = np.linspace(line.p1[0], line.p2[0], 20)
            ys = np.linspace(line.p1[1], line.p2[1], 20)
            pts_pitch = np.column_stack([xs, ys])
            pts_img, valid = project_points(H_pitch2img, pts_pitch)
            valid_pts = pts_img[valid].astype(np.int32)
            if len(valid_pts) >= 2:
                cv2.polylines(frame, [valid_pts], isClosed=False, color=(240, 240, 240), thickness=2, lineType=cv2.LINE_AA)

        # Center circle
        circle_angles = np.linspace(0, 2 * np.pi, 60)
        c_x = pitch.center_circle_radius * np.cos(circle_angles)
        c_y = pitch.center_circle_radius * np.sin(circle_angles)
        c_pitch = np.column_stack([c_x, c_y])
        c_img, valid = project_points(H_pitch2img, c_pitch)
        if np.any(valid):
            cv2.polylines(frame, [c_img[valid].astype(np.int32)], isClosed=True, color=(240, 240, 240), thickness=2, lineType=cv2.LINE_AA)

        # Add simulated score bug
        cv2.rectangle(frame, (40, 30), (280, 80), (20, 20, 20), -1)
        cv2.putText(frame, "PITCHLAB TV  1 - 0", (55, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"Created demo clip: {output_path} ({total_frames} frames, {w}x{h})")


if __name__ == "__main__":
    create_demo_broadcast_video()
