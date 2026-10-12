from __future__ import annotations

from typing import Sequence

import numpy as np

from pitchlab.core.config import KalmanBallConfig
from pitchlab.core.types import EntityClass, Track2D


class BallKalmanTracker:
    """Specialized 4-state Kalman Filter for single-ball tracking.

    State vector: [cx, cy, vx, vy]^T
    Measurement: [cx, cy]^T
    """

    def __init__(self, cfg: KalmanBallConfig, ball_id: int = 0) -> None:
        self.cfg = cfg
        self.ball_id = ball_id

        # 4D state: [cx, cy, vx, vy]
        self.x = np.zeros((4, 1), dtype=np.float32)
        # Covariance
        self.P = np.eye(4, dtype=np.float32) * 50.0

        dt = 1.0
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        self.Q = np.eye(4, dtype=np.float32) * float(self.cfg.process_noise)
        self.R = np.eye(2, dtype=np.float32) * float(self.cfg.measurement_noise)

        self.last_bbox_wh: tuple[float, float] = (15.0, 15.0)
        self.missing_frames: int = 0
        self.is_initialized: bool = False

    def predict(self) -> None:
        """Projects state and error covariance ahead."""
        if not self.is_initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, detections: Sequence[Track2D], frame_idx: int) -> Track2D | None:
        """Processes candidates, picks the closest to current prediction, and updates filter."""
        ball_dets = [d for d in detections if d.cls == EntityClass.BALL]

        self.predict()

        if not ball_dets:
            if not self.is_initialized:
                return None
            self.missing_frames += 1
            if self.missing_frames > self.cfg.max_missing_frames:
                self.is_initialized = False
                return None
            
            # Predict box based on state
            cx, cy = float(self.x[0, 0]), float(self.x[1, 0])
            w, h = self.last_bbox_wh
            pred_box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            return Track2D(
                track_id=self.ball_id,
                frame_idx=frame_idx,
                bbox_xyxy=pred_box,
                conf=0.2,
                cls=EntityClass.BALL,
            )

        # Select best detection (closest Euclidean distance to predicted cx, cy)
        best_det: Track2D | None = None
        best_dist = float("inf")

        for det in ball_dets:
            b = det.bbox_xyxy
            dcx = (b[0] + b[2]) / 2.0
            dcy = (b[1] + b[3]) / 2.0

            if not self.is_initialized:
                best_det = det
                break

            pred_cx, pred_cy = float(self.x[0, 0]), float(self.x[1, 0])
            dist = float(np.hypot(dcx - pred_cx, dcy - pred_cy))
            if dist < best_dist and dist <= self.cfg.max_dist_px:
                best_dist = dist
                best_det = det

        if best_det is None:
            # All detections were too far from predicted position
            if self.is_initialized:
                self.missing_frames += 1
                if self.missing_frames <= self.cfg.max_missing_frames:
                    cx, cy = float(self.x[0, 0]), float(self.x[1, 0])
                    w, h = self.last_bbox_wh
                    return Track2D(
                        track_id=self.ball_id,
                        frame_idx=frame_idx,
                        bbox_xyxy=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                        conf=0.2,
                        cls=EntityClass.BALL,
                    )
                self.is_initialized = False
            return None

        # Measurement update
        b = best_det.bbox_xyxy
        meas_cx = (b[0] + b[2]) / 2.0
        meas_cy = (b[1] + b[3]) / 2.0
        meas_w = b[2] - b[0]
        meas_h = b[3] - b[1]
        self.last_bbox_wh = (meas_w, meas_h)

        z = np.array([[meas_cx], [meas_cy]], dtype=np.float32)

        if not self.is_initialized:
            self.x = np.array([[meas_cx], [meas_cy], [0.0], [0.0]], dtype=np.float32)
            self.P = np.eye(4, dtype=np.float32) * 10.0
            self.is_initialized = True
            self.missing_frames = 0
        else:
            y_res = z - (self.H @ self.x)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + (K @ y_res)
            self.P = (np.eye(4, dtype=np.float32) - (K @ self.H)) @ self.P
            self.missing_frames = 0

        # Output smooth filtered box
        corr_cx, corr_cy = float(self.x[0, 0]), float(self.x[1, 0])
        corr_box = (
            corr_cx - meas_w / 2,
            corr_cy - meas_h / 2,
            corr_cx + meas_w / 2,
            corr_cy + meas_h / 2,
        )

        return Track2D(
            track_id=self.ball_id,
            frame_idx=frame_idx,
            bbox_xyxy=corr_box,
            conf=best_det.conf,
            cls=EntityClass.BALL,
        )
