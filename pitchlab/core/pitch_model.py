#!/usr/bin/env python3
"""Unified pitch geometry model for FIFA 105m x 68m soccer field.

Coordinate system:
  X: along the field [0.0 .. 105.0] meters (left to right)
  Y: across the field [0.0 .. 68.0] meters (top to bottom)
  Z: height above pitch [meters] (Z=0 is ground level)
  Origin (0, 0, 0): top-left corner of the pitch.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, hypot, pi, sin
from typing import Dict, List, Tuple

import numpy as np

PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0
CENTER_X: float = 52.5
CENTER_Y: float = 34.0
CENTER_CIRCLE_RADIUS: float = 9.15
GOAL_HEIGHT: float = 2.44

# Goal area (5.5m)
GOAL_BOX_DEPTH: float = 5.5
GOAL_BOX_Y_TOP: float = 24.84
GOAL_BOX_Y_BOTTOM: float = 43.16

# Penalty area (16.5m)
PENALTY_AREA_DEPTH: float = 16.5
PENALTY_AREA_Y_TOP: float = 13.84
PENALTY_AREA_Y_BOTTOM: float = 54.16

# Penalty marks (11.0m from goal line)
LEFT_PENALTY_POINT: Tuple[float, float] = (11.0, CENTER_Y)
RIGHT_PENALTY_POINT: Tuple[float, float] = (94.0, CENTER_Y)

# Penalty arcs (radius 9.15m, only visible outside the penalty box)
PENALTY_ARC_RADIUS: float = 9.15
# Angle where arc intersects penalty line x=16.5m (distance 5.5m from 11m mark): cos(theta) = 5.5 / 9.15
ANGLE_BOUND: float = acos(5.5 / PENALTY_ARC_RADIUS)

# Corner arcs
CORNER_ARC_RADIUS: float = 1.0


@dataclass(frozen=True)
class LineSegment2D:
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass(frozen=True)
class LineSegment3D:
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]


@dataclass(frozen=True)
class CircleArc2D:
    center: Tuple[float, float]
    radius: float
    start_angle: float  # radians
    end_angle: float    # radians


@dataclass(frozen=True)
class PitchKeypoint:
    name: str
    position: Tuple[float, ...]  # 2D (x, y) or 3D (x, y, z)


class PitchModel:
    """Canonical mathematical representation of a standard 105m x 68m FIFA soccer pitch."""

    def __init__(self) -> None:
        self.length = PITCH_LENGTH
        self.width = PITCH_WIDTH

    def get_keypoints(self) -> Dict[str, Tuple[float, float]]:
        """Named 2D landmark coordinates on the ground plane (Z=0)."""
        return {
            "TOP_LEFT_CORNER": (0.0, 0.0),
            "TOP_RIGHT_CORNER": (PITCH_LENGTH, 0.0),
            "BOTTOM_LEFT_CORNER": (0.0, PITCH_WIDTH),
            "BOTTOM_RIGHT_CORNER": (PITCH_LENGTH, PITCH_WIDTH),
            "CENTER_MARK": (CENTER_X, CENTER_Y),
            "MIDFIELD_TOP": (CENTER_X, 0.0),
            "MIDFIELD_BOTTOM": (CENTER_X, PITCH_WIDTH),
            "L_GOAL_POST_TOP": (0.0, 30.34),
            "L_GOAL_POST_BOTTOM": (0.0, 37.66),
            "R_GOAL_POST_TOP": (PITCH_LENGTH, 30.34),
            "R_GOAL_POST_BOTTOM": (PITCH_LENGTH, 37.66),
            "L_PENALTY_MARK": LEFT_PENALTY_POINT,
            "R_PENALTY_MARK": RIGHT_PENALTY_POINT,
            "L_GOAL_BOX_TL": (0.0, GOAL_BOX_Y_TOP),
            "L_GOAL_BOX_BL": (0.0, GOAL_BOX_Y_BOTTOM),
            "L_GOAL_BOX_TR": (GOAL_BOX_DEPTH, GOAL_BOX_Y_TOP),
            "L_GOAL_BOX_BR": (GOAL_BOX_DEPTH, GOAL_BOX_Y_BOTTOM),
            "R_GOAL_BOX_TL": (PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_TOP),
            "R_GOAL_BOX_BL": (PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_BOTTOM),
            "R_GOAL_BOX_TR": (PITCH_LENGTH, GOAL_BOX_Y_TOP),
            "R_GOAL_BOX_BR": (PITCH_LENGTH, GOAL_BOX_Y_BOTTOM),
            "L_PENALTY_BOX_TL": (0.0, PENALTY_AREA_Y_TOP),
            "L_PENALTY_BOX_BL": (0.0, PENALTY_AREA_Y_BOTTOM),
            "L_PENALTY_BOX_TR": (PENALTY_AREA_DEPTH, PENALTY_AREA_Y_TOP),
            "L_PENALTY_BOX_BR": (PENALTY_AREA_DEPTH, PENALTY_AREA_Y_BOTTOM),
            "R_PENALTY_BOX_TL": (PITCH_LENGTH - PENALTY_AREA_DEPTH, PENALTY_AREA_Y_TOP),
            "R_PENALTY_BOX_BL": (PITCH_LENGTH - PENALTY_AREA_DEPTH, PENALTY_AREA_Y_BOTTOM),
            "R_PENALTY_BOX_TR": (PITCH_LENGTH, PENALTY_AREA_Y_TOP),
            "R_PENALTY_BOX_BR": (PITCH_LENGTH, PENALTY_AREA_Y_BOTTOM),
        }

    def get_keypoints_3d(self) -> Dict[str, Tuple[float, float, float]]:
        """3D landmark coordinates including goal posts and crossbar."""
        keys2d = self.get_keypoints()
        keypoints_3d: Dict[str, Tuple[float, float, float]] = {
            name: (x, y, 0.0) for name, (x, y) in keys2d.items()
        }
        # Goal tops
        keypoints_3d["L_CROSSBAR_TOP"] = (0.0, 30.34, GOAL_HEIGHT)
        keypoints_3d["L_CROSSBAR_BOTTOM"] = (0.0, 37.66, GOAL_HEIGHT)
        keypoints_3d["R_CROSSBAR_TOP"] = (PITCH_LENGTH, 30.34, GOAL_HEIGHT)
        keypoints_3d["R_CROSSBAR_BOTTOM"] = (PITCH_LENGTH, 37.66, GOAL_HEIGHT)
        return keypoints_3d

    def get_line_segments(self) -> List[LineSegment2D]:
        """All straight pitch line segments."""
        segments: List[LineSegment2D] = [
            # Outer touchlines and goal lines
            LineSegment2D((0.0, 0.0), (PITCH_LENGTH, 0.0)),
            LineSegment2D((PITCH_LENGTH, 0.0), (PITCH_LENGTH, PITCH_WIDTH)),
            LineSegment2D((PITCH_LENGTH, PITCH_WIDTH), (0.0, PITCH_WIDTH)),
            LineSegment2D((0.0, PITCH_WIDTH), (0.0, 0.0)),
            # Center line
            LineSegment2D((CENTER_X, 0.0), (CENTER_X, PITCH_WIDTH)),
            # Left 5.5m goal box
            LineSegment2D((0.0, GOAL_BOX_Y_TOP), (GOAL_BOX_DEPTH, GOAL_BOX_Y_TOP)),
            LineSegment2D((GOAL_BOX_DEPTH, GOAL_BOX_Y_TOP), (GOAL_BOX_DEPTH, GOAL_BOX_Y_BOTTOM)),
            LineSegment2D((GOAL_BOX_DEPTH, GOAL_BOX_Y_BOTTOM), (0.0, GOAL_BOX_Y_BOTTOM)),
            # Right 5.5m goal box
            LineSegment2D((PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_TOP), (PITCH_LENGTH, GOAL_BOX_Y_TOP)),
            LineSegment2D(
                (PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_TOP),
                (PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_BOTTOM)
            ),
            LineSegment2D((PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_BOTTOM), (PITCH_LENGTH, GOAL_BOX_Y_BOTTOM)),
            # Left 16.5m penalty box
            LineSegment2D((0.0, PENALTY_AREA_Y_TOP), (PENALTY_AREA_DEPTH, PENALTY_AREA_Y_TOP)),
            LineSegment2D((PENALTY_AREA_DEPTH, PENALTY_AREA_Y_TOP), (PENALTY_AREA_DEPTH, PENALTY_AREA_Y_BOTTOM)),
            LineSegment2D((PENALTY_AREA_DEPTH, PENALTY_AREA_Y_BOTTOM), (0.0, PENALTY_AREA_Y_BOTTOM)),
            # Right 16.5m penalty box
            LineSegment2D((PITCH_LENGTH - PENALTY_AREA_DEPTH, PENALTY_AREA_Y_TOP), (PITCH_LENGTH, PENALTY_AREA_Y_TOP)),
            LineSegment2D(
                (PITCH_LENGTH - PENALTY_AREA_DEPTH, PENALTY_AREA_Y_TOP),
                (PITCH_LENGTH - PENALTY_AREA_DEPTH, PENALTY_AREA_Y_BOTTOM)
            ),
            LineSegment2D(
                (PITCH_LENGTH - PENALTY_AREA_DEPTH, PENALTY_AREA_Y_BOTTOM),
                (PITCH_LENGTH, PENALTY_AREA_Y_BOTTOM)
            ),
        ]
        return segments

    def get_arcs(self) -> List[CircleArc2D]:
        """All curved pitch lines (center circle, penalty box arcs, corner arcs)."""
        arcs: List[CircleArc2D] = [
            # Center circle
            CircleArc2D((CENTER_X, CENTER_Y), CENTER_CIRCLE_RADIUS, 0.0, 2 * pi),
            # Left penalty arc (only outside 16.5m line: x > 16.5)
            CircleArc2D(LEFT_PENALTY_POINT, PENALTY_ARC_RADIUS, -ANGLE_BOUND, ANGLE_BOUND),
            # Right penalty arc (only outside 16.5m line: x < 88.5)
            CircleArc2D(RIGHT_PENALTY_POINT, PENALTY_ARC_RADIUS, pi - ANGLE_BOUND, pi + ANGLE_BOUND),
            # Corner arcs (1m radius)
            CircleArc2D((0.0, 0.0), CORNER_ARC_RADIUS, 0.0, pi / 2),
            CircleArc2D((PITCH_LENGTH, 0.0), CORNER_ARC_RADIUS, pi / 2, pi),
            CircleArc2D((0.0, PITCH_WIDTH), CORNER_ARC_RADIUS, -pi / 2, 0.0),
            CircleArc2D((PITCH_LENGTH, PITCH_WIDTH), CORNER_ARC_RADIUS, pi, 3 * pi / 2),
        ]
        return arcs

    def get_goal_posts_3d(self) -> List[LineSegment3D]:
        """3D segments for goal posts and crossbars."""
        return [
            # Left goal posts and crossbar
            LineSegment3D((0.0, 30.34, 0.0), (0.0, 30.34, GOAL_HEIGHT)),
            LineSegment3D((0.0, 37.66, 0.0), (0.0, 37.66, GOAL_HEIGHT)),
            LineSegment3D((0.0, 30.34, GOAL_HEIGHT), (0.0, 37.66, GOAL_HEIGHT)),
            # Right goal posts and crossbar
            LineSegment3D((PITCH_LENGTH, 30.34, 0.0), (PITCH_LENGTH, 30.34, GOAL_HEIGHT)),
            LineSegment3D((PITCH_LENGTH, 37.66, 0.0), (PITCH_LENGTH, 37.66, GOAL_HEIGHT)),
            LineSegment3D((PITCH_LENGTH, 30.34, GOAL_HEIGHT), (PITCH_LENGTH, 37.66, GOAL_HEIGHT)),
        ]

    def sample_line_points(self, density: float = 0.5) -> np.ndarray:
        """Sample dense points along all lines and arcs for Chamfer distance and edge matching."""
        points: List[Tuple[float, float]] = []
        for seg in self.get_line_segments():
            x1, y1 = seg.start
            x2, y2 = seg.end
            seg_len = hypot(x2 - x1, y2 - y1)
            num_samples = max(int(seg_len / density) + 1, 2)
            for i in range(num_samples):
                t = i / (num_samples - 1)
                points.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))

        for arc in self.get_arcs():
            diff = arc.end_angle - arc.start_angle
            if diff < 0:
                diff += 2 * pi
            arc_len = abs(diff * arc.radius)
            num_samples = max(int(arc_len / density) + 1, 2)
            for i in range(num_samples):
                theta = arc.start_angle + (i / (num_samples - 1)) * diff
                points.append((arc.center[0] + arc.radius * cos(theta), arc.center[1] + arc.radius * sin(theta)))

        return np.asarray(points, dtype=np.float64)

    def is_inside_pitch(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Returns True if point (x, y) is within pitch boundaries with optional tolerance."""
        return (-margin <= x <= PITCH_LENGTH + margin) and (-margin <= y <= PITCH_WIDTH + margin)

    def project_ground_points(self, points_2d: np.ndarray, H_pitch2img: np.ndarray) -> np.ndarray:
        """Project (N, 2) ground points in meters to (N, 2) image pixel coordinates using H_pitch2img."""
        if points_2d.ndim != 2 or points_2d.shape[1] != 2:
            raise ValueError("points_2d must be of shape (N, 2)")
        n = points_2d.shape[0]
        homog = np.hstack([points_2d, np.ones((n, 1), dtype=np.float64)])  # (N, 3)
        proj = homog @ H_pitch2img.T  # (N, 3)
        denom = proj[:, 2:3]
        denom[np.abs(denom) < 1e-6] = 1e-6
        return proj[:, :2] / denom
