"""FIFA standard pitch template in meters.

Follows coordinate convention from Section 6 of docs/PLAN.md:
Origin (0,0) at center mark.
X-axis: along length [-length/2, length/2] (left goal line to right goal line).
Y-axis: along width [-width/2, width/2] (bottom touchline to top touchline).
Units: meters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import Polygon, box


@dataclass(frozen=True)
class LineSegment:
    name: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]


@dataclass(frozen=True)
class ArcDefinition:
    name: str
    center: Tuple[float, float]
    radius: float
    angle_start_deg: float
    angle_end_deg: float


class PitchTemplate:
    """Standard FIFA pitch geometry with variable length and width.
    
    Fixed FIFA standard dimensions:
    - Penalty area: extends 16.5m from goal line, width 40.32m (16.5 + 7.32 + 16.5)
    - Goal area: extends 5.5m from goal line, width 18.32m (5.5 + 7.32 + 5.5)
    - Penalty spot: 11.0m from goal line
    - Center circle radius: 9.15m
    - Penalty arc radius: 9.15m from penalty mark (outside penalty box)
    - Corner arc radius: 1.0m
    - Goal width: 7.32m
    """

    def __init__(self, length_m: float = 105.0, width_m: float = 68.0) -> None:
        if length_m <= 0 or width_m <= 0:
            raise ValueError(f"Pitch dimensions must be positive, got {length_m}x{width_m}")
        self.length = float(length_m)
        self.width = float(width_m)

        self.x_min = -self.length / 2.0
        self.x_max = self.length / 2.0
        self.y_min = -self.width / 2.0
        self.y_max = self.width / 2.0

        # Fixed FIFA metrics
        self.center_circle_radius = 9.15
        self.penalty_spot_dist = 11.0
        self.penalty_arc_radius = 9.15
        self.corner_arc_radius = 1.0

        self.penalty_box_length = 16.5
        self.penalty_box_width = 40.32      # 16.5 * 2 + 7.32
        self.penalty_box_y_half = self.penalty_box_width / 2.0  # 20.16

        self.goal_box_length = 5.5
        self.goal_box_width = 18.32        # 5.5 * 2 + 7.32
        self.goal_box_y_half = self.goal_box_width / 2.0        # 9.16

        self.goal_width = 7.32
        self.goal_y_half = self.goal_width / 2.0                # 3.66

    @property
    def shapely_polygon(self) -> Polygon:
        """Returns the outer pitch boundary as a Shapely Polygon."""
        return box(self.x_min, self.y_min, self.x_max, self.y_max)

    def is_inside(self, x: float, y: float, margin_m: float = 0.0) -> bool:
        """Checks if a point is within pitch boundaries plus an optional margin."""
        return (
            (self.x_min - margin_m <= x <= self.x_max + margin_m) and
            (self.y_min - margin_m <= y <= self.y_max + margin_m)
        )

    def get_keypoints(self) -> Dict[str, Tuple[float, float]]:
        """Returns semantic landmark keypoints on the pitch in meters."""
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max
        p_len = self.penalty_box_length
        p_y = self.penalty_box_y_half
        g_len = self.goal_box_length
        g_y = self.goal_box_y_half

        kps: Dict[str, Tuple[float, float]] = {
            # Corners
            "corner_bottom_left": (x_min, y_min),
            "corner_top_left": (x_min, y_max),
            "corner_bottom_right": (x_max, y_min),
            "corner_top_right": (x_max, y_max),

            # Halfway line intersections
            "halfway_bottom": (0.0, y_min),
            "halfway_top": (0.0, y_max),
            "center_mark": (0.0, 0.0),

            # Center circle intersections with lines
            "center_circle_left": (-self.center_circle_radius, 0.0),
            "center_circle_right": (self.center_circle_radius, 0.0),
            "center_circle_bottom": (0.0, -self.center_circle_radius),
            "center_circle_top": (0.0, self.center_circle_radius),

            # Penalty spots
            "penalty_spot_left": (x_min + self.penalty_spot_dist, 0.0),
            "penalty_spot_right": (x_max - self.penalty_spot_dist, 0.0),

            # Left Penalty Area
            "penalty_area_left_tl": (x_min, p_y),
            "penalty_area_left_bl": (x_min, -p_y),
            "penalty_area_left_tr": (x_min + p_len, p_y),
            "penalty_area_left_br": (x_min + p_len, -p_y),

            # Right Penalty Area
            "penalty_area_right_tr": (x_max, p_y),
            "penalty_area_right_br": (x_max, -p_y),
            "penalty_area_right_tl": (x_max - p_len, p_y),
            "penalty_area_right_bl": (x_max - p_len, -p_y),

            # Left Goal Area
            "goal_area_left_tl": (x_min, g_y),
            "goal_area_left_bl": (x_min, -g_y),
            "goal_area_left_tr": (x_min + g_len, g_y),
            "goal_area_left_br": (x_min + g_len, -g_y),

            # Right Goal Area
            "goal_area_right_tr": (x_max, g_y),
            "goal_area_right_br": (x_max, -g_y),
            "goal_area_right_tl": (x_max - g_len, g_y),
            "goal_area_right_bl": (x_max - g_len, -g_y),

            # Goal Posts
            "goal_post_left_top": (x_min, self.goal_y_half),
            "goal_post_left_bottom": (x_min, -self.goal_y_half),
            "goal_post_right_top": (x_max, self.goal_y_half),
            "goal_post_right_bottom": (x_max, -self.goal_y_half),
        }
        return kps

    def get_lines(self) -> List[LineSegment]:
        """Returns all straight line segments of the pitch markings."""
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max
        p_len = self.penalty_box_length
        p_y = self.penalty_box_y_half
        g_len = self.goal_box_length
        g_y = self.goal_box_y_half

        lines: List[LineSegment] = [
            # Boundary lines
            LineSegment("touchline_bottom", (x_min, y_min), (x_max, y_min)),
            LineSegment("touchline_top", (x_min, y_max), (x_max, y_max)),
            LineSegment("goalline_left", (x_min, y_min), (x_min, y_max)),
            LineSegment("goalline_right", (x_max, y_min), (x_max, y_max)),

            # Halfway line
            LineSegment("halfway_line", (0.0, y_min), (0.0, y_max)),

            # Left Penalty Area
            LineSegment("penalty_area_left_top", (x_min, p_y), (x_min + p_len, p_y)),
            LineSegment("penalty_area_left_bottom", (x_min, -p_y), (x_min + p_len, -p_y)),
            LineSegment("penalty_area_left_front", (x_min + p_len, -p_y), (x_min + p_len, p_y)),

            # Right Penalty Area
            LineSegment("penalty_area_right_top", (x_max - p_len, p_y), (x_max, p_y)),
            LineSegment("penalty_area_right_bottom", (x_max - p_len, -p_y), (x_max, -p_y)),
            LineSegment("penalty_area_right_front", (x_max - p_len, -p_y), (x_max - p_len, p_y)),

            # Left Goal Area
            LineSegment("goal_area_left_top", (x_min, g_y), (x_min + g_len, g_y)),
            LineSegment("goal_area_left_bottom", (x_min, -g_y), (x_min + g_len, -g_y)),
            LineSegment("goal_area_left_front", (x_min + g_len, -g_y), (x_min + g_len, g_y)),

            # Right Goal Area
            LineSegment("goal_area_right_top", (x_max - g_len, g_y), (x_max, g_y)),
            LineSegment("goal_area_right_bottom", (x_max - g_len, -g_y), (x_max, -g_y)),
            LineSegment("goal_area_right_front", (x_max - g_len, -g_y), (x_max - g_len, g_y)),
        ]
        return lines

    def get_arcs(self) -> List[ArcDefinition]:
        """Returns circle and arc definitions (center, radius, angle range in degrees)."""
        x_min, x_max = self.x_min, self.x_max

        # Angle range for penalty arc outside the 16.5m penalty box
        # Penalty spot is 11m from goal line, box edge is 16.5m -> dx = 5.5m
        # Arc radius is 9.15m -> cos(theta) = 5.5 / 9.15
        dx = self.penalty_box_length - self.penalty_spot_dist  # 16.5 - 11.0 = 5.5
        theta_rad = math.acos(dx / self.penalty_arc_radius)
        theta_deg = math.degrees(theta_rad)

        arcs: List[ArcDefinition] = [
            # Full center circle
            ArcDefinition("center_circle", (0.0, 0.0), self.center_circle_radius, 0.0, 360.0),

            # Left penalty arc (opening to the right, outside penalty box)
            ArcDefinition("penalty_arc_left", (x_min + self.penalty_spot_dist, 0.0),
                          self.penalty_arc_radius, -theta_deg, theta_deg),

            # Right penalty arc (opening to the left, outside penalty box)
            ArcDefinition("penalty_arc_right", (x_max - self.penalty_spot_dist, 0.0),
                          self.penalty_arc_radius, 180.0 - theta_deg, 180.0 + theta_deg),
        ]
        return arcs
