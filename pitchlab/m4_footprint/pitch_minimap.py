"""2D pitch minimap renderer with camera footprint visualization.

Renders standard FIFA pitch in meters to a 2D image and overlays the
camera field-of-view (footprint) polygon with alpha transparency.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple
import cv2
import numpy as np

from pitchlab.core.pitch_template import PitchTemplate


class MinimapRenderer:
    """Renders 2D top-down football pitch with camera footprint."""

    def __init__(
        self,
        pitch: PitchTemplate,
        width_px: int = 440,
        height_px: int = 285,
        margin_px: int = 20,
    ) -> None:
        self.pitch = pitch
        self.w = width_px
        self.h = height_px
        self.margin = margin_px

        self.draw_w = self.w - 2 * self.margin
        self.draw_h = self.h - 2 * self.margin

        self.scale_x = self.draw_w / self.pitch.length
        self.scale_y = self.draw_h / self.pitch.width

        # Base background image with pitch markings
        self.base_pitch_img = self._render_base_pitch()

    def m2px(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Converts pitch meter coordinates (center=0,0) to minimap pixel (px, py)."""
        px = int(self.margin + (x_m - self.pitch.x_min) * self.scale_x)
        # Note: image Y is inverted relative to pitch Y (top is positive Y on pitch)
        py = int(self.margin + (self.pitch.y_max - y_m) * self.scale_y)
        return px, py

    def _render_base_pitch(self) -> np.ndarray:
        """Pre-renders the static football pitch with lawn stripes and FIFA lines."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Grass stripes
        num_stripes = 10
        stripe_w = self.draw_w / num_stripes
        c_green1 = (35, 120, 45)   # Dark grass BGR
        c_green2 = (42, 138, 52)   # Light grass BGR

        for i in range(num_stripes):
            x1 = int(self.margin + i * stripe_w)
            x2 = int(self.margin + (i + 1) * stripe_w)
            color = c_green1 if i % 2 == 0 else c_green2
            cv2.rectangle(img, (x1, self.margin), (x2, self.h - self.margin), color, -1)

        # White markings
        white = (245, 245, 245)
        thickness = 1

        # 1. Straight lines from pitch template
        for line in self.pitch.get_lines():
            p1 = self.m2px(*line.p1)
            p2 = self.m2px(*line.p2)
            cv2.line(img, p1, p2, white, thickness, cv2.LINE_AA)

        # 2. Center circle & spot
        center_px = self.m2px(0.0, 0.0)
        radius_px = int(self.pitch.center_circle_radius * self.scale_x)
        cv2.circle(img, center_px, radius_px, white, thickness, cv2.LINE_AA)
        cv2.circle(img, center_px, 2, white, -1, cv2.LINE_AA)

        # 3. Penalty spots
        kps = self.pitch.get_keypoints()
        p_spot_l = self.m2px(*kps["penalty_spot_left"])
        p_spot_r = self.m2px(*kps["penalty_spot_right"])
        cv2.circle(img, p_spot_l, 2, white, -1, cv2.LINE_AA)
        cv2.circle(img, p_spot_r, 2, white, -1, cv2.LINE_AA)

        # 4. Penalty arcs
        for arc in self.pitch.get_arcs():
            if arc.name.startswith("penalty_arc"):
                c_px = self.m2px(*arc.center)
                r_px = int(arc.radius * self.scale_x)
                # Angle conversion for inverted image Y-axis
                start_angle = -arc.angle_end_deg
                end_angle = -arc.angle_start_deg
                cv2.ellipse(img, c_px, (r_px, r_px), 0, start_angle, end_angle, white, thickness, cv2.LINE_AA)

        # 5. Goals
        goal_depth_px = int(2.0 * self.scale_x)
        g_l_top = self.m2px(-self.pitch.length / 2, self.pitch.goal_y_half)
        g_l_bot = self.m2px(-self.pitch.length / 2, -self.pitch.goal_y_half)
        cv2.rectangle(img, (g_l_top[0] - goal_depth_px, g_l_top[1]), (g_l_bot[0], g_l_bot[1]), white, 1)

        g_r_top = self.m2px(self.pitch.length / 2, self.pitch.goal_y_half)
        g_r_bot = self.m2px(self.pitch.length / 2, -self.pitch.goal_y_half)
        cv2.rectangle(img, (g_r_top[0], g_r_top[1]), (g_r_top[0] + goal_depth_px, g_r_bot[1]), white, 1)

        return img

    def draw_minimap(
        self,
        footprint_m: Optional[np.ndarray] = None,
        status: str = "ok",
    ) -> np.ndarray:
        """Draws pitch minimap with camera footprint overlay.
        
        Args:
            footprint_m: (N, 2) array of footprint polygon vertices in pitch meters.
            status: Calibration status ("ok", "tracked", "unreliable", "no_pitch").
            
        Returns:
            Rendered minimap as BGR image array (H, W, 3).
        """
        img = self.base_pitch_img.copy()

        if footprint_m is not None and len(footprint_m) >= 3:
            pts_px = np.array([self.m2px(x, y) for x, y in footprint_m], dtype=np.int32)

            # Color based on status
            if status == "ok":
                fill_color = (255, 230, 0)     # Bright cyan/yellow in BGR
                border_color = (255, 255, 255)
            elif status == "tracked":
                fill_color = (220, 160, 50)     # Blue/cyan
                border_color = (255, 200, 100)
            else:
                fill_color = (50, 50, 220)      # Red for unreliable
                border_color = (80, 80, 255)

            # Translucent fill using alpha blending
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts_px], fill_color)
            alpha = 0.35
            cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

            # Solid polygon boundary
            cv2.polylines(img, [pts_px], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

        # Draw outer minimap border and status badge
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), (50, 50, 50), 1)

        status_colors = {
            "ok": (40, 200, 40),
            "tracked": (220, 180, 40),
            "unreliable": (40, 40, 220),
            "no_pitch": (128, 128, 128),
        }
        badge_color = status_colors.get(status, (150, 150, 150))
        cv2.putText(
            img,
            f"STATUS: {status.upper()}",
            (10, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            badge_color,
            1,
            cv2.LINE_AA,
        )

        return img
