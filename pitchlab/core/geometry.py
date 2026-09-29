# pitchlab/core/geometry.py
"""
Geometry utilities for PitchLab.

This module provides:
* ``PitchTemplate`` – defines the standard soccer pitch dimensions (FIFA) and
  computes key points (corners, centre, penalty spots, etc.).
* ``compute_homography`` – given a set of image points (pixel coordinates) and
  the corresponding world points (meters) it returns the 3×3 homography matrix
  mapping image → world coordinates.  The implementation uses ``numpy`` and
  ``cv2`` if available, falling back to a pure‑numpy DLT algorithm.
* ``apply_homography`` – transforms pixel points to world coordinates using the
  homography.
* ``project_points`` – projects points using a homography matrix.
* ``compute_footprint`` – computes the visible field area on the pitch using homography.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, List, Any
from shapely.geometry import Polygon, box

from .types import Meters, Point2D

# ---------------------------------------------------------------------------
# Pitch template
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PitchTemplate:
    """Standard pitch layout.

    Attributes
    ----------
    length_m: Meters
        Length of the pitch (touch line) in meters.
    width_m: Meters
        Width of the pitch (goal line) in meters.
    """

    length_m: Meters = Meters(105.0)
    width_m: Meters = Meters(68.0)

    # Helper – coordinates of the four corners (clockwise starting
    # from bottom‑left when looking from the attacking side).
    @property
    def corners(self) -> List[Point2D]:
        l, w = float(self.length_m), float(self.width_m)
        return [
            Point2D(0.0, 0.0),
            Point2D(l, 0.0),
            Point2D(l, w),
            Point2D(0.0, w),
        ]

    @property
    def centre(self) -> Point2D:
        return Point2D(float(self.length_m) / 2.0, float(self.width_m) / 2.0)

    @property
    def penalty_spots(self) -> Tuple[Point2D, Point2D]:
        """Return (home, away) penalty spot positions.

        Distance from the goal line is 11.0 m (standard). The X‑coordinate is
        measured from the left goal line.
        """
        offset = 11.0
        return (
            Point2D(offset, self.width_m / 2.0),
            Point2D(self.length_m - offset, self.width_m / 2.0),
        )

# ---------------------------------------------------------------------------
# Homography utilities
# ---------------------------------------------------------------------------

def _dltsolve(A: np.ndarray) -> np.ndarray:
    """Direct Linear Transform (DLT) solver for homography."""
    _, _, vh = np.linalg.svd(A)
    h = vh[-1]
    H = h.reshape(3, 3)
    return H / H[2, 2]


def compute_homography(
    image_pts: Sequence[Tuple[float, float]],
    world_pts: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Compute homography matrix mapping *image* → *world* coordinates."""
    if len(image_pts) != len(world_pts):
        raise ValueError("image_pts and world_pts must have the same length")
    if len(image_pts) < 4:
        raise ValueError("At least four point correspondences are required")

    rows = []
    for (x_img, y_img), (x_wld, y_wld) in zip(image_pts, world_pts):
        rows.append([0, 0, 0, -x_img, -y_img, -1, y_wld * x_img, y_wld * y_img, y_wld])
        rows.append([x_img, y_img, 1, 0, 0, 0, -x_wld * x_img, -x_wld * y_img, -x_wld])
    A = np.array(rows, dtype=float)
    return _dltsolve(A)


def apply_homography(H: np.ndarray, pts: Sequence[Tuple[float, float]]) -> List[Point2D]:
    """Apply homography ``H`` to a list of pixel points."""
    pts_h = np.column_stack([np.array(pts, dtype=float), np.ones(len(pts))])
    world_h = (H @ pts_h.T).T
    world = world_h[:, :2] / world_h[:, 2:3]
    return [Point2D(float(x), float(y)) for x, y in world]


# --- DODANA BRAKUJĄCA FUNKCJA PROJECT_POINTS ---
def project_points(points: np.ndarray | Sequence[Tuple[float, float]], H: np.ndarray) -> np.ndarray:
    """Project points using the homography matrix H.
    Compatible with calls expecting project_points(points, H) or similar signatures.
    """
    pts = np.array(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2), dtype=float)
        
    # Handle both (N, 2) and flattened formats gracefully
    original_shape = pts.shape
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
        
    pts_h = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    projected_h = (H @ pts_h.T).T
    
    # Avoid division by zero
    z = projected_h[:, 2:3]
    z[z == 0] = 1e-9
    
    projected = projected_h[:, :2] / z
    return projected


def compute_footprint(H: np.ndarray, image_width: int = 1920, image_height: int = 1080) -> Polygon:
    """Computes the 2D polygon representation of the camera's field of view 
    on the pitch plane based on the homography matrix H.
    """
    # Define the 4 corners of the image frame in pixels
    corners_img = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]
    
    # Project these image corners into pitch world coordinates (meters)
    world_points = apply_homography(H, corners_img)
    polygon_points = [(p.x, p.y) for p in world_points]
    
    # Return a Shapely Polygon representation of the footprint
    return Polygon(polygon_points)


# ---------------------------------------------------------------------------
# Validation utilities (simple sanity checks)
# ---------------------------------------------------------------------------

def validate_homography(H: np.ndarray) -> bool:
    """Return ``True`` if ``H`` appears to be a valid homography."""
    if H.shape != (3, 3):
        return False
    if not np.isfinite(H).all():
        return False
    return abs(np.linalg.det(H)) > 1e-9

"""End of geometry utilities."""