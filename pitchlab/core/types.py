# pitchlab/core/types.py
"""
Data type definitions for the PitchLab core package.

These types are deliberately minimal and use standard Python typing with
`dataclasses` for immutability where appropriate. Validation is performed
through `pydantic` for any external user‑supplied data to guard against
malformed payloads (e.g., JSON from an API or GCS object).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, Tuple, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

# Basic numeric new‑type aliases for clarity
Meters = NewType("Meters", float)
Pixels = NewType("Pixels", float)

# Simple 2‑D coordinate representations
@dataclass(frozen=True, slots=True)
class Point2D:
    """A point in 2‑D space.

    The class is immutable (`frozen=True`) which makes it hashable and safe
    to share across threads without additional locking.  The `slots`
    optimisation reduces memory overhead for large collections of points.
    """

    x: float
    y: float

    def __iter__(self):
        return iter((self.x, self.y))

    def as_tuple(self) -> Tuple[float, float]:
        """Return the point as a plain ``(x, y)`` tuple."""
        return (self.x, self.y)

# Domain‑specific aliases for readability
Position = Point2D  # Player or ball position on the pitch (meters)
Velocity = Point2D  # Velocity vector (m/s)
Acceleration = Point2D  # Acceleration vector (m/s²)

# Identifiers – using simple integer new‑type aliases to prevent accidental mixing
PlayerID = NewType("PlayerID", int)
BallID = NewType("BallID", int)

# --- STATUS KALIBRACJI (Wymagany przez m5_eval) ---
class CalibStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING = "pending"

# --- STRUKTURA KALIBRACJI ---

class FrameCalib(BaseModel):
    """Represents camera calibration data for a specific video frame.
    
    Uses Pydantic for robust data validation and safe integration with 
    FastAPI endpoints and JSON schemas.
    """
    frame_idx: int = Field(default=0, description="The index of the video frame.")
    homography_matrix: Optional[List[List[float]]] = Field(
        default=None, 
        description="3x3 transformation matrix mapping image coordinates to pitch coordinates."
    )
    camera_position: Optional[List[float]] = Field(default=None, description="Estimated 3D position of the camera.")
    focal_length: Optional[float] = Field(default=None, description="Focal length of the camera lens.")
    distortion_coefficients: List[float] = Field(default_factory=list, description="Lens distortion parameters.")

    class Config:
        from_attributes = True

# --- TYPY WIDOKÓW (Wymagane przez moduł ewaluacji / m5_eval) ---
class ViewType(str, Enum):
    TACTICAL = "tactical"
    BROADCAST = "broadcast"
    TACTICAL_CAROUSEL = "tactical_carousel"
    CUSTOM = "custom"

# --- STAŁA SCHEMATU IMPORTOWANA PRZEZ PIPELINE / SERVER ---
SCHEMA_VERSION = "1.0.0"