"""Shared domain types for the PitchLab pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from math import hypot
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

FrameIndex: TypeAlias = int
TrackID: TypeAlias = int
EntityID: TypeAlias = int
Timestamp: TypeAlias = float
BBox: TypeAlias = tuple[float, float, float, float]  # x1, y1, x2, y2 px
Point2D: TypeAlias = tuple[float, float]
Homography: TypeAlias = npt.NDArray[np.float64]  # (3,3)


class CalibStatus(str, Enum):
    OK = "ok"
    TRACKED = "tracked"
    UNRELIABLE = "unreliable"
    NO_PITCH = "no_pitch"


class EntityClass(str, Enum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


@dataclass
class ShotInfo:
    shot_id: int
    start_frame: int
    end_frame: int
    view: Literal["main", "closeup", "crowd_graphic", "other"]


@dataclass
class FrameCalib:
    frame_idx: int
    t: float
    status: CalibStatus
    H_img2pitch: npt.NDArray | None  # (3,3)
    conf: float
    footprint_m: npt.NDArray | None  # Nx2
    source: Literal["keypoints", "propagated"]

    def is_usable(self) -> bool:
        """Return True if status is OK or TRACKED and conf > 0.5."""
        return self.status in (CalibStatus.OK, CalibStatus.TRACKED) and self.conf > 0.5


@dataclass
class Track2D:
    frame_idx: int
    track_id: int
    cls: EntityClass
    bbox_xyxy: BBox
    conf: float
    visible: bool = True
    interpolated: bool = False

    @property
    def center(self) -> Point2D:
        """Return the center of the bounding box."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def bottom_center(self) -> Point2D:
        """Return the bottom-center point, representing the feet position."""
        x1, _, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, y2

    @property
    def area(self) -> float:
        """Return bounding-box area in pixels squared.

        Degenerate or inverted boxes have zero area.
        """
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class TeamAssignment:
    track_id: int
    team: str  # 'A', 'B', 'referee'
    is_goalkeeper: bool
    confidence: float


@dataclass
class WorldState:
    frame_idx: int
    t: float
    entity_id: int
    team: str | None
    cls: str
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    visible: bool
    pos_sigma: float
    schema_version: str = "1.0"

    @property
    def speed(self) -> float:
        """Return speed in meters per second."""
        return hypot(self.vx, self.vy)
