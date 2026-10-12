"""Video I/O and Parquet serialization for PitchLab data types."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from .types import EntityClass, TeamAssignment, Track2D, WorldState

logger = logging.getLogger(__name__)

# Explicit schemas preserve columns and types even for empty datasets.
_TRACK_SCHEMA: dict[str, str] = {
    "frame_idx": "int64",
    "track_id": "int64",
    "cls": "string",
    "x1": "float64",
    "y1": "float64",
    "x2": "float64",
    "y2": "float64",
    "conf": "float64",
    "visible": "bool",
    "interpolated": "bool",
}

_WORLD_SCHEMA: dict[str, str] = {
    "frame_idx": "int64",
    "t": "float64",
    "entity_id": "int64",
    "team": "string",
    "cls": "string",
    "x": "float64",
    "y": "float64",
    "vx": "float64",
    "vy": "float64",
    "ax": "float64",
    "ay": "float64",
    "visible": "bool",
    "pos_sigma": "float64",
    "schema_version": "string",
}

_TEAM_SCHEMA: dict[str, str] = {
    "track_id": "int64",
    "team": "string",
    "is_goalkeeper": "bool",
    "confidence": "float64",
}


class VideoReader:
    """Read zero-based video frames as OpenCV BGR arrays.

    iter_frames() starts at frame zero on each invocation and yields
    (frame_index, frame) pairs. Sampling means retaining every Nth frame,
    not targeting a particular frames-per-second rate.

    Instances maintain a single capture cursor and are not thread-safe.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._capture = cv2.VideoCapture(str(self.path))
        self._closed = False

        if not self._capture.isOpened():
            self._capture.release()
            self._closed = True
            raise OSError(f"Unable to open video: {self.path}")

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.debug(
            "Opened video %s: %dx%d, %.3f FPS, %d frames",
            self.path,
            self.width,
            self.height,
            self.fps,
            self.frame_count,
        )

    def _ensure_open(self) -> None:
        """Raise if this reader has been closed."""
        if self._closed or not self._capture.isOpened():
            raise RuntimeError("VideoReader is closed")

    def _seek(self, idx: int) -> None:
        """Position the capture at a frame, reopening for a failed rewind."""
        self._ensure_open()
        if self._capture.set(cv2.CAP_PROP_POS_FRAMES, idx):
            return

        if idx != 0:
            raise OSError(f"Unable to seek to frame {idx} in {self.path}")

        self._capture.release()
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self.close()
            raise OSError(f"Unable to reopen video: {self.path}")

    def iter_frames(
        self,
        sample_rate: int = 1,
    ) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
        """Yield every sample_rate-th frame, starting at frame zero."""
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a positive integer")

        self._seek(0)
        idx = 0
        while True:
            self._ensure_open()
            if not self._capture.grab():
                break

            if idx % sample_rate == 0:
                success, frame = self._capture.retrieve()
                if not success or frame is None:
                    raise OSError(f"Unable to decode frame {idx} in {self.path}")
                yield idx, frame

            idx += 1

    def get_frame(self, idx: int) -> npt.NDArray[np.uint8]:
        """Read one frame by zero-based index.

        Out-of-range or unreadable frames raise IndexError. Seeking accuracy
        depends on the video format and OpenCV backend.
        """
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
            raise ValueError("idx must be a nonnegative integer")

        self._ensure_open()
        if self.frame_count > 0 and idx >= self.frame_count:
            raise IndexError(f"Frame {idx} is outside the video")

        self._seek(idx)
        success, frame = self._capture.read()
        if not success or frame is None:
            raise IndexError(f"Unable to read frame {idx} in {self.path}")
        return frame

    def close(self) -> None:
        """Release video resources; repeated calls are safe."""
        if not self._closed:
            self._capture.release()
            self._closed = True
            logger.debug("Closed video %s", self.path)

    def __enter__(self) -> VideoReader:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __iter__(self):
        return self.iter_frames()


def _dataframe(
    records: list[dict[str, object]],
    schema: dict[str, str],
) -> pd.DataFrame:
    """Construct a consistently typed DataFrame, including for empty input."""
    return pd.DataFrame.from_records(records, columns=list(schema)).astype(schema)


def _save_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame without its index, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    logger.debug("Saved %d rows to %s", len(frame), output_path)


def _load_parquet(path: str | Path, schema: dict[str, str]) -> pd.DataFrame:
    """Read a Parquet file and verify its required columns."""
    frame = pd.read_parquet(path)
    missing = set(schema) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(sorted(missing))}"
        )
    return frame.loc[:, list(schema)].astype(schema)


def track2d_to_dataframe(tracks: list[Track2D]) -> pd.DataFrame:
    """Convert tracks to rows with flattened bounding-box coordinates."""
    records: list[dict[str, object]] = []
    for track in tracks:
        x1, y1, x2, y2 = track.bbox_xyxy
        records.append(
            {
                "frame_idx": track.frame_idx,
                "track_id": track.track_id,
                "cls": EntityClass(track.cls).value,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": track.conf,
                "visible": track.visible,
                "interpolated": track.interpolated,
            }
        )
    return _dataframe(records, _TRACK_SCHEMA)


def save_track2d_parquet(tracks: list[Track2D], path: str | Path) -> None:
    """Save 2D tracks to Parquet."""
    _save_parquet(track2d_to_dataframe(tracks), path)


def load_track2d_parquet(path: str | Path) -> list[Track2D]:
    """Load 2D tracks, restoring entity enums and bounding-box tuples."""
    frame = _load_parquet(path, _TRACK_SCHEMA)
    return [
        Track2D(
            frame_idx=int(row["frame_idx"]),
            track_id=int(row["track_id"]),
            cls=EntityClass(row["cls"]),
            bbox_xyxy=(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            ),
            conf=float(row["conf"]),
            visible=bool(row["visible"]),
            interpolated=bool(row["interpolated"]),
        )
        for row in frame.to_dict(orient="records")
    ]


save_tracks_parquet = save_track2d_parquet
load_tracks_parquet = load_track2d_parquet


def save_worldstate_parquet(states: list[WorldState], path: str | Path) -> None:
    """Save world states, preserving nullable team assignments."""
    records = [asdict(state) for state in states]
    _save_parquet(_dataframe(records, _WORLD_SCHEMA), path)


def load_worldstate_parquet(path: str | Path) -> list[WorldState]:
    """Load world states, converting missing team values to None."""
    frame = _load_parquet(path, _WORLD_SCHEMA)
    return [
        WorldState(
            frame_idx=int(row["frame_idx"]),
            t=float(row["t"]),
            entity_id=int(row["entity_id"]),
            team=None if pd.isna(row["team"]) else str(row["team"]),
            cls=str(row["cls"]),
            x=float(row["x"]),
            y=float(row["y"]),
            vx=float(row["vx"]),
            vy=float(row["vy"]),
            ax=float(row["ax"]),
            ay=float(row["ay"]),
            visible=bool(row["visible"]),
            pos_sigma=float(row["pos_sigma"]),
            schema_version=str(row["schema_version"]),
        )
        for row in frame.to_dict(orient="records")
    ]


def save_teamassignment_parquet(
    assignments: list[TeamAssignment],
    path: str | Path,
) -> None:
    """Save team assignments to Parquet."""
    records = [asdict(assignment) for assignment in assignments]
    _save_parquet(_dataframe(records, _TEAM_SCHEMA), path)


def load_teamassignment_parquet(path: str | Path) -> list[TeamAssignment]:
    """Load team assignments from Parquet."""
    frame = _load_parquet(path, _TEAM_SCHEMA)
    return [
        TeamAssignment(
            track_id=int(row["track_id"]),
            team=str(row["team"]),
            is_goalkeeper=bool(row["is_goalkeeper"]),
            confidence=float(row["confidence"]),
        )
        for row in frame.to_dict(orient="records")
    ]
