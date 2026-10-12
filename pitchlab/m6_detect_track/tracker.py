from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from pitchlab.core.config import ByteTrackConfig
from pitchlab.core.types import EntityClass, Track2D
from pitchlab.m6_detect_track.interpolation import _copy_track

if TYPE_CHECKING:
    import supervision as sv


class ByteTrackWrapper:
    """Wrapper around supervision.ByteTrack for multi-person tracking."""

    def __init__(self, cfg: ByteTrackConfig) -> None:
        self.cfg = cfg
        self._tracker: sv.ByteTrack | None = None
        self._id_offset = 0
        self._max_id_seen = 0

    def _get_tracker(self) -> sv.ByteTrack:
        if self._tracker is None:
            import supervision as sv

            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.cfg.track_thresh,
                lost_track_buffer=self.cfg.track_buffer,
                minimum_matching_threshold=self.cfg.match_thresh,
                frame_rate=self.cfg.frame_rate,
            )
        return self._tracker

    def reset(self) -> None:
        """Resets tracker state, preserving an ID offset to prevent collision."""
        self._id_offset = max(self._id_offset, self._max_id_seen + 1)
        self._tracker = None

    def update(self, detections: Sequence[Track2D], frame_idx: int) -> list[Track2D]:
        """Updates tracks with person detections for the current frame."""
        import supervision as sv

        tracker = self._get_tracker()
        player_dets = [d for d in detections if d.cls == EntityClass.PLAYER]
        if not player_dets:
            return []

        xyxy = np.array([list(d.bbox_xyxy) for d in player_dets], dtype=np.float32)
        confidence = np.array([d.conf for d in player_dets], dtype=np.float32)
        class_id = np.zeros(len(player_dets), dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked = tracker.update_with_detections(sv_detections)
        if len(tracked) == 0 or tracked.tracker_id is None:
            return []

        out_tracks: list[Track2D] = []
        for box, conf, tid in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
            assigned_id = int(tid) + self._id_offset
            self._max_id_seen = max(self._max_id_seen, assigned_id)
            out_tracks.append(
                Track2D(
                    track_id=assigned_id,
                    frame_idx=frame_idx,
                    bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    conf=float(conf) if conf is not None else 1.0,
                    cls=EntityClass.PLAYER,
                )
            )

        return out_tracks
