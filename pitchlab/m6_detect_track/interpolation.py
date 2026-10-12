from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Sequence

from pitchlab.core.types import EntityClass, Track2D


def _copy_track(track: Track2D, **changes) -> Track2D:
    """Helper to shallow-copy a Track2D with selected field overrides."""
    return dataclasses.replace(track, **changes)


def interpolate_ball_gaps(
    tracks: Sequence[Track2D],
    max_gap_frames: int = 5,
) -> list[Track2D]:
    """Linearly interpolates short missing detections for ball tracks.

    Non-ball tracks are passed through untouched.
    """
    if not tracks:
        return []

    player_tracks: list[Track2D] = []
    ball_tracks_by_id: dict[int, list[Track2D]] = defaultdict(list)

    for tr in tracks:
        if tr.cls == EntityClass.BALL:
            ball_tracks_by_id[tr.track_id].append(tr)
        else:
            player_tracks.append(tr)

    interpolated_ball_tracks: list[Track2D] = []

    for track_id, b_tracks in ball_tracks_by_id.items():
        b_tracks.sort(key=lambda t: t.frame_idx)
        out_b: list[Track2D] = []

        for i in range(len(b_tracks)):
            curr = b_tracks[i]
            if not out_b:
                out_b.append(curr)
                continue

            prev = out_b[-1]
            gap = curr.frame_idx - prev.frame_idx

            if 1 < gap <= max_gap_frames + 1:
                # Linear interpolation for each intermediate frame
                prev_x1, prev_y1, prev_x2, prev_y2 = prev.bbox_xyxy
                curr_x1, curr_y1, curr_x2, curr_y2 = curr.bbox_xyxy

                for step in range(1, gap):
                    alpha = step / float(gap)
                    interp_idx = prev.frame_idx + step
                    interp_box = (
                        prev_x1 + alpha * (curr_x1 - prev_x1),
                        prev_y1 + alpha * (curr_y1 - prev_y1),
                        prev_x2 + alpha * (curr_x2 - prev_x2),
                        prev_y2 + alpha * (curr_y2 - prev_y2),
                    )
                    interp_conf = prev.conf + alpha * (curr.conf - prev.conf)
                    
                    interp_track = Track2D(
                        track_id=track_id,
                        frame_idx=interp_idx,
                        bbox_xyxy=interp_box,
                        conf=float(interp_conf),
                        cls=EntityClass.BALL,
                    )
                    out_b.append(interp_track)

            out_b.append(curr)

        interpolated_ball_tracks.extend(out_b)

    all_tracks = player_tracks + interpolated_ball_tracks
    all_tracks.sort(key=lambda t: (t.frame_idx, t.track_id))
    return all_tracks
