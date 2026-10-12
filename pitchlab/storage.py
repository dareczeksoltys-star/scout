import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pitchlab.models import ClipSegment, VideoInfo, SegmentsDatabase


class DataManager:
    def __init__(self, data_dir: Path = Path("data"), segments_file: Path = Path("data/segments.json")):
        self.data_dir = data_dir
        self.segments_file = segments_file
        self.lock = threading.Lock()

        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.data_dir / "videos", exist_ok=True)
        os.makedirs(self.data_dir / "raw", exist_ok=True)
        os.makedirs(self.data_dir / "out", exist_ok=True)

        # Ensure segments file exists with initial content
        if not self.segments_file.exists():
            with open(self.segments_file, "w", encoding="utf-8") as f:
                json.dump({"segments": []}, f, indent=2)

    def list_videos(self) -> List[VideoInfo]:
        video_extensions = (".mp4", ".mkv", ".avi", ".mov", ".webm")
        video_infos: List[VideoInfo] = []
        seen_paths = set()

        for dirpath in [self.data_dir / "videos", self.data_dir / "raw"]:
            if not dirpath.exists():
                continue
            for file in Path(dirpath).rglob("*"):
                if file.is_file() and file.suffix.lower() in video_extensions:
                    resolved = str(file.resolve())
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
                    video_info = VideoInfo(
                        id=file.stem,
                        filename=file.name,
                        filepath=str(file),
                        size_bytes=file.stat().st_size,
                        duration_seconds=None,
                        uploaded_at=datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    )
                    video_infos.append(video_info)

        return video_infos

    def get_video(self, video_id: str) -> Optional[VideoInfo]:
        for video in self.list_videos():
            if video.id == video_id or video.filename == video_id:
                return video
        return None

    def save_uploaded_video(self, filename: str, content: bytes) -> VideoInfo:
        # Sanitize filename to prevent directory traversal
        safe_name = Path(filename).name
        video_path = self.data_dir / "videos" / safe_name
        with open(video_path, "wb") as f:
            f.write(content)

        return VideoInfo(
            id=video_path.stem,
            filename=video_path.name,
            filepath=str(video_path),
            size_bytes=video_path.stat().st_size,
            duration_seconds=None,
            uploaded_at=datetime.fromtimestamp(video_path.stat().st_mtime).isoformat(),
        )

    def get_segments(self, video_id: Optional[str] = None) -> List[ClipSegment]:
        with self.lock:
            with open(self.segments_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = [ClipSegment(**s) for s in data.get("segments", [])]
            if video_id is not None:
                segments = [s for s in segments if s.video_id == video_id]

            return segments

    def get_segment(self, segment_id: str) -> Optional[ClipSegment]:
        for segment in self.get_segments():
            if segment.id == segment_id:
                return segment
        return None

    def save_segment(self, segment: ClipSegment) -> ClipSegment:
        with self.lock:
            with open(self.segments_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["segments"].append(segment.model_dump())

            with open(self.segments_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return segment

    def delete_segment(self, segment_id: str) -> bool:
        with self.lock:
            with open(self.segments_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = data.get("segments", [])
            for i, segment in enumerate(segments):
                if segment.get("id") == segment_id:
                    del segments[i]
                    break
            else:
                return False  # Segment not found

            data["segments"] = segments
            with open(self.segments_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True

    def export_data(self) -> dict:
        return {"segments": [s.model_dump() for s in self.get_segments()]}
