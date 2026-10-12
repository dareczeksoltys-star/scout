from typing import List, Optional
from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    id: str
    filename: str
    filepath: str
    size_bytes: int
    duration_seconds: Optional[float] = None
    uploaded_at: str


class ClipSegment(BaseModel):
    id: str
    video_id: str
    video_name: str
    start_time: float
    end_time: float
    start_formatted: str
    end_formatted: str
    duration: float
    label: str
    notes: Optional[str] = ""
    created_at: str


class SegmentCreateRequest(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    label: Optional[str] = "Akcja"
    notes: Optional[str] = ""


class SegmentsDatabase(BaseModel):
    segments: List[ClipSegment] = []


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_fractional = seconds % 60
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds_fractional:05.2f}"
    else:
        return f"{minutes:02}:{seconds_fractional:05.2f}"
