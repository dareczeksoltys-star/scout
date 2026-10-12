from __future__ import annotations

import asyncio
import io
import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from pitchlab.m6_detect_track.pipeline import analyze_video_segment
from pitchlab.models import ClipSegment, SegmentCreateRequest, VideoInfo, format_timestamp
from pitchlab.storage import DataManager

app = FastAPI(title="PitchLab Video Annotation MVP", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_mgr = DataManager()
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def range_streamer(file_path: Path, range_header: Optional[str]) -> Response:
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")

    file_size = os.path.getsize(file_path)
    mime_type = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }.get(file_path.suffix.lower(), "video/mp4")

    if range_header:
        range_match = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if range_match:
            start = int(range_match.group(1))
            raw_end = range_match.group(2)
            end = int(raw_end) if raw_end else file_size - 1

            if start >= file_size:
                raise HTTPException(status_code=416, detail="Requested range not satisfiable")

            end = min(end, file_size - 1)
            if start > end:
                raise HTTPException(status_code=416, detail="Requested range not satisfiable")

            chunk_size = end - start + 1
            with open(file_path, "rb") as f:
                f.seek(start)
                content = f.read(chunk_size)

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type": mime_type,
            }
            return StreamingResponse(
                io.BytesIO(content),
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                headers=headers,
                media_type=mime_type,
            )

    return FileResponse(
        path=str(file_path),
        media_type=mime_type,
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/")
async def read_root() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index HTML not found")
    return FileResponse(str(index_file))


@app.post("/api/upload", response_model=VideoInfo)
async def upload_video(file: UploadFile = File(...)) -> VideoInfo:
    allowed_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Niedozwolony format pliku: {suffix}. Obsługiwane formaty: {', '.join(allowed_extensions)}",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Plik jest pusty.")

    video_info = data_mgr.save_uploaded_video(file.filename or "video.mp4", content)
    return video_info


@app.get("/api/videos", response_model=List[VideoInfo])
async def get_videos() -> List[VideoInfo]:
    return data_mgr.list_videos()


@app.get("/api/video/{video_id}/stream")
async def stream_video(request: Request, video_id: str) -> Response:
    video = data_mgr.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Wideo nie zostało znalezione.")
    return range_streamer(Path(video.filepath), request.headers.get("range"))


@app.post("/api/segments", response_model=ClipSegment)
async def create_segment(payload: SegmentCreateRequest) -> ClipSegment:
    if payload.end_time <= payload.start_time:
        raise HTTPException(
            status_code=400,
            detail="Czas zakończenia musi być większy niż czas rozpoczęcia.",
        )

    video = data_mgr.get_video(payload.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Wideo nie zostało znalezione.")

    segment = ClipSegment(
        id=f"seg_{uuid.uuid4().hex[:8]}",
        video_id=payload.video_id,
        video_name=video.filename,
        start_time=round(payload.start_time, 2),
        end_time=round(payload.end_time, 2),
        start_formatted=format_timestamp(payload.start_time),
        end_formatted=format_timestamp(payload.end_time),
        duration=round(payload.end_time - payload.start_time, 2),
        label=payload.label or "Akcja",
        notes=payload.notes or "",
        created_at=datetime.now().isoformat(),
    )

    data_mgr.save_segment(segment)
    return segment


@app.get("/api/segments", response_model=List[ClipSegment])
async def get_segments(video_id: Optional[str] = Query(None)) -> List[ClipSegment]:
    return data_mgr.get_segments(video_id)


@app.delete("/api/segments/{segment_id}")
async def delete_segment(segment_id: str) -> dict:
    deleted = data_mgr.delete_segment(segment_id)
    if deleted:
        return {"deleted": True, "id": segment_id}
    raise HTTPException(status_code=404, detail="Fragment nie został odnaleziony.")


@app.get("/api/segments/export")
async def export_segments() -> JSONResponse:
    return JSONResponse(content=data_mgr.export_data())


ANALYSIS_DIR = Path("data/out").resolve()

# Serialize inference within this server process to avoid GPU/CPU overload.
# Multiple worker processes need a shared job queue or cross-process lock.
_analysis_semaphore = asyncio.Semaphore(1)

_SEGMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _analysis_path(segment_id: str) -> Path:
    # segment_id comes from the URL: do not allow path traversal.
    if not _SEGMENT_ID_PATTERN.fullmatch(segment_id):
        raise HTTPException(status_code=422, detail="Invalid segment ID")
    return ANALYSIS_DIR / f"analysis_{segment_id}.json"


def _field(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def _load_segment_input(segment_id: str) -> tuple[Path, float, float]:
    segment = data_mgr.get_segment(segment_id)
    if segment is None:
        raise HTTPException(status_code=404, detail="Segment not found")

    video = data_mgr.get_video(_field(segment, "video_id"))
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_path = getattr(video, "filepath", None) or getattr(video, "path", None)
    if raw_path is None and isinstance(video, dict):
        raw_path = video.get("filepath") or video.get("path")
    video_path = Path(raw_path) if raw_path else Path("")
    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video file is unavailable",
        )

    return (
        video_path,
        float(_field(segment, "start_time")),
        float(_field(segment, "end_time")),
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Prevent GET from observing a partially written cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _analyze_and_cache(
    segment_id: str,
    cache_path: Path,
    video_path: Path,
    start_time: float,
    end_time: float,
) -> dict[str, Any]:
    result = analyze_video_segment(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        cfg=None,
        output_dir=ANALYSIS_DIR,
        segment_id=segment_id,
    )
    _write_json_atomic(cache_path, result)
    return result


@app.post("/api/segments/{segment_id}/analyze")
async def analyze_segment(segment_id: str) -> dict[str, Any]:
    cache_path = _analysis_path(segment_id)

    # Assumes a synchronous data manager.
    video_path, start_time, end_time = await run_in_threadpool(
        _load_segment_input, segment_id
    )

    async with _analysis_semaphore:
        try:
            # Keep decoding/inference off FastAPI's event loop.
            return await run_in_threadpool(
                _analyze_and_cache,
                segment_id,
                cache_path,
                video_path,
                start_time,
                end_time,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Video file is unavailable",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/segments/{segment_id}/analysis")
def get_segment_analysis(segment_id: str) -> dict[str, Any]:
    cache_path = _analysis_path(segment_id)

    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="No cached analysis exists for this segment",
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Cached analysis is invalid; run analysis again",
        ) from exc
