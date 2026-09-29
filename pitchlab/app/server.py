"""FastAPI application server for PitchLab Studio."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pitchlab.app.db import MatchRepository
from pitchlab.core.config import AppConfig
from pitchlab.core.io import sanitize_filename
from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.m2_calib.pipeline import run_calibration
from pitchlab.m4_footprint.video_overlay import render_control_video
from pitchlab.m5_eval.reporter import run_benchmark_and_report

app = FastAPI(title="PitchLab Studio API", version="1.0.0")

# Security: CORS restricted to local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo = MatchRepository()
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/out")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def background_process_pipeline(match_id: str, raw_video_path: Path):
    """Executes calibration, footprint, and video rendering pipeline in background."""
    try:
        cfg = AppConfig.load()
        pitch = PitchTemplate(cfg.pitch.length_m, cfg.pitch.width_m)
        match_out_dir = OUT_DIR / match_id
        match_out_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Calibration (M2)
        repo.update_match(match_id, {"status": "processing", "current_step": "M2_calib", "progress_pct": 25})
        out_parquet = match_out_dir / "calib.parquet"
        calib_results = run_calibration(
            video_path=raw_video_path,
            output_parquet_path=out_parquet,
            backend_name=cfg.calibration.default_backend,
            config=cfg,
        )

        # Step 2: Footprint & Control Video (M4)
        repo.update_match(match_id, {"current_step": "M4_footprint", "progress_pct": 65})
        out_video = match_out_dir / "control_preview.mp4"
        out_json = match_out_dir / "calib_summary.json"

        render_control_video(
            video_path=raw_video_path,
            calib_results=calib_results,
            output_video_path=out_video,
            output_json_path=out_json,
            pitch=pitch,
            config=cfg,
        )

        # Step 3: Load summary and complete
        repo.update_match(match_id, {"current_step": "M9_events", "progress_pct": 90})
        summary_data = {}
        if out_json.exists():
            with open(out_json, "r", encoding="utf-8") as f:
                summary_data = json.load(f)

        repo.update_match(match_id, {
            "status": "completed",
            "current_step": "done",
            "progress_pct": 100,
            "preview_url": f"/api/video/{match_id}/preview",
            "calib_summary": summary_data,
        })
    except Exception as e:
        print(f"[PITCHLAB ERROR] Processing failed for match {match_id}: {e}")
        repo.update_match(match_id, {
            "status": "error",
            "error_msg": str(e),
            "progress_pct": 0,
        })


@app.get("/api/matches")
def list_matches():
    return repo.list_matches()


@app.get("/api/matches/{match_id}")
def get_match(match_id: str):
    m = repo.get_match(match_id)
    if not m:
        raise HTTPException(status_code=404, detail="Match not found")
    return m


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...), title: Optional[str] = Form(None)):
    clean_name = sanitize_filename(file.filename or "highlight.mp4")
    match_entry = repo.create_match(clean_name, title=title)
    match_id = match_entry["id"]

    target_path = RAW_DIR / f"{match_id}_{clean_name}"
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    repo.update_match(match_id, {"local_raw_path": str(target_path)})
    return match_entry


@app.post("/api/matches/{match_id}/process")
def process_match(match_id: str, background_tasks: BackgroundTasks):
    m = repo.get_match(match_id)
    if not m:
        raise HTTPException(status_code=404, detail="Match not found")

    raw_path_str = m.get("local_raw_path")
    if not raw_path_str or not Path(raw_path_str).exists():
        # Check fallback
        fallback = RAW_DIR / f"{match_id}_{m['filename']}"
        if fallback.exists():
            raw_path_str = str(fallback)
        elif Path("data/raw/demo_highlight.mp4").exists():
            raw_path_str = "data/raw/demo_highlight.mp4"
        else:
            raise HTTPException(status_code=400, detail="Source video file not found on disk")

    background_tasks.add_task(background_process_pipeline, match_id, Path(raw_path_str))
    repo.update_match(match_id, {"status": "queued", "progress_pct": 5})
    return {"status": "queued", "match_id": match_id}


@app.post("/api/matches/{match_id}/tags")
def add_tag(match_id: str, tag: str = Form(...), timestamp: float = Form(...), comment: str = Form("")):
    entry = repo.add_tag(match_id, tag_name=tag, timestamp_sec=timestamp, comment=comment)
    if not entry:
        raise HTTPException(status_code=404, detail="Match not found")
    return entry


@app.get("/api/video/{match_id}/raw")
def stream_raw_video(match_id: str):
    m = repo.get_match(match_id)
    if not m:
        raise HTTPException(status_code=404, detail="Match not found")
    raw_path = m.get("local_raw_path") or str(RAW_DIR / f"{match_id}_{m['filename']}")
    if not Path(raw_path).exists():
        if Path("data/raw/demo_highlight.mp4").exists():
            raw_path = "data/raw/demo_highlight.mp4"
        else:
            raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(raw_path, media_type="video/mp4")


@app.get("/api/video/{match_id}/preview")
def stream_preview_video(match_id: str):
    preview_path = OUT_DIR / match_id / "control_preview.mp4"
    if not preview_path.exists():
        # Fallback to demo preview if exists
        fallback = OUT_DIR / "demo_highlight" / "control_preview.mp4"
        if fallback.exists():
            preview_path = fallback
        else:
            raise HTTPException(status_code=404, detail="Preview video not ready")
    return FileResponse(str(preview_path), media_type="video/mp4")


@app.get("/api/eval")
def get_eval_report():
    report_file = Path("docs/EVAL_REPORT.md")
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            return {"report_markdown": f.read()}
    return {"report_markdown": "No report generated yet. Run `pitchlab eval`."}


@app.get("/", response_class=HTMLResponse)
def get_studio_ui():
    """Serves the complete PitchLab Studio Web App."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PitchLab Studio UI loading...</h1>"


# --- BLOK URUCHOMIENIOWY SERWERA ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pitchlab.app.server:app", host="127.0.0.1", port=8000, reload=True)