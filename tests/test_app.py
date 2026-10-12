from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from pitchlab.models import (
    ClipSegment,
    SegmentCreateRequest,
    VideoInfo,
    format_timestamp,
)
from pitchlab.server import app
from pitchlab.storage import DataManager


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_models():
    # Test format_timestamp
    assert format_timestamp(75.5) == "01:15.50"
    assert format_timestamp(5.2) == "00:05.20"
    assert format_timestamp(3661.0) == "01:01:01.00"

    # Test VideoInfo
    v = VideoInfo(
        id="test_vid",
        filename="test_vid.mp4",
        filepath="/tmp/test_vid.mp4",
        size_bytes=1024,
        uploaded_at="2026-10-11T12:00:00",
    )
    assert v.id == "test_vid"
    assert v.filename == "test_vid.mp4"

    # Test SegmentCreateRequest
    req = SegmentCreateRequest(
        video_id="test_vid",
        start_time=10.0,
        end_time=25.5,
        label="Gol",
        notes="Świetny strzał",
    )
    assert req.start_time == 10.0
    assert req.end_time == 25.5


def test_storage_lifecycle(tmp_path: Path):
    data_dir = tmp_path / "data"
    segments_file = data_dir / "segments.json"
    mgr = DataManager(data_dir=data_dir, segments_file=segments_file)

    # Save dummy video
    v_info = mgr.save_uploaded_video("sample_match.mp4", b"fake mp4 video bytes for testing")
    assert v_info.filename == "sample_match.mp4"
    assert v_info.id == "sample_match"

    # Verify listing
    videos = mgr.list_videos()
    assert any(v.id == "sample_match" for v in videos)

    # Verify get_video
    found = mgr.get_video("sample_match")
    assert found is not None
    assert found.filename == "sample_match.mp4"

    # Save segment
    seg = ClipSegment(
        id="seg_123",
        video_id="sample_match",
        video_name="sample_match.mp4",
        start_time=12.0,
        end_time=24.5,
        start_formatted="00:12.00",
        end_formatted="00:24.50",
        duration=12.5,
        label="Kontratak",
        notes="Szybka akcja",
        created_at="2026-10-11T12:05:00",
    )
    mgr.save_segment(seg)

    # Retrieve segments
    segs = mgr.get_segments("sample_match")
    assert len(segs) == 1
    assert segs[0].id == "seg_123"
    assert mgr.get_segment("seg_123") is not None
    assert mgr.get_segment("nonexistent") is None

    # Export
    exported = mgr.export_data()
    assert "segments" in exported
    assert len(exported["segments"]) == 1

    # Delete segment
    deleted = mgr.delete_segment("seg_123")
    assert deleted is True
    assert len(mgr.get_segments()) == 0


def test_api_endpoints(client: TestClient):
    # Test GET /
    response = client.get("/")
    assert response.status_code == 200
    assert "PitchLab" in response.text

    # Test GET /api/videos
    response = client.get("/api/videos")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

    # Test POST /api/upload
    dummy_bytes = b"header_dummy_video_stream_0123456789" * 10
    response = client.post(
        "/api/upload",
        files={"file": ("test_highlight.mp4", dummy_bytes, "video/mp4")},
    )
    assert response.status_code == 200
    video_data = response.json()
    assert "id" in video_data
    vid_id = video_data["id"]

    # Test POST /api/segments (valid)
    seg_resp = client.post(
        "/api/segments",
        json={
            "video_id": vid_id,
            "start_time": 5.0,
            "end_time": 15.5,
            "label": "Akcja bramkowa / Gol",
            "notes": "Testowy fragment",
        },
    )
    assert seg_resp.status_code == 200
    created_seg = seg_resp.json()
    assert created_seg["duration"] == 10.5
    assert created_seg["start_formatted"] == "00:05.00"
    assert created_seg["end_formatted"] == "00:15.50"
    seg_id = created_seg["id"]

    # Test POST /api/segments (invalid range start >= end)
    err_resp = client.post(
        "/api/segments",
        json={
            "video_id": vid_id,
            "start_time": 20.0,
            "end_time": 10.0,
            "label": "Błąd",
        },
    )
    assert err_resp.status_code == 400

    # Test GET /api/segments
    list_resp = client.get("/api/segments")
    assert list_resp.status_code == 200
    all_segs = list_resp.json()
    assert any(s["id"] == seg_id for s in all_segs)

    # Test GET /api/video/{vid_id}/stream with Range header (HTTP 206)
    stream_resp = client.get(
        f"/api/video/{vid_id}/stream",
        headers={"Range": "bytes=0-15"},
    )
    assert stream_resp.status_code == 206
    assert "bytes 0-15/" in stream_resp.headers.get("Content-Range", "")

    # Test GET /api/segments/export
    export_resp = client.get("/api/segments/export")
    assert export_resp.status_code == 200
    assert "segments" in export_resp.json()

    # Test DELETE /api/segments/{seg_id}
    del_resp = client.delete(f"/api/segments/{seg_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True
