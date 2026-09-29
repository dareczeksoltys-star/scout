# pitchlab/core/io.py
"""
I/O utilities for PitchLab.

The module abstracts file access for the two primary storage back‑ends used by the
application:

* Local filesystem paths (e.g. ``/tmp/video.mp4``)
* Google Cloud Storage URIs (``gs://bucket/path``)

Only a small, well‑defined subset of functionality is required for the core
pipeline at this stage – loading configuration files and reading video streams.
All functions raise explicit, typed exceptions and perform basic validation to
prevent path‑traversal attacks or accidental overwrites.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Generator

import yaml
from pathvalidate import sanitize_filename


# Optional GCS import – the package is listed as a dependency. Import lazily so
# that the core library can be used without GCS credentials when only local
# files are needed.
try:
    from google.cloud import storage
except Exception:  # pragma: no cover – import guarded for environments without GCS.
    storage = None  # type: ignore

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _is_gs_path(path: str) -> bool:
    """Return ``True`` if *path* is a ``gs://`` URI.

    The check is deliberately simple – a valid GCS URI must start with the
    scheme ``gs://`` and contain at least a bucket name.
    """
    return path.startswith("gs://") and len(path) > 5

def _validate_local_path(path: str) -> pathlib.Path:
    """Validate *path* as a safe local filesystem path.

    The function resolves the path and ensures it is absolute.  It also guards
    against ``..`` components that could escape a designated root directory.
    For now the root is the repository base (``SCOUT``) but the logic can be
    extended later.
    """
    p = pathlib.Path(path).expanduser().resolve()
    # Ensure the path is inside the repository directory.
    repo_root = pathlib.Path(os.getenv("SCOUT_ROOT", "c:/Users/admin/Desktop/SCOUT")).resolve()
    if not str(p).startswith(str(repo_root)):
        raise PermissionError(f"Access to path {p} is outside the allowed repository root.")
    return p

# ---------------------------------------------------------------------------
# Configuration (YAML) handling
# ---------------------------------------------------------------------------

def read_yaml_config(path: str) -> Dict[str, Any]:
    """Read a YAML configuration file from *path*.

    Supports both local files and ``gs://`` URIs.
    """
    if _is_gs_path(path):
        if storage is None:
            raise RuntimeError("google-cloud-storage is not available in the current environment.")
        client = storage.Client()
        _, bucket_name, *blob_parts = path.split('/')
        blob_name = "/".join(blob_parts)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        raw = blob.download_as_text()
    else:
        p = _validate_local_path(path)
        with p.open("r", encoding="utf-8") as f:
            raw = f.read()
    return yaml.safe_load(raw) or {}

def write_yaml_config(data: Dict[str, Any], path: str, *, overwrite: bool = False) -> None:
    """Write *data* as a YAML file to *path*.

    Parameters
    ----------
    data:
        Mapping to serialise.
    path:
        Destination – local or ``gs://``.
    overwrite:
        If ``False`` (default) the function will raise ``FileExistsError`` when a
        local file already exists.  Overwrites are allowed for GCS objects.
    """
    yaml_str = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if _is_gs_path(path):
        if storage is None:
            raise RuntimeError("google-cloud-storage is not available.")
        client = storage.Client()
        _, bucket_name, *blob_parts = path.split('/')
        blob_name = "/".join(blob_parts)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(yaml_str, content_type="application/x-yaml")
    else:
        p = _validate_local_path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"File {p} already exists and overwrite=False.")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(yaml_str)

# ---------------------------------------------------------------------------
# Video handling
# ---------------------------------------------------------------------------

def open_video(path: str):
    """Open a video file for reading.

    Returns an ``cv2.VideoCapture`` object when OpenCV is installed.  For GCS
    locations the function streams the file to a temporary local file first.
    The caller is responsible for releasing the capture object.
    """
    try:
        import cv2  # Imported lazily – optional dependency.
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is required for video handling.") from exc

    if _is_gs_path(path):
        import tempfile
        if storage is None:
            raise RuntimeError("google-cloud-storage is required for GCS video access.")
        client = storage.Client()
        _, bucket_name, *blob_parts = path.split('/')
        blob_name = "/".join(blob_parts)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(blob_name).suffix) as tmp:
            blob.download_to_filename(tmp.name)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
    else:
        p = _validate_local_path(path)
        cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"Failed to open video at {path}")
    return cap

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def read_json(path: str) -> Any:
    """Read a JSON file from *path* (local or GCS)."""
    if _is_gs_path(path):
        if storage is None:
            raise RuntimeError("google-cloud-storage not available.")
        client = storage.Client()
        _, bucket_name, *blob_parts = path.split('/')
        blob_name = "/".join(blob_parts)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        raw = blob.download_as_text()
    else:
        p = _validate_local_path(path)
        with p.open("r", encoding="utf-8") as f:
            raw = f.read()
    return json.loads(raw)

def write_json(data: Any, path: str, *, overwrite: bool = False) -> None:
    """Write *data* to *path* as JSON."""
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    if _is_gs_path(path):
        if storage is None:
            raise RuntimeError("google-cloud-storage not available.")
        client = storage.Client()
        _, bucket_name, *blob_parts = path.split('/')
        blob_name = "/".join(blob_parts)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json_str, content_type="application/json")
    else:
        p = _validate_local_path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"File {p} already exists and overwrite=False.")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(json_str)

# Provide explicit save_json alias/function as required by external modules
def save_json(data: Any, path: str, *, overwrite: bool = False) -> None:
    """Save data to path as JSON (alias for write_json)."""
    write_json(data, path, overwrite=overwrite)

# ---------------------------------------------------------------------------
# PARQUET helper
# ---------------------------------------------------------------------------

def save_parquet(data: Any, path: str, *, overwrite: bool = False) -> None:
    """Save data (pandas DataFrame or convertible object) to a parquet file (local or GCS)."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Pandas is required for parquet handling.") from exc

    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data

    if _is_gs_path(path):
        if storage is None:
            raise RuntimeError("google-cloud-storage not available.")
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
            tmp_path = tmp.name
        df.to_parquet(tmp_path, index=False)
        
        client = storage.Client()
        _, bucket_name, *blob_parts = path.split('/')
        blob_name = "/".join(blob_parts)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(tmp_path, content_type="application/octet-stream")
        os.unlink(tmp_path)
    else:
        p = _validate_local_path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"File {p} already exists and overwrite=False.")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(p), index=False)

# ---------------------------------------------------------------------------
# CACHE helper
# ---------------------------------------------------------------------------

def resolve_local_cache(path: str) -> str:
    """Ensure a given path (local or GCS) is available locally.
    If it's a GCS URI, download it to a local cache path and return that path.
    If it's already local, validate and return it as a string.
    """
    if _is_gs_path(path):
        if storage is None:
            raise RuntimeError("google-cloud-storage is required for GCS caching.")
        import tempfile
        client = storage.Client()
        _, bucket_name, *blob_parts = path.split('/')
        blob_name = "/".join(blob_parts)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(blob_name).suffix) as tmp:
            tmp_path = tmp.name
        blob.download_to_filename(tmp_path)
        return tmp_path
    else:
        p = _validate_local_path(path)
        return str(p)

# ---------------------------------------------------------------------------
# PIPELINE WRAPPER
# ---------------------------------------------------------------------------

class VideoReader:
    """A context-manager wrapper around open_video that provides high-level 
    access to frames and stream properties.
    """
    def __init__(self, path: str):
        self.path = path
        self.cap = open_video(path)

    def read_frames(self) -> Generator[Any, None, None]:
        """Generator that yields individual image frames from the video stream."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def close(self) -> None:
        """Release the video capture resource."""
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

"""End of I/O utilities."""