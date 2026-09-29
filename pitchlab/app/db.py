"""Database abstraction supporting both local file/JSON storage and Google Cloud Firestore."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class MatchRepository:
    """Manages match metadata, processing status, events, and tags with Firestore fallback."""

    def __init__(self, data_dir: str | Path = "data", project_id: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.local_db_path = self.data_dir / "matches_db.json"
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")

        self.firestore_client = None
        if self.project_id:
            try:
                from google.cloud import firestore
                self.firestore_client = firestore.Client(project=self.project_id)
                print(f"[PITCHLAB DB] Connected to Google Cloud Firestore (Project: {self.project_id})")
            except Exception as e:
                print(f"[PITCHLAB DB] Firestore connection not active ({e}). Using local store.")

        self._init_local_db()

    def _init_local_db(self):
        if not self.local_db_path.exists():
            with open(self.local_db_path, "w", encoding="utf-8") as f:
                json.dump({"matches": {}}, f, indent=2)

    def _read_local_db(self) -> Dict[str, Any]:
        try:
            with open(self.local_db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"matches": {}}

    def _write_local_db(self, data: Dict[str, Any]):
        with open(self.local_db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def create_match(self, filename: str, title: Optional[str] = None) -> Dict[str, Any]:
        match_id = str(uuid.uuid4())[:8]
        match_data = {
            "id": match_id,
            "title": title or Path(filename).stem.replace("_", " ").title(),
            "filename": filename,
            "status": "uploaded",
            "current_step": "idle",
            "progress_pct": 0,
            "error_msg": None,
            "created_at": time.time(),
            "video_url": f"/api/video/{match_id}/raw",
            "preview_url": None,
            "calib_summary": None,
            "events": [],
            "tags": [],
        }

        # Local save
        db = self._read_local_db()
        db["matches"][match_id] = match_data
        self._write_local_db(db)

        # Firestore save if connected
        if self.firestore_client:
            try:
                self.firestore_client.collection("matches").document(match_id).set(match_data)
            except Exception as e:
                print(f"[PITCHLAB DB] Firestore write error: {e}")

        return match_data

    def update_match(self, match_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        db = self._read_local_db()
        if match_id not in db["matches"]:
            return None

        db["matches"][match_id].update(updates)
        self._write_local_db(db)

        if self.firestore_client:
            try:
                self.firestore_client.collection("matches").document(match_id).update(updates)
            except Exception as e:
                print(f"[PITCHLAB DB] Firestore update error: {e}")

        return db["matches"][match_id]

    def get_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        db = self._read_local_db()
        return db["matches"].get(match_id)

    def list_matches(self) -> List[Dict[str, Any]]:
        db = self._read_local_db()
        matches = list(db["matches"].values())
        matches.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return matches

    def add_tag(self, match_id: str, tag_name: str, timestamp_sec: float, comment: str = "") -> Optional[Dict[str, Any]]:
        db = self._read_local_db()
        if match_id not in db["matches"]:
            return None

        tag_entry = {
            "id": str(uuid.uuid4())[:8],
            "tag": tag_name,
            "timestamp": timestamp_sec,
            "comment": comment,
            "created_at": time.time(),
        }

        db["matches"][match_id].setdefault("tags", []).append(tag_entry)
        self._write_local_db(db)

        if self.firestore_client:
            try:
                self.firestore_client.collection("matches").document(match_id).collection("tags").document(tag_entry["id"]).set(tag_entry)
            except Exception as e:
                print(f"[PITCHLAB DB] Firestore tag save error: {e}")

        return tag_entry
