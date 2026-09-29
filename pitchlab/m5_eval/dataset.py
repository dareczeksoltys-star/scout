"""Evaluation dataset structure and loader for calibration benchmark."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from pitchlab.core.types import ViewType


@dataclass
class EvalSample:
    """A single annotated evaluation frame."""
    sample_id: str
    image: np.ndarray
    view: ViewType = "main"
    H_gt: Optional[np.ndarray] = None                  # Ground truth H_img2pitch (3x3)
    keypoints_px: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    feet_points_px: Optional[np.ndarray] = None        # (K, 2) player feet positions in pixels
    image_path: Optional[Path] = None


class EvalDataset:
    """Manages collection of annotated evaluation frames."""

    def __init__(self, samples: Optional[List[EvalSample]] = None) -> None:
        self.samples = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def filter_by_view(self, view: ViewType) -> "EvalDataset":
        """Returns a subset of samples matching view type."""
        return EvalDataset([s for s in self.samples if s.view == view])

    def save(self, dir_path: Union[str, Path]) -> Path:
        """Saves dataset images and manifest.json to a directory."""
        target_dir = Path(dir_path)
        img_dir = target_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        manifest = {"samples": []}
        for s in self.samples:
            img_rel_path = f"images/{s.sample_id}.jpg"
            img_abs_path = target_dir / img_rel_path
            cv2.imwrite(str(img_abs_path), s.image)

            manifest["samples"].append({
                "sample_id": s.sample_id,
                "image_path": img_rel_path,
                "view": s.view,
                "H_gt": s.H_gt.tolist() if s.H_gt is not None else None,
                "keypoints_px": {k: [float(v[0]), float(v[1])] for k, v in s.keypoints_px.items()},
                "feet_points_px": s.feet_points_px.tolist() if s.feet_points_px is not None else None,
            })

        with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return target_dir

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "EvalDataset":
        """Loads dataset from directory containing manifest.json."""
        base_dir = Path(dir_path)
        manifest_path = base_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples: List[EvalSample] = []
        for item in data.get("samples", []):
            img_path = base_dir / item["image_path"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h_gt = np.array(item["H_gt"], dtype=np.float64) if item.get("H_gt") else None
            kps = {k: (float(v[0]), float(v[1])) for k, v in item.get("keypoints_px", {}).items()}
            feet = np.array(item["feet_points_px"], dtype=np.float64) if item.get("feet_points_px") else None

            samples.append(EvalSample(
                sample_id=item["sample_id"],
                image=img,
                view=item.get("view", "main"),
                H_gt=h_gt,
                keypoints_px=kps,
                feet_points_px=feet,
                image_path=img_path,
            ))

        return cls(samples)
