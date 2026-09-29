"""Base protocol and registry for calibration backends.

Follows Section 7 (M2) of docs/PLAN.md:
class Calibrator(Protocol):
    name: str
    def calibrate(self, frame: np.ndarray, pitch: PitchTemplate, frame_idx: int = 0, t: float = 0.0) -> FrameCalib: ...
"""
from __future__ import annotations

from typing import Callable, Dict, Protocol, Type, runtime_checkable
import numpy as np

from pitchlab.core.pitch_template import PitchTemplate
from pitchlab.core.types import FrameCalib


@runtime_checkable
class Calibrator(Protocol):
    """Protocol that every pitch calibration backend must implement."""
    name: str

    def calibrate(
        self,
        frame: np.ndarray,
        pitch: PitchTemplate,
        frame_idx: int = 0,
        t: float = 0.0,
    ) -> FrameCalib:
        """Calibrates camera on a single video frame.
        
        Args:
            frame: BGR image array (H, W, 3).
            pitch: PitchTemplate defining FIFA dimensions.
            frame_idx: Index of current video frame.
            t: Timestamp of frame in seconds.
            
        Returns:
            FrameCalib object with status ("ok", "unreliable", "no_pitch"),
            homography H_img2pitch (3x3), confidence, and footprint polygon in meters.
        """
        ...


_CALIBRATOR_REGISTRY: Dict[str, Callable[..., Calibrator]] = {}


def register_calibrator(name: str):
    """Decorator to register a calibrator factory."""
    def decorator(cls_or_fn: Callable[..., Calibrator]):
        _CALIBRATOR_REGISTRY[name.lower()] = cls_or_fn
        return cls_or_fn
    return decorator


def get_calibrator(name: str, **kwargs) -> Calibrator:
    """Instantiates a calibration backend by name."""
    name_key = name.lower()
    if name_key not in _CALIBRATOR_REGISTRY:
        # Try lazy loading known backends
        if name_key == "roboflow":
            import pitchlab.m2_calib.backend_roboflow  # noqa: F401
        elif name_key == "pnlcalib":
            import pitchlab.m2_calib.backend_pnlcalib  # noqa: F401
        elif name_key == "snbaseline":
            import pitchlab.m2_calib.backend_snbaseline  # noqa: F401

    if name_key not in _CALIBRATOR_REGISTRY:
        available = list(_CALIBRATOR_REGISTRY.keys())
        raise ValueError(f"Unknown calibration backend: '{name}'. Available: {available}")

    return _CALIBRATOR_REGISTRY[name_key](**kwargs)


def make_empty_calib(frame_idx: int, t: float, status: str = "no_pitch") -> FrameCalib:
    """Helper to construct an uncalibrated FrameCalib."""
    return FrameCalib(
        frame_idx=frame_idx,
        t=t,
        status=status,  # type: ignore
        H_img2pitch=None,
        conf=0.0,
        footprint_m=None,
        source="keypoints",
    )
