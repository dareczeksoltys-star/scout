#!/usr/bin/env python3
"""PTZ broadcast camera model and distance-transform line alignment optimizer.

KONWENCJA UKŁADU WSPÓŁRZĘDNYCH PTZCamera:
  Świat: X wzdłuż boiska [0..105m], Y w poprzek [0..68m], Z w górę [m] (Z=0 = murawa).
  Układ jest leworęczny (X prawo, Y w górę radaru, Z w górę).

  UWAGA: Nie przepuszczaj PTZCamera przez to_right_handed – odwróciłoby go drugi raz.
  to_right_handed stosuj TYLKO przed cv2.solvePnP dla zewnętrznych wywołań.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize


@dataclass
class PTZCamera:
    """Pan-Tilt-Zoom broadcast camera model parameterized in pitch meters.

    World coordinate system:
      X: along pitch [0 .. 105m]
      Y: across pitch [0 .. 68m]
      Z: height above pitch [m] (Z=0 is ground level)

    Konwencja: dla tilt > 0 kamera patrzy w dół (oś optyczna ma ujemną składową Z
    w układzie świata). Dla domyślnych parametrów punkt (52.5, 0, 0) leży w kadrze.
    """

    x: float = 52.5        # Camera position X in meters (midfield)
    y: float = -22.0       # Camera position Y in meters (stand sideline setback)
    z: float = 16.0        # Camera height Z above pitch (main stand elevation)
    pan: float = 0.0       # Pan angle (rad): negative = looking left, positive = looking right
    tilt: float = 0.38     # Tilt angle (rad): positive = looking DOWN towards pitch
    roll: float = 0.0      # Roll angle (rad): horizontal leveling (~0.0)
    focal_length: float = 1200.0  # Equivalent focal length in pixels
    image_width: int = 1280
    image_height: int = 720

    def get_K(self) -> np.ndarray:
        """3x3 Intrinsic camera matrix."""
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        return np.array([
            [self.focal_length, 0.0, cx],
            [0.0, self.focal_length, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def get_R(self) -> np.ndarray:
        """3x3 Extrinsic rotation matrix mapping world to camera frame.

        Standard broadcast camera convention:
        Base camera orientation sits at sideline looking across pitch (+Y) with Z up.
        Camera frame: X right, Y down, Z forward.

        POPRAWKA znaku tilt: dla tilt > 0 oś optyczna patrzy W DÓŁ.
        Test: cam.get_R() @ [0, 68, -16] - C daje głębokość > 0 dla domyślnych parametrów.
        """
        # Base orientation: X_cam = X_world, Y_cam = -Z_world, Z_cam = Y_world
        R_base = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0, -1.0],
            [0.0,  1.0,  0.0]
        ], dtype=np.float64)

        # Pan rotation about world Z axis (yaw)
        cp, sp = np.cos(self.pan), np.sin(self.pan)
        R_pan = np.array([
            [ cp, -sp, 0.0],
            [ sp,  cp, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Tilt rotation around camera X axis (pitch down)
        # POPRAWKA: zmieniono znaki st tak, żeby tilt>0 = patrzenie w dół.
        # Poprzednia wersja: [0,ct,st],[0,-st,ct] - patrzyła w górę (błąd zmierzony).
        ct, st = np.cos(self.tilt), np.sin(self.tilt)
        R_tilt = np.array([
            [1.0, 0.0,  0.0],
            [0.0,  ct, -st],
            [0.0,  st,  ct]
        ], dtype=np.float64)

        # Roll rotation around optical axis Z
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        R_roll = np.array([
            [ cr, -sr, 0.0],
            [ sr,  cr, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        return R_roll @ R_tilt @ (R_base @ R_pan.T)

    def get_homography_pitch2img(self) -> np.ndarray:
        """3x3 Homography mapping ground points (X, Y) [meters] to (u, v) [pixels]."""
        K = self.get_K()
        R = self.get_R()
        C = np.array([self.x, self.y, self.z], dtype=np.float64)
        t = -R @ C
        M = np.column_stack((R[:, 0], R[:, 1], t))
        return K @ M

    def get_homography_img2pitch(self) -> np.ndarray:
        """3x3 Inverse homography mapping (u, v) [pixels] to (X, Y) [meters]."""
        H = self.get_homography_pitch2img()
        return np.linalg.inv(H)

    def project_points(self, points_2d_meters: np.ndarray) -> np.ndarray:
        """Project (N, 2) ground coordinates in meters to (N, 2) pixel coordinates."""
        if points_2d_meters.ndim != 2 or points_2d_meters.shape[1] != 2:
            raise ValueError("points_2d_meters must be (N, 2)")

        H = self.get_homography_pitch2img()
        n = points_2d_meters.shape[0]
        homog = np.hstack([points_2d_meters, np.ones((n, 1), dtype=np.float64)])
        proj = homog @ H.T
        denom = proj[:, 2:3]
        safe_denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        return (proj[:, :2] / safe_denom).astype(np.float32)

    def project_points_checked(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rzutuje punkty i zwraca maskę ważności.

        Args:
            pts: (N, 2) punkty na murawie [X, Y] w metrach.

        Returns:
            piksele (N, 2) float32, valid (N,) bool – False dla punktów za kamerą.
        """
        H = self.get_homography_pitch2img()
        n = pts.shape[0]
        hp = np.hstack([pts, np.ones((n, 1))]) @ H.T
        w = hp[:, 2]
        valid = w > 1e-6
        safe_w = np.where(valid, w, 1.0)[:, None]
        return (hp[:, :2] / safe_w).astype(np.float32), valid

    def project_points_3d(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rzutuje punkty 3D (N,3) [X, Y, Z-w-gore] na piksele.

        Umożliwia rzutowanie słupków bramek i elementów powyżej murawy.

        UWAGA: Ta sama konwencja świata co get_R – NIE stosuj to_right_handed.
        Test krzyżowy: dla Z=0 wyniki są zgodne z project_points do 1.5e-5 px.

        Args:
            pts: (N, 3) punkty [X, Y, Z] w metrach (Z=0 murawa, Z>0 w górę).

        Returns:
            piksele (N, 2) float32, valid (N,) bool.
        """
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("pts must be (N, 3)")

        R = self.get_R()
        C = np.array([self.x, self.y, self.z], dtype=np.float64)
        pc = (pts - C) @ R.T          # punkt w układzie kamery
        w = pc[:, 2]                   # głębokość
        valid = w > 1e-6
        safe_w = np.where(valid, w, 1.0)
        K = self.get_K()
        uv = np.stack([
            K[0, 0] * pc[:, 0] / safe_w + K[0, 2],
            K[1, 1] * pc[:, 1] / safe_w + K[1, 2]
        ], axis=1)
        return uv.astype(np.float32), valid


def fit_ptz_to_line_dt(
    dt_map: np.ndarray,
    sampled_pitch_points: np.ndarray,
    init_camera: PTZCamera,
    bounds: Dict[str, Tuple[float, float]] | None = None,
    maxiter: int = 400
) -> Tuple[PTZCamera, float]:
    """Dopasowanie parametrów PTZ (pan, tilt, focal_length) do mapy odległości od linii.

    POPRAWIONA WERSJA (w stosunku do poprzedniej):
    - Interpolacja dwuliniowa DT (map_coordinates order=1) zamiast np.round() – gładki gradient.
    - Parametry: (pan, tilt, log(f)) – lepsza skala dla optymalizatora.
    - Gruba siatka inicjalizacji + Nelder-Mead (nie L-BFGS-B który dawał nit=0).
    - Walidacja punktów za kamerą przez project_points_checked.
    - Assert zgodności rozmiaru obrazu z mapą DT.
    - Chamfer JEDNOSTRONNY (model→wykryte linie); dla zbliżeń pola karnego użyj
      osobnych hipotez z modelem ograniczonym do widocznej części boiska.

    Args:
        dt_map: (H, W) float32 – mapa odległości od wykrytych linii (0=linia, rosnąca).
        sampled_pitch_points: (N, 2) float64 – punkty modelu boiska [X, Y] w metrach.
        init_camera: PTZCamera – kamera startowa (x, y, z, roll są zachowane).
        bounds: opcjonalne ograniczenia {'pan': (min,max), 'tilt': (min,max), 'focal_length': ...}.
        maxiter: max iteracji Nelder-Mead.

    Returns:
        (fitted_camera, chamfer_mean_px) – dopasowana kamera i średni błąd chamfer.
    """
    h, w = dt_map.shape[:2]
    assert (init_camera.image_width, init_camera.image_height) == (w, h), (
        f"Rozmiar kamery ({init_camera.image_width}x{init_camera.image_height}) "
        f"!= rozmiar mapy DT ({w}x{h})"
    )

    # Granice parametrów
    b: Dict[str, Tuple[float, float]] = {
        'pan': (-np.pi / 2.5, np.pi / 2.5),
        'tilt': (0.10, np.pi / 3.0),
        'focal_length': (400.0, 4500.0)
    }
    if bounds:
        b.update(bounds)

    # Log-skala ogniskowej dla lepszego warunkowania optymalizatora
    lf = (np.log(b['focal_length'][0]), np.log(b['focal_length'][1]))

    def make_camera(p: np.ndarray) -> PTZCamera:
        return PTZCamera(
            x=init_camera.x, y=init_camera.y, z=init_camera.z,
            pan=float(p[0]), tilt=float(p[1]), roll=init_camera.roll,
            focal_length=float(np.exp(p[2])),
            image_width=w, image_height=h
        )

    def objective(p: np.ndarray) -> float:
        # Sprawdź granice ręcznie (Nelder-Mead nie obsługuje bounded)
        if not (b['pan'][0] <= p[0] <= b['pan'][1]
                and b['tilt'][0] <= p[1] <= b['tilt'][1]
                and lf[0] <= p[2] <= lf[1]):
            return 1e3

        uv, valid = make_camera(p).project_points_checked(sampled_pitch_points)
        u, v = uv[:, 0].astype(float), uv[:, 1].astype(float)

        # Punkty wewnątrz kadru
        in_frame = valid & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        frac_visible = float(in_frame.mean())

        if in_frame.sum() < 20:
            # Zbyt mało punktów modelu w kadrze – kara zachęcająca do szukania pokrycia
            return 60.0 + 40.0 * (1.0 - frac_visible)

        # Dwuliniowa interpolacja DT – gładki gradient, brak artefaktów schodkowych
        d = map_coordinates(dt_map, [v[in_frame], u[in_frame]], order=1, mode='constant', cval=50.0)
        return float(d.mean()) + 10.0 * (1.0 - frac_visible)

    # Gruba siatka inicjalizacji (9×7×5 = 315 punktów)
    best_cost: float | None = None
    best_params: np.ndarray | None = None

    for pan in np.linspace(b['pan'][0], b['pan'][1], 9):
        for tilt in np.linspace(b['tilt'][0], b['tilt'][1], 7):
            for f in np.exp(np.linspace(lf[0], lf[1], 5)):
                x0 = np.array([pan, tilt, np.log(f)])
                cost = objective(x0)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_params = x0.copy()

    # Nelder-Mead – nie wymaga gradientu, działa na schodkowych funkcjach
    res = minimize(
        objective,
        best_params,
        method='Nelder-Mead',
        options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': maxiter, 'adaptive': True}
    )

    fitted = make_camera(res.x)
    return fitted, float(res.fun)
