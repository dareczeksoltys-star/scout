"""Pitch line and intersection detection module for broadcast video frames.

Poprawki względem poprzedniej wersji (wynikają z przeglądu kodu i testów):
- detect_pitch_mask: morfologia skalowana do rozdzielczości obrazu (nie sztywne 7px);
  wypełnienie otworów (cv2.MORPH_CLOSE z dużym jądrem) dla szerokich linii na zbliżeniach.
- detect_line_mask: maska linii = jasność względem lokalnej mediany murawy AND niska
  saturacja (zamiast AND(tophat, kolor)); rozmiary jądra skalowane do wysokości obrazu.
- Dodano exclude_people_mask() odrzucającą duże plamy (napisy, reklamy).
- detect_line_segments: szkieletyzacja przed Hough (jeśli skimage dostępny), rozmiary
  HoughLinesP skalowane do obrazu.
- find_line_intersections: UWAGA – intersections to wyłącznie podpowiedzi dla hipotez,
  nie korespondencje do findHomography. Nie przypisuj ich do keypointów modelu bez
  dodatkowej weryfikacji.
- compute_distance_transform: bez twardego max_dist_px (gładszy gradient dla optymizatora).
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_pitch_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Detect grass pitch mask in BGR image using HSV color segmentation.

    Wersja poprawiona:
    - Rozmiar jądra morfologicznego skalowany do rozdzielczości (nie sztywne 7px).
    - Duże zamknięcie (MORPH_CLOSE) wypełnia szerokie białe linie, które w poprzedniej
      wersji powodowały dziury w masce murawy na zbliżeniach pola karnego.
    - Wypełnienie konturowe (cv2.floodFill-like z findContours) dla spójności maski.

    Returns:
        uint8 binary mask (0 or 255).
    """
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be np.ndarray")

    h, w = image_bgr.shape[:2]
    # Jądro morfologiczne skalowane do rozdzielczości: ~1.5% wysokości, min 5px
    ks = max(5, int(h * 0.015))
    if ks % 2 == 0:
        ks += 1

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Szeroki zakres zieleni dla różnych warunków oświetlenia broadcast
    lower_green = np.array([28, 30, 30], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Małe zamknięcie: wypełnienie cienkich linii (paski koszenia)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)

    # Duże zamknięcie: wypełnienie szerokich białych linii boiska (kluczowe na zbliżeniach)
    ks_large = max(ks * 4, 21)
    if ks_large % 2 == 0:
        ks_large += 1
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_large, ks_large))
    mask_filled = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, kernel_large)

    # Usunięcie szumu
    mask_clean = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel_small)

    return mask_clean


def _estimate_grass_brightness(image_bgr: np.ndarray, pitch_mask: np.ndarray) -> float:
    """Szacuje medianę jasności trawy wewnątrz maski (w skali 0–255)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grass_pixels = gray[pitch_mask > 0]
    if len(grass_pixels) < 100:
        return 80.0  # wartość domyślna
    return float(np.median(grass_pixels))


def exclude_people_mask(image_bgr: np.ndarray, pitch_mask: np.ndarray) -> np.ndarray:
    """Wyklucza duże jasne plamy (napisy na murawie, reklamy) z maski linii.

    Używa morfologii (duże otwarcie) aby odfiltrować prostokątne/duże blobs,
    które są zbyt szerokie/wysokie by być liniami boiska.

    Returns:
        uint8 maska obszarów DO WYKLUCZENIA (255 = wyklucz).
    """
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be np.ndarray")

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Jasne obszary o niskiej saturacji (białe napisy, koszulki)
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    low_sat = cv2.inRange(hsv, np.array([0, 0, 180], dtype=np.uint8),
                          np.array([180, 60, 255], dtype=np.uint8))
    candidates = cv2.bitwise_and(bright, low_sat)

    # Ogranicz do wnętrza murawy
    candidates = cv2.bitwise_and(candidates, pitch_mask)

    # Znajdź kontury i wyklucz zbyt duże plamy (> ~3% pola obrazu lub zbyt szerokie)
    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    exclusion_mask = np.zeros_like(candidates)

    max_line_width_px = max(8, int(w * 0.03))   # linia boiska max ~3% szerokości
    min_blob_area = int(h * w * 0.001)           # odrzucaj plamy > 0.1% pola obrazu

    for cnt in contours:
        area = cv2.contourArea(cnt)
        _, _, cw, ch = cv2.boundingRect(cnt)
        # Duże prostokątne blobs = napisy/reklamy; szerokie = na pewno nie linia
        if area > min_blob_area and cw > max_line_width_px and ch > max_line_width_px:
            cv2.drawContours(exclusion_mask, [cnt], -1, 255, cv2.FILLED)

    return exclusion_mask


def detect_line_mask(image_bgr: np.ndarray, pitch_mask: np.ndarray,
                     exclusion_mask: np.ndarray | None = None) -> np.ndarray:
    """Extract bright white pitch lines strictly inside the grass pitch mask.

    Wersja poprawiona:
    - Maska linii = jasność WZGLĘDEM lokalnej mediany murawy AND niska saturacja.
    - Rozmiary jąder skalowane do wysokości obrazu (nie sztywne px).
    - Opcjonalna maska wykluczeń (napisy, duże plamy) odejmowana przed zwrotem.

    Returns:
        binary line mask uint8 (0 or 255).
    """
    if not isinstance(image_bgr, np.ndarray) or not isinstance(pitch_mask, np.ndarray):
        raise TypeError("image_bgr and pitch_mask must be np.ndarray")

    h, w = image_bgr.shape[:2]
    # Skala jąder do rozdzielczości
    tophat_ks = max(5, int(h * 0.025))
    if tophat_ks % 2 == 0:
        tophat_ks += 1

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Maska koloru: niska saturacja + wysoka jasność
    lower_white = np.array([0, 0, 160], dtype=np.uint8)
    upper_white = np.array([180, 65, 255], dtype=np.uint8)
    color_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 2. Jasność względem lokalnej mediany murawy (robustna na zmienne oświetlenie)
    grass_median = _estimate_grass_brightness(image_bgr, pitch_mask)
    brightness_threshold = min(240, int(grass_median * 1.25 + 15))
    _, bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    # 3. Top-Hat morfologiczny – wydobycie cienkich jasnych linii na ciemnym tle
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_ks, tophat_ks))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel)
    tophat_thresh = max(12, int(grass_median * 0.12))
    _, tophat_mask = cv2.threshold(tophat, tophat_thresh, 255, cv2.THRESH_BINARY)

    # 4. Połączenie: (kolor AND jasność) OR top-hat
    combined = cv2.bitwise_or(cv2.bitwise_and(color_mask, bright_mask), tophat_mask)

    # Ogranicz do trawy
    line_mask = cv2.bitwise_and(combined, combined, mask=pitch_mask)

    # Usuń szum
    clean_ks = max(3, int(h * 0.004))
    if clean_ks % 2 == 0:
        clean_ks += 1
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (clean_ks, clean_ks))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel_clean)

    # Odejmij maskę wykluczeń (napisy, koszulki)
    if exclusion_mask is not None:
        line_mask = cv2.bitwise_and(line_mask, cv2.bitwise_not(exclusion_mask))

    return line_mask


def detect_line_segments(
    line_mask: np.ndarray, min_length: float = 25.0
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Detect line segments using Hough Lines on the binary line mask.

    Wersja poprawiona:
    - Szkieletyzacja przed Hough (jeśli skimage dostępny) – gruba linia → 1px szerokość,
      łuki nie rozpadają się na wiele równoległych odcinków.
    - Parametry HoughLinesP skalowane do rozdzielczości obrazu.

    Returns:
        list of line segments [((x1, y1), (x2, y2)), ...].
    """
    if not isinstance(line_mask, np.ndarray):
        raise TypeError("line_mask must be np.ndarray")

    if line_mask.dtype != np.uint8:
        line_mask = line_mask.astype(np.uint8)

    h, w = line_mask.shape[:2]

    # Szkieletyzacja – redukuje grube linie do 1px, lepszy Hough
    skel_mask = line_mask.copy()
    try:
        from skimage.morphology import skeletonize
        bool_mask = line_mask > 0
        skel = skeletonize(bool_mask)
        skel_mask = (skel * 255).astype(np.uint8)
    except ImportError:
        logger.debug("skimage nie dostępny, pomijam szkieletyzację")

    # Parametry Hough skalowane do rozdzielczości
    min_line_len = max(int(min_length), int(h * 0.04))
    max_gap = max(8, int(h * 0.015))
    hough_threshold = max(20, int(h * 0.03))

    lines = cv2.HoughLinesP(
        skel_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_len,
        maxLineGap=max_gap
    )

    segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    if lines is not None:
        for line in lines:
            pts = line.reshape(-1)
            if len(pts) < 4:
                continue
            x1, y1, x2, y2 = pts[:4]
            length = math.hypot(x2 - x1, y2 - y1)
            if length >= min_length:
                segments.append(((int(x1), int(y1)), (int(x2), int(y2))))

    return segments


def _compute_intersection(p1: Tuple[int, int], p2: Tuple[int, int],
                          p3: Tuple[int, int], p4: Tuple[int, int]) -> Tuple[float, float]:
    """Compute 2D intersection point between two line segments."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-4:
        raise ValueError("Parallel lines")

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (float(px), float(py))


def find_line_intersections(
    segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    max_distance: float = 18.0
) -> List[Tuple[float, float]]:
    """Find and cluster intersection points between line segments.

    WAŻNE: Zwrócone punkty to PODPOWIEDZI dla hipotez kalibracji – NIE korespondencje
    do keypointów modelu PitchModel. Nie używaj ich bezpośrednio do findHomography bez
    dodatkowej weryfikacji (linia–linia matching, RANSAC itp.).
    """
    raw_intersections: List[Tuple[float, float]] = []
    n = len(segments)

    def on_segment(pt: Tuple[float, float], seg: Tuple[Tuple[int, int], Tuple[int, int]], tol: float = 15.0) -> bool:
        (x1, y1), (x2, y2) = seg
        px, py = pt
        return (min(x1, x2) - tol <= px <= max(x1, x2) + tol) and (min(y1, y2) - tol <= py <= max(y1, y2) + tol)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                pt = _compute_intersection(segments[i][0], segments[i][1], segments[j][0], segments[j][1])
                if on_segment(pt, segments[i]) and on_segment(pt, segments[j]):
                    raw_intersections.append(pt)
            except ValueError:
                continue

    # Cluster close points
    clusters: List[List[Tuple[float, float]]] = []
    for pt in raw_intersections:
        placed = False
        for c in clusters:
            cx = sum(p[0] for p in c) / len(c)
            cy = sum(p[1] for p in c) / len(c)
            if math.hypot(pt[0] - cx, pt[1] - cy) <= max_distance:
                c.append(pt)
                placed = True
                break
        if not placed:
            clusters.append([pt])

    return [
        (float(sum(p[0] for p in c) / len(c)), float(sum(p[1] for p in c) / len(c)))
        for c in clusters
    ]


def compute_distance_transform(line_mask: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance transform from white lines in pixels.

    Wersja poprawiona: bez twardego max_dist_px – gładszy gradient dla optymalizatora.
    Interpolacja dwuliniowa w fit_ptz_to_line_dt obsługuje wartości > 50.

    Returns:
        float32 array, 0.0 na liniach, rosnąca poza nimi.
    """
    if not isinstance(line_mask, np.ndarray):
        raise TypeError("line_mask must be np.ndarray")

    if line_mask.dtype != np.uint8:
        line_mask = line_mask.astype(np.uint8)

    inverted = cv2.bitwise_not(line_mask)
    dt = cv2.distanceTransform(inverted, distanceType=cv2.DIST_L2, maskSize=5)
    return dt.astype(np.float32)
