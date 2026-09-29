# PITCHLAB — Rejestr Licencji Zależności i Wag Modeli (LICENSES.md)

Każda biblioteka, model bazowy oraz wagi dodawane do repozytorium muszą być odnotowane w tym dokumencie wraz z oceną ryzyka dla wersji SaaS/komercyjnej.

---

## 1. Zależności rdzenne i biblioteki bazowe

| Pakiet | Wersja | Licencja | Dozwolone w SaaS? | Uwagi |
|---|---|---|---|---|
| **Python** | 3.11 | PSF License | Tak | Środowisko bazowe |
| **PyTorch** | >= 2.0 | Modified BSD | Tak | Framework obliczeń tensorowych |
| **OpenCV (opencv-python-headless)** | >= 4.8 | Apache 2.0 | Tak | Używać wariantu headless na serwerze/kontenerze |
| **NumPy** | >= 1.24 | BSD-3-Clause | Tak | Operacje na macierzach |
| **pandas / pyarrow** | >= 2.0 | BSD-3-Clause / Apache 2.0 | Tak | Zapis/odczyt formatu Parquet |
| **Shapely** | >= 2.0 | BSD-3-Clause | Tak | Obliczenia geometryczne i przycinanie wielokątów |
| **supervision** | >= 0.19 | MIT | Tak | ByteTrack, wizualizacja detekcji |
| **PySceneDetect (scenedetect)** | >= 0.6 | BSD-3-Clause | Tak | Detekcja cięć montażowych kamery |
| **kloppy** | >= 3.0 | MIT | Tak | Odczyt formatów danych trackingowych |
| **mplsoccer** | >= 1.2 | MIT | Tak | Szablony i wizualizacja boiska |
| **fsspec / gcsfs** | >= 2024.1 | BSD-3-Clause | Tak | Transparentny dostęp do GCS (`gs://`) i dysku |
| **FastAPI / Uvicorn** | >= 0.110 | MIT | Tak | Backend aplikacji i API |
| **google-cloud-firestore** | >= 2.15 | Apache 2.0 | Tak | Baza metadanych i stanu czasu rzeczywistego |
| **google-cloud-storage** | >= 2.14 | Apache 2.0 | Tak | Obsługa bucketa GCS |
| **pytest** | >= 8.0 | MIT | Tak | Środowisko testowe |

---

## 2. Modele CV i wagi (Ocena ryzyka licencyjnego)

| Model / Moduł | Repozytorium / Źródło | Licencja | Ryzyko SaaS | Rekomendacja |
|---|---|---|---|---|
| **Roboflow Sports / pitch keypoints** | `roboflow/sports` | MIT / Apache (kod), wagi: CC BY 4.0 lub model license | Niskie/Średnie | Szybki start (M2). Wymaga weryfikacji warunków wag Roboflow Universe. |
| **PnLCalib** | `mguti97/PnLCalib` | Badawcza / Open Source | Do weryfikacji | Główny kandydat M2 do kalibracji z korekcją dystorsji. |
| **SoccerNet sn-calibration** | `SoccerNet` | MIT / Badawcza | Do weryfikacji | Baseline M2 i format ewaluacji M5. |
| **Ultralytics YOLO (v8/v11)** | `ultralytics/ultralytics` | **AGPL-3.0** | **WYSOKIE** (AGPL) | Kod/wagi pod AGPL wymagają udostępnienia kodu całego systemu lub komercyjnej licencji od Ultralytics. Alternatywa czysto komercyjna: **RT-DETR** (Apache 2.0) lub **YOLOv10** (Apache 2.0) dla detekcji zawodników w M6. |
| **SigLIP / CLIP (koszulki)** | HuggingFace / Google Research | Apache 2.0 | Niskie (bezpieczne) | Embeddingi koszulek do klastrowania drużyn (M7). |
