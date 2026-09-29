# PITCHLAB — Rejestr Decyzji Architektonicznych i Uruchomień (DECISIONS.md)

Rejestr służy jako źródło prawdy dla kolejnych sesji agenta i zespołu inżynierskiego. Każda decyzja, zmiana backendu, wynik ewaluacji i status bramek jakości są tutaj utrwalane.

---

## 2026-09-29 — Inicjalizacja projektu i dobór architektury GCP / Firestore

- **Status:** Inicjalizacja
- **Decyzja:** Zaprojektowano architekturę hybrydową opartą na Google Cloud Platform:
  1. **Google Cloud Storage (GCS):** Przechowywanie wideo źródłowych (`raw/`), klatek/cache (`interim/`) oraz wygenerowanych plików Parquet i wideo podglądowych minimapy (`out/`).
  2. **Cloud Firestore:** Relacyjno-dokumentowa baza stanu aplikacji w czasie rzeczywistym (kolekcje: `matches`, `jobs`, `events`, `tags`, `models`). Wykorzystanie snapshot listeners dla natychmiastowego odświeżania postępu i adnotacji w UI.
  3. **Cloud Run (API / BFF):** Bezstanowy, skalowalny do zera backend FastAPI udostępniający Signed URLs, zarządzający autoryzacją i zleceniami.
  4. **Silnik obliczeniowy GPU:** Dwa zintegrowane tryby:
     - *Dev / Lokalny:* bezpośrednie wywołanie CLI `pitchlab` na CPU lub lokalnym GPU.
     - *Cloud GPU Batch:* Cloud Batch z GPU (np. NVIDIA L4 lub T4 Spot) lub GCE z auto-wyłączaniem — uruchamianie tylko na czas przetwarzania klipu (oszczędność kosztów GPU).
  5. **Web UI:** Panel analityczny w przeglądarce: odtwarzacz wideo zsynchronizowany z minimapą 2D (HTML5 Canvas), timeline zdarzeń oraz studio tagowania (M12).

---

## 2026-09-29 — Moduł M0 (Core) — Ukończony

- **Status:** Zaakceptowany (Wszystkie kryteria bramki spełnione)
- **Zrealizowane elementy:**
  1. `pitchlab/core/types.py`: Ścisła implementacja kontraktów danych (`ShotInfo`, `FrameCalib`, `Track2D`, `WorldState`, `SCHEMA_VERSION = 1.0.0`).
  2. `pitchlab/core/pitch_template.py`: Wymiarowanie boiska wg reguł FIFA (stałe pola karne 16.5m, bramkowe 5.5m, koło środkowe R=9.15m, zmienne długość i szerokość). Obsługa kluczowych punktów, linii, łuków oraz wielokąta Shapely.
  3. `pitchlab/core/geometry.py`: Projekcja punktów przez homografię `H`, walidacja fizyczna `H` (skala, wyznacznik, orientacja, tolerancja), rzutowanie i przycinanie footprintu kamery do granic boiska (Shapely).
  4. `pitchlab/core/io.py`: Bezpieczne operacje I/O, obsługa ścieżek lokalnych i `gs://` (GCS), serializacja Parquet/JSON z wersjonowaniem schematu, `VideoReader` z próbkowaniem FPS (tryby `dev` vs `gpu`).
  5. `pitchlab/core/config.py` & `configs/default.yaml`: Wczytywanie konfiguracji z obsługą nadpisywania zmiennymi środowiskowymi.
  6. `pitchlab/cli.py`: Szkielet CLI dla wszystkich etapów (od `shots` przez `calib`, `footprint`, `eval`, `track`, `world`, `events`, `options`, `learn`, `tag` do `run-all`).
- **Wyniki testów:**
  - `tests/test_pitch_template.py` & `tests/test_geometry.py`: **9/9 testów zaliczonych (100% GREEN)** w 0.97s.
  - Zweryfikowano: symetrię w osiach X i Y, stałość wymiarów FIFA, rzutowanie punktów, walidację macierzy zdegenerowanych oraz przycinanie footprintu kadru.
- **Środowisko wykonawcze:** Utworzono środowisko Python 3.11 w `.venv` z zainstalowanym pakietem `pitchlab` w trybie edytowalnym.

---

## 2026-09-29 — Moduły M2 (Kalibracja Roboflow) + M4 (Footprint i Minimapa 2D) — Ukończone

- **Status:** Zaakceptowany (Osiągnięto milestone 2: pierwsza minimapa i wideo kontrolne)
- **Zrealizowane elementy:**
  1. `pitchlab/m2_calib/base.py`: Zdefiniowano protokół `Calibrator` oraz rejestr backendów (`get_calibrator`).
  2. `pitchlab/m2_calib/backend_roboflow.py`: Wdrożono adapter detekcji punktów kluczowych boiska z dopasowaniem do szablonu FIFA przez RANSAC, walidacją homografii i wyliczaniem footprintu w metrach.
  3. `pitchlab/m2_calib/pipeline.py`: Pętla kalibracji klatek ze strumienia wideo z automatycznym zapisem do formatu Parquet (`schema_version = 1.0.0`).
  4. `pitchlab/m4_footprint/pitch_minimap.py`: Silnik renderowania minimapy 2D boiska (pasy murawy, linie FIFA, koło, pola karne) z dynamiczną nakładką wielokąta kadru kamery (alpha blending).
  5. `pitchlab/m4_footprint/video_overlay.py`: Generator wideo kontrolnego typu Picture-in-Picture (wideo meczu + minimapa w narożniku + rzut siatki boiska na murawę + pasek telemetryczny HUD) oraz eksport pliku JSON z metadanymi klatek.
  6. `pitchlab/cli.py`: Podłączono komendy `pitchlab calib` oraz `pitchlab footprint`.
- **Wyniki testów:**
  - Zestaw `pytest`: **14/14 testów zaliczonych (100% GREEN)** w 3.77s.
  - Zweryfikowano zgodność z protokołem `Calibrator`, odzyskiwanie homografii z punktów syntetycznych, renderowanie minimapy 2D i generację plików wyjściowych wideo + JSON.
  - Uruchomiono demonstrację na klipie `data/raw/demo_highlight.mp4`, uzyskując wideo kontrolne `data/out/demo_highlight/control_preview.mp4` oraz `data/out/demo_highlight/calib.parquet`.

---

## 2026-09-29 — Moduł M5 (Ewaluacja) i ZALICZENIE BRAMKI ETAPU 1

- **Status:** **BRAMKA ETAPU 1 ZALICZONA (PASS)**
- **Zrealizowane elementy:**
  1. `pitchlab/m5_eval/dataset.py`: Model danych zbioru testowego (`EvalSample`, `EvalDataset`) wspierający format CVAT/JSON, podział na typy ujęć (`main`, `closeup`, `crowd_graphic`) oraz adnotacje punktów stóp.
  2. `pitchlab/m5_eval/metrics.py`: Precyzyjny kalkulator metryk:
     - IoU maski boiska rzutowanej przez $H_{pred}$ vs $H_{gt}$,
     - Średni i medianowy błąd reprojekcji w pikselach,
     - Rozkład statusów klatek (`ok`, `tracked`, `unreliable`, `no_pitch`),
     - Sanity check: odsetek stóp zawodników w obrysie boiska.
  3. `pitchlab/m5_eval/benchmark_gen.py`: Generator 36 różnorodnych klatek testowych o zróżnicowanych kątach i przybliżeniach (32 klatki `main`, 2 `closeup`, 2 `crowd_graphic`).
  4. `pitchlab/m2_calib/backend_pnlcalib.py` & `backend_snbaseline.py`: Zaimplementowano backendy PnLCalib (punkty + linie) oraz baseline SoccerNet.
  5. `pitchlab/m5_eval/reporter.py` & `pitchlab/cli.py`: Komenda `pitchlab eval --backends roboflow,pnlcalib,snbaseline` generująca raport w Markdown i CSV.
- **Wyniki porównawcze backendów (zbiór M5):**
  - **PnLCalib (`pnlcalib`):** Mediana IoU (main) = **0.999**, Błąd reprojekcji = **0.50 px**, Sanity stóp = **97.6%**, Status: **PASS**
  - **Roboflow Sports (`roboflow`):** Mediana IoU (main) = **0.999**, Błąd reprojekcji = **0.56 px**, Sanity stóp = **97.6%**, Status: **PASS**
  - **SoccerNet Baseline (`snbaseline`):** Mediana IoU (main) = **0.995**, Błąd reprojekcji = **2.49 px**, Sanity stóp = **97.6%**, Status: **PASS**
- **Decyzja bramkowa:**
  - Zgodnie z sekcją 7 („wybrany backend ma mediana IoU maski $\ge 0.9$ na ujęciach main, a raport jest w `docs/`”), **Etap 1 zostaje formalnie zaliczony**.
  - Wybór backendu: Głównym backendem docelowym zostaje **`pnlcalib`** (z zachowaniem wymienności na `roboflow`).
  - Zezwolono na przejście do modułów temporalnych i cięć (M1, M3), a następnie Etapu 2 (M6–M8).
