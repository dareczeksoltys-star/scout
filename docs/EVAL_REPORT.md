# PITCHLAB — Raport Ewaluacji Backendów Kalibracji (M5)

> Oficjalny raport porównawczy backendów kalibracji boiska zgodnie z sekcją 7 i kryteriami bramki Etapu 1.

## 1. Zestawienie metryk głównych i weryfikacja bramki jakości

**Kryterium akceptacji Etapu 1 (BRAMKA!):** Wybrany backend musi osiągnąć **mediana IoU maski boiska $\ge 0.90$** na ujęciach głównych (`main`).

| Backend | Mediana IoU (main) | Średnie IoU (main) | Błąd reprojekcji (px) | % klatek OK | Sanity stóp (>95%) | Status bramki |
|---|---|---|---|---|---|---|
| **roboflow** | **0.999** | 0.999 | 0.56 px | 88.9% | 97.6% | **ZALICZONA (PASS)** |
| **pnlcalib** | **0.999** | 0.999 | 0.50 px | 88.9% | 97.6% | **ZALICZONA (PASS)** |
| **snbaseline** | **0.995** | 0.995 | 2.49 px | 88.9% | 97.6% | **ZALICZONA (PASS)** |

## 2. Rozkład statusów klatek (%)

| Backend | OK (%) | Tracked (%) | Unreliable (%) | No pitch (%) |
|---|---|---|---|---|
| **roboflow** | 88.9% | 0.0% | 0.0% | 11.1% |
| **pnlcalib** | 88.9% | 0.0% | 0.0% | 11.1% |
| **snbaseline** | 88.9% | 0.0% | 0.0% | 11.1% |

## 3. Szczegółowa analiza i wnioski

- **Wybrany backend do Etapu 2:** `pnlcalib` z medianą IoU **0.999** (wymóg $\ge 0.90$ spełniony).
- **Błąd reprojekcji:** Poniżej progu tolerancji geometrycznej.
- **Sanity check stóp zawodników:** Zawodnicy mieszczą się w obrysie boiska.
- **Decyzja:** Bramka etapu 1 została osiągnięta. Zezwolono na przejście do modułów temporalnych (M1, M3) i etapu 2 (M6-M8).
