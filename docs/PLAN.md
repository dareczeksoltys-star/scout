# PITCHLAB — plan wykonawczy dla agenta AI (kodowanie)

> Ten dokument jest briefem dla agenta piszącego kod. Czytaj go w całości przed pierwszą linijką kodu. Pracuj moduł po module, w kolejności z sekcji 7. Nie przeskakuj etapów.

---

## 1. Cel

Użytkownik wrzuca **skróty meczów piłkarskich** (pliki wideo z transmisji TV). System:

1. rozpoznaje, **który fragment boiska widzi kamera** i wymiaruje boisko (znane wymiary, np. 105×68 m) mimo zmian kamery, zoomu i cięć,
2. wykrywa zawodników i piłkę, przelicza ich pozycje na **współrzędne boiska w metrach**,
3. liczy wektory prędkości i **przyspieszeń** zawodników oraz **trajektorię piłki**,
4. dla każdego momentu posiadania piłki wylicza **możliwości podań**,
5. **sam uczy się wag** (co decyduje o wyborze podania, o jakości akcji) na podstawie danych z wgranych skrótów,
6. pozwala użytkownikowi ręcznie **tagować** fragmenty akcji (pressing, dobra wrzutka itd.) i używa tych tagów jako dodatkowych etykiet.

Etap 1 (najpierw!) = punkty 1. Reszta dopiero po zaliczeniu bramek jakości etapu 1.

## 2. Zasady nadrzędne

- **Używamy istniejących rozwiązań.** Nie implementuj od zera tego, co ma dojrzałe repozytorium/bibliotekę (lista w sekcji 4). Piszemy cienkie adaptery i logikę łączącą.
- **Wszystko modułami.** Każdy moduł: własny podpakiet, jawny interfejs (Protocol), własne testy, własna komenda CLI, wejście/wyjście w plikach (Parquet/JSON), żeby dało się uruchamiać etapy niezależnie i wznawiać.
- **Backendy wymienne.** Kalibracja i detekcja stoją za interfejsem; porównujemy kilka backendów na tym samym zbiorze testowym.
- **Mierzymy, zanim wybierzemy.** Bez modułu ewaluacji (M5) nie wybieramy backendu i nie idziemy do etapu 2.
- **Licencje.** Przed dodaniem zależności/wag zapisz jej licencję w `docs/LICENSES.md`. Oznacz wprost: Ultralytics YOLO = AGPL (komercyjnie wymaga licencji), wagi z Roboflow Universe i modele z repozytoriów naukowych = sprawdź osobno. Projekt może stać się SaaS-em — nie wprowadzaj zależności z niejasną licencją bez flagi w PR.
- **Uczciwość co do danych.** Nie zgaduj i nie fabrykuj metryk. Jeśli coś nie działa lub danych jest za mało, zapisz to w raporcie z uruchomienia.
- **Jedna zmiana = jeden moduł.** Nie refaktoryzuj poza zakresem zadania. Piny wersji (`uv`/`poetry` lock), `ruff`, `mypy` (tam gdzie sensowne), `pytest`.

## 3. Środowisko

- Python 3.11, PyTorch, OpenCV, NumPy, pandas/pyarrow, Shapely, `supervision`, `scenedetect`, `kloppy`, `mplsoccer`, `pytest`.
- **Bez Jetsona i bez optymalizacji pod edge.** Nie dodawaj TensorRT/ONNX ani kodu specyficznego dla urządzeń wbudowanych. Cel: uruchamianie na zwykłym komputerze i na maszynie z GPU w GCP.
- **Dwa tryby uruchomienia (ten sam kod, różne configi):**
  - `dev` — zwykły komputer, CPU lub słabe GPU; niski FPS próbkowania, lżejsze modele, krótkie klipy; służy do pisania kodu i testów (`pytest` musi przechodzić bez GPU),
  - `gpu` — maszyna z GPU w GCP; pełny FPS, docelowe modele, przetwarzanie skrótów.
  Urządzenie wybieraj automatycznie (`cuda` jeśli dostępne, inaczej `cpu`) i loguj wybór.
- **GCP:**
  - obliczenia: pojedyncza maszyna GCE z GPU NVIDIA (np. L4 lub T4 — sprawdź dostępność w wybranym regionie), obraz z CUDA lub własny Docker; na start bez orkiestracji,
  - dane: bucket GCS na wideo i wyniki (`raw/`, `interim/`, `out/`); `core/io.py` ma obsługiwać ścieżki lokalne i `gs://` (fsspec + gcsfs), a przetwarzanie robi na lokalnym dysku maszyny (cache),
  - sekrety i dostęp: konto usługi z minimalnymi uprawnieniami do bucketa; żadnych kluczy w repozytorium,
  - zadania wsadowe (opcjonalnie, później): ten sam obraz Dockera jako Vertex AI Custom Job lub Batch — nie buduj tego w etapach 1–2.
- **Praca w Google Antigravity:** repozytorium jest źródłem prawdy; ten dokument leży w repo jako `docs/PLAN.md`. Po każdym module agent uruchamia testy i zapisuje wynik w `docs/DECISIONS.md`, żeby kolejna sesja agenta miała pełny kontekst bez pamięci poprzedniej.
- Wejście: pliki wideo. Nagrywanie telewizora kamerą **nie jest wspierane** (mora, zniekształcenia); docelowo przechwyt HDMI/strumień zapisany do pliku.

## 4. Gotowe rozwiązania do użycia

| Zadanie | Kandydat | Uwagi |
|---|---|---|
| Kalibracja boiska (punkty+linie, 3D/homografia) | **PnLCalib** (mguti97/PnLCalib) | Główny kandydat; ma optymalizację zniekształceń obiektywu |
| Kalibracja — baseline | **SoccerNet/sn-calibration** | Punkt odniesienia i format metryki |
| Kalibracja — szybki start | **roboflow/sports** (model punktów boiska + `ViewTransformer`) | Dobry do pierwszego uruchomienia |
| Geometria | **OpenCV** (`findHomography`, `perspectiveTransform`, `calcOpticalFlowPyrLK`) | |
| Cięcia kamery | **PySceneDetect** | |
| Wielokąty | **Shapely** | Przycinanie footprintu do boiska |
| Detekcja zawodników/piłki | YOLO (detekcja) + **SAHI/tiling** dla piłki | Sprawdź licencję (AGPL) |
| Tracking | **ByteTrack** (przez `supervision`) | |
| Drużyny | embeddingi koszulek (SigLIP) + UMAP + KMeans (wzór w roboflow/sports) | |
| Format danych trackingowych | **kloppy** | Wczytywanie otwartych zbiorów trackingowych |
| Kontrola przestrzeni | model *pitch control* (Spearman 2017; Fernández & Bornn) — użyj istniejącej implementacji (np. Friends of Tracking / Metrica), zweryfikuj licencję | |
| Wizualizacja | **mplsoccer** | |
| Otwarte dane do pretrainingu | Metrica sample data, SkillCorner open data, StatsBomb open data, SoccerNet (calibration/tracking/GSR) | **Zweryfikuj dostępność i licencje** przed użyciem |

Jeśli któregoś repozytorium nie da się zainstalować lub jest porzucone — zapisz to w `docs/DECISIONS.md` i wybierz alternatywę, nie pisz własnej wersji bez zgody.

## 5. Struktura repozytorium

```
pitchlab/
  pyproject.toml
  Dockerfile                         # ten sam obraz lokalnie i na GCP
  docs/ (PLAN.md, LICENSES.md, DECISIONS.md, RUNBOOK.md)
  configs/ (default.yaml)            # wymiary boiska, FPS próbkowania, progi
  pitchlab/
    core/
      types.py                       # dataclasses (sekcja 6)
      pitch_template.py              # szablon boiska (length, width)
      geometry.py                    # projekcja, footprint, walidacja H
      io.py                          # wideo, Parquet, JSON; ścieżki lokalne i gs:// (fsspec/gcsfs)
      config.py
    m1_shots/                        # cięcia + klasyfikacja typu ujęcia
    m2_calib/                        # base.py + backend_*.py
    m3_temporal/                     # propagacja H, wygładzanie
    m4_footprint/                    # footprint + minimapa/overlay
    m5_eval/                         # zbiór testowy, metryki, raporty
    m6_detect_track/                 # zawodnicy, piłka, tracking w pikselach
    m7_teams/                        # przypisanie drużyn, rola (sędzia/bramkarz)
    m8_world/                        # projekcja do metrów, kinematyka
    m9_events/                       # posiadanie piłki, podania, zdarzenia
    m10_options/                     # pitch control, opcje podań, cechy
    m11_learn/                       # uczenie wag
    m12_tags/                        # ręczne tagi + eksport etykiet
    cli.py
  tests/
  data/ (raw/, interim/, out/)       # poza gitem
```

## 6. Kontrakty danych (nie zmieniaj bez powodu)

```python
# core/types.py
from dataclasses import dataclass
from typing import Literal
import numpy as np

CalibStatus = Literal["ok", "tracked", "unreliable", "no_pitch"]

@dataclass
class ShotInfo:
    shot_id: int
    start_frame: int
    end_frame: int
    view: Literal["main", "closeup", "crowd_graphic", "other"]

@dataclass
class FrameCalib:
    frame_idx: int
    t: float                                   # sekundy
    status: CalibStatus
    H_img2pitch: np.ndarray | None             # 3x3, wynik w metrach
    conf: float
    footprint_m: np.ndarray | None             # Nx2, wielokąt kadru na boisku (m)
    source: Literal["keypoints", "propagated"]

@dataclass
class Track2D:                                 # M6, w pikselach
    frame_idx: int; track_id: int
    cls: Literal["player", "goalkeeper", "referee", "ball"]
    bbox_xyxy: tuple[float, float, float, float]
    conf: float

@dataclass
class WorldState:                              # M8, w metrach (jedna klatka)
    frame_idx: int; t: float
    entity_id: int; team: int | None           # 0/1/None (sędzia, piłka)
    cls: str
    x: float; y: float                         # pozycja na boisku (m)
    vx: float; vy: float; ax: float; ay: float # z filtrowania (m/s, m/s²)
    visible: bool                              # False = poza kadrem/interpolowany
    pos_sigma: float                           # niepewność pozycji (m)
```

Wyjścia zapisuj w Parquet (`data/out/<match>/<module>.parquet`), z kolumną `schema_version`.

**Konwencje współrzędnych:** początek (0,0) w środku boiska, oś X wzdłuż długości, Y wzdłuż szerokości, jednostki: metry, sekundy. Kierunek ataku drużyn ustalany w M9 i zapisywany w metadanych.

## 7. Moduły — specyfikacja i kryteria akceptacji

### ETAP 1 — kalibracja boiska (najpierw!)

**M0 `core`** — typy, konfiguracja, czytanie wideo (próbkowanie co N-tą klatkę, FPS w configu), zapis Parquet/JSON, `PitchTemplate(length, width)` zwracający linie/łuki/punkty kluczowe w metrach. Pole karne, bramkowe, koło środkowe mają **stałe rozmiary FIFA**; zmienne są tylko `length`, `width`.
*Akceptacja:* testy jednostkowe szablonu (punkty w granicach boiska, symetria).

**M1 `m1_shots`** — PySceneDetect tnie wideo na ujęcia; klasyfikator typu ujęcia (start: heurystyka — udział zieleni w HSV, liczba wykrytych linii; opcjonalnie mały model po zebraniu etykiet). Kalibrujemy tylko ujęcia `main` i (ostrożnie) `closeup` z widocznymi liniami.
*Akceptacja:* na zbiorze testowym ≥95% klatek `crowd_graphic` nie trafia do kalibracji.

**M2 `m2_calib`** — interfejs:

```python
class Calibrator(Protocol):
    name: str
    def calibrate(self, frame: np.ndarray, pitch: PitchTemplate) -> FrameCalib: ...
```

Backendy: `backend_roboflow` (start), `backend_pnlcalib`, `backend_snbaseline`. Każdy adapter zamienia wynik na `H_img2pitch` w metrach. **Walidacja `H`** w `core/geometry.py`: rozsądna skala, brak odbicia lustrzanego, dodatni wyznacznik, punkty wewnątrz tolerancji boiska; niespełnienie → `status="unreliable"`.
*Akceptacja:* każdy backend przechodzi ten sam zestaw testów kontraktowych i zwraca wynik na zbiorze M5.

**M3 `m3_temporal`** — utrzymanie kalibracji między klatkami: gdy detekcja nie ma dość linii (zbliżenie), propaguj ostatnią dobrą `H` ruchem tła (LK optical flow lub ORB+RANSAC na obszarach niebędących zawodnikami); przy cięciu kamery reset stanu; wygładzanie `H` (np. filtr na parametrach homografii albo na rogach kadru), żeby footprint nie skakał; `status="tracked"` dla klatek propagowanych.
*Akceptacja:* w przejściu przez zbliżenie kadr nie dryfuje >3 m względem ręcznej oceny na klatkach kontrolnych; brak skoków footprintu >5 m między sąsiednimi klatkami bez cięcia.

**M4 `m4_footprint`** — rzutowanie rogów klatki przez `H` → wielokąt na boisku, **przycięty do prostokąta boiska (Shapely)**; górne rogi często leżą za horyzontem/na trybunach, więc obowiązkowo obsłuż przypadek `H` dającego punkty w nieskończoności/za kamerą. Wizualizacja: minimapa 2D z zaznaczonym kadrem + overlay siatki metrów na wideo. Eksport wideo kontrolnego.
*Akceptacja:* wideo kontrolne dla każdego skrótu; JSON `FrameCalib` dla każdej klatki.

**M5 `m5_eval`** — zbiór testowy: 30–50 klatek z różnych ujęć ręcznie oznaczonych (linie/punkty boiska) — narzędzie do adnotacji w prostym GUI/Jupyter lub format CVAT. Metryki: IoU zrzutowanej maski boiska z ręczną maską, błąd reprojekcji (px), odsetek `ok/tracked/unreliable/no_pitch`, test sanity: stopy zawodników z detekcji mieszczą się w granicach boiska (>95%). Raport porównawczy backendów (tabela Markdown + CSV).
*Akceptacja etapu 1 (bramka!):* wybrany backend ma mediana IoU maski ≥0.9 na ujęciach `main`, a raport jest w `docs/`. Dopiero po tym etap 2.

### ETAP 2 — zawodnicy, piłka, kinematyka

**M6 `m6_detect_track`** — detekcja (zawodnicy/sędzia/bramkarz) + osobna ścieżka dla piłki (tiling/SAHI, mały obiekt), ByteTrack dla zawodników, dla piłki filtr Kalmana + interpolacja luk (oznacz `visible=False`).
*Akceptacja:* pokrycie piłki i stabilność ID mierzone na ręcznie sprawdzonych fragmentach; raport w M5.

**M7 `m7_teams`** — przypisanie drużyny (embeddingi koszulek + klasteryzacja), wykrycie sędziego/bramkarzy; ustalenie kierunku ataku.
*Akceptacja:* >95% poprawnych drużyn na próbce sprawdzonej ręcznie.

**M8 `m8_world`** — projekcja punktów stóp (środek dolnej krawędzi bbox) przez `H` klatki do metrów; **kinematyka po projekcji, nie w pikselach**: ruch kamery zaszumia prędkości, więc filtr Kalmana/Savitzky–Golay na pozycjach w metrach, twarde limity fizyczne (prędkość ≲ 12 m/s, przyspieszenie ≲ 9 m/s²; przekroczenie → odrzucenie/wygładzenie), `pos_sigma` rosnące z odległością od kamery i przy `status="tracked"`. Zawodnicy poza kadrem: `visible=False`, bez zmyślania pozycji.
*Akceptacja:* rozkłady prędkości/przyspieszeń zgodne z fizyką (raport histogramów); brak nierealnych skoków.

### ETAP 3 — zdarzenia, opcje podań, uczenie

**M9 `m9_events`** — automatyczna detekcja posiadania piłki (bliskość zawodnika do piłki, niska prędkość piłki, histereza), zmiana posiadania → kandydat na podanie; klasyfikacja wyniku: podanie do partnera / przechwyt / strata; pozycje i czasy w metrach. Reguły jawne i konfigurowalne (progi w configu). To jest **źródło etykiet bez ręcznego tagowania**.
*Akceptacja:* ręczna kontrola próbki zdarzeń (precyzja/czułość raportowane).

**M10 `m10_options`** — dla każdego „momentu decyzji” (zawodnik z piłką) wygeneruj kandydatów = widoczni koledzy z drużyny i policz cechy:
- odległość i kąt podania względem wektora ruchu podającego,
- wektor prędkości i **przyspieszenia** odbiorcy oraz jego kierunek względem linii podania (czy wybiega na piłkę),
- czystość linii podania (min. odległość rywala do linii, czas dobiegu rywala do przechwytu),
- *pitch control* w punkcie odbioru,
- postęp w kierunku bramki (ΔX),
- presja na odbiorcy (odległość i prędkość zbliżania najbliższego rywala),
- maska widoczności: kandydaci i rywale spoza kadru **muszą być jawnie oznaczeni** (obserwacja częściowa), nie „zerowani”.
*Akceptacja:* testy syntetyczne (znane układy → oczekiwane cechy), wizualizacja opcji na minimapie.

**M11 `m11_learn`** — uczenie wag (sekcja 8).

**M12 `m12_tags`** — prosty magazyn tagów (klip: start/koniec, tag, komentarz) w Parquet/JSON + eksport etykiet do M11; lista tagów konfigurowalna (pressing, dobra wrzutka, „nie zauważył presji”, odbiór po przyspieszeniu…).

## 8. Jak system „sam uczy się wag” (założenia i ograniczenia)

**Zasada:** etykiety biorą się automatycznie z samego meczu (M9), a nie z ręcznego opisu każdej akcji.

1. **Zadanie samonadzorowane: przewidywanie odbiorcy podania.** Dla każdego momentu decyzji model zwraca rozkład po kandydatach; etykietą jest partner, który faktycznie dostał piłkę.
2. **Model startowy: conditional logit (softmax po kandydatach)** z cechami z M10 — kilkanaście parametrów, w pełni interpretowalne wagi (raport: które cechy ważą najwięcej, przedziały ufności z bootstrapu). Implementacja w PyTorch z regularyzacją L2. Interfejs `Scorer` tak, żeby później podmienić na MLP/GNN bez zmiany reszty.
3. **Drugi cel: wynik podania** (udane/przechwycone) → wagi „ryzyka” linii podania.
4. **Tagi użytkownika (M12)** jako dodatkowe, ręczne etykiety (np. wykrywanie presji/„nie zauważył”), uczone osobnym klasyfikatorem na cechach kinematycznych (start: reguły + regresja logistyczna).
5. **Pretraining na otwartych danych trackingowych** (kloppy: Metrica/SkillCorner itd.) → inicjalizacja wag, potem dostrojenie na danych z transmisji. Uwaga na *domain shift*: dane z TV mają niepełną widoczność i szum kalibracji.

**Ograniczenia, które agent ma zapisać w raporcie i nie ukrywać:**
- 10 skrótów to najwyżej setki podań. Wagi będą niepewne; raportuj przedziały ufności i pokazuj krzywą „jakość vs liczba klipów”. Pipeline ma umożliwiać dokładanie kolejnych meczów bez zmian w kodzie.
- **xG nie da się uczciwie nauczyć ze skrótów** — są zbiorem stronniczym (głównie gole i sytuacje). xG trenuj/kalibruj na otwartych danych strzałów, a skróty użyj tylko do walidacji cech geometrycznych (kąt, odległość, zawodnicy na linii strzału).
- Skrót TV nie pokazuje wszystkich zawodników; każdy wynik zależy od widoczności i jakości kalibracji — raport zawsze zawiera odsetek klatek `ok/tracked`.

## 9. CLI i przepływ

```
pitchlab shots    <video>                     # M1
pitchlab calib    <video> --backend pnlcalib  # M2+M3
pitchlab footprint <video>                    # M4 (+wideo kontrolne)
pitchlab eval     --backends roboflow,pnlcalib,snbaseline   # M5
pitchlab track    <video>                     # M6+M7
pitchlab world    <video>                     # M8
pitchlab events   <match>                     # M9
pitchlab options  <match>                     # M10
pitchlab learn    --matches <m1,m2,...>       # M11
pitchlab tag      <match>                     # M12
pitchlab run-all  <video>                     # cały łańcuch dla jednego skrótu
```

Każda komenda wznawialna (pomija gotowe wyniki, chyba że `--force`), loguje wersje modeli/wag i hash configu w metadanych wyjścia.

## 10. Kolejność pracy i definicja „gotowe”

1. **M0** + szkielet CLI + testy szablonu boiska.
2. **M2** z backendem Roboflow + **M4** → pierwsza minimapa na jednym skrócie (demo w pierwszym dniu pracy).
3. **M5** (zbiór testowy + metryki) — bez tego nie idziemy dalej.
4. Dołożenie PnLCalib i baseline'u SoccerNet, raport porównawczy, wybór backendu → **bramka etapu 1**.
5. **M1** i **M3** (zmiany kamery, zbliżenia, propagacja).
6. **M6 → M7 → M8**, po każdym raport jakości.
7. **M9 → M10**, potem **M11** i **M12**.
8. Na końcu (opcjonalnie): Dockerfile i uruchamianie zadań wsadowych na GCP (GCE lub Vertex AI Custom Job) dla wielu skrótów naraz.

**Definicja „gotowe” dla każdego modułu:** kod + testy zielone + CLI + krótki wpis w `docs/RUNBOOK.md` + wynik na zbiorze kontrolnym w raporcie + wpis o licencjach zależności.

## 11. Protokół pracy agenta

- Przed startem modułu: przeczytaj jego sekcję, wypisz w 5 punktach plan implementacji i pytania blokujące (tylko realne blokady).
- Po module: uruchom testy, wygeneruj wideo/raport kontrolny, podsumuj wynik metryk i ryzyka w `docs/DECISIONS.md`.
- Gdy metryka bramki nie jest spełniona — **zatrzymaj się i zgłoś**, nie przechodź do kolejnego etapu.
- Nie zmieniaj kontraktów danych (sekcja 6) bez zapisu w `docs/DECISIONS.md` i podbicia `schema_version`.
