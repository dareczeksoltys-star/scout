# PitchLab: System Matematycznej i Przestrzennej Analizy Wideo Piłki Nożnej

System do precyzyjnej ekstrakcji danych matematycznych i trajektorii z materiałów wideo meczów piłkarskich. Architektura modularna zgodna z modelem `pitchlab` (Moduł 1: przycinanie i próbkowanie klatek wideo, Moduł 2: lokalna bezkosztowa ekstrakcja pozycji zawodników i piłki na boisko 2D, Moduł 3: kontrakt danych dla analityki taktycznej i wektorów ruchu).

---

### Decyzje Architektoniczne i Podsumowanie Ustalenia

> [!IMPORTANT]
> **Bezkosztowa lokalna analiza CV (Zgodnie z informacją od użytkownika):**
> Z uwagi na koszty Gemini Vision API, silnik Modułu 2 opiera się na **w 100% lokalnych, darmowych algorytmach wizji komputerowej** uruchamianych w przeglądarce (Canvas Image Processing, segmentacja barwna strojów, homografia rzutowania perspektywicznego na boisko 105×68m oraz śledzenie plam ruchu/obiektów). Opcjonalnie udostępniony jest przełącznik mikro-weryfikacji wybranej pojedynczej klatki przez Gemini Flash (tylko na żądanie użytkownika).

- **Moduł 1 (Krojenie wideo):** Odtwarzacz z zakresem znaczników In/Out, suwakiem osi czasu, próbkowaniem klatek (domyślnie co 60 klatek z możliwością regulacji) oraz generowaniem zestawu klatek ze znacznikami czasu (timecodes mm:ss.ms) i indeksem klatki.
- **Moduł 2 (Ekstrakcja liczbowa):** Wykrywanie współrzędnych pikselowych $[x, y, w, h]$ zawodników, sędziego i piłki, grupowanie zespołów po barwach koszulek, transformacja homograficzna na standardowe wymiary boiska (105m × 68m) oraz eksport danych liczbowych do JSON i CSV.
- **Kontrakt Modułu 3:** Formalna specyfikacja wejść/wyjść dla zaawansowanej matematyki meczowej (wektory prędkości, diagramy Woronoja kontroli przestrzeni, mapy ciepła, centroidy drużyn).

---

### 1. Przegląd i Koncepcja Narzędzia

Aplikacja przekształca surowy plik wideo meczu (lub wybrany fragment) w ustrukturyzowany zbiór danych matematycznych o trajektoriach i pozycjach obiektów sportowych na boisku.

- **Dla kogo:** Analitycy taktyczni, skauci piłkarscy, trenerzy oraz programiści modeli ML trenujący algorytmy na wyekstrahowanych sekwencjach współrzędnych.
- **Główna wartość:** Natychmiastowe pozyskanie danych metrycznych bez konieczności płatnych subskrypcji API w chmurze — pełne działanie w przeglądarce z opcją zapisu klatek do ZIP oraz danych do JSON/CSV.

---

### 2. Doświadczenie Użytkownika i Design Wizualny

Interfejs klasy SaaS Analytics / Data Console utrzymany w profesjonalnej estetyce sportowej technologii wideo:
- **Układ:** Przestrzeń robocza podzielona na logiczne moduły z paskiem nawigacyjnym i przełączaniem kroków analizy (`Moduł 1: Wideo & Klatki` → `Moduł 2: Ekstrakcja Danych CV` → `Moduł 3: Kontrakt & Metryki`).
- **Paleta barw:** Neutralne ciemne tło analityczne (Slate `#0B0F19` / `#111827`), chłodne szarości paneli roboczych, akcent boiskowy szmaragdowy (`#10B981`) dla oznaczeń i murawy, akcenty kontrastowe dla drużyn (np. cyjan vs koral) oraz piłki (złoty `#F59E0B`).
- **Typografia:** `Plus Jakarta Sans` dla nagłówków i etykiet, `JetBrains Mono` / `tabular-nums` dla wszystkich współrzędnych, znaczników czasu, prędkości i macierzy liczb.
- **Moduł 1 — Przepływ Wideo:**
  - Podgląd wideo z możliwością załadowania dołączonego pliku demonstracyjnego (bramki meczowe z promptu) lub wgrania własnego pliku (MP4/WebM).
  - Pasek wyboru zakresu czasowego z uchwytami `Marker IN` i `Marker OUT`.
  - Kontrolka interwału klatek (domyślnie `60 klatek` / ~2 sekundy przy 30fps, z regulacją 10–120 klatek).
  - Taśma wygenerowanych miniatur klatek z precyzyjnymi znacznikami `00:04.120 (klatka #120)`.
  - Przycisk pobrania paczki klatek ZIP oraz przejścia do analizy w Module 2.
- **Moduł 2 — Przepływ Ekstrakcji Danych:**
  - Widok klatki roboczej z nakładkami detekcji (bounding boxy, ID śledzenia, etykiety Team A / Team B / Sędzia / Piłka).
  - Obok: **Radar Boiska 2D (105m × 68m)** pokazujący zrzutowane pozycje zawodników w rzucie z góry w czasie rzeczywistym.
  - Tabela liczbowa ze współrzędnymi $(x, y)$ pikseli oraz $(X, Y)$ w metrach na boisku.
  - Możliwość manualnej korekty / kalibracji 4 punktów narożnych boiska (homografia).
  - Eksport ustrukturyzowanego pliku `match_tracking_data.json` i `positions.csv`.
- **Moduł 3 — Podgląd Metryk i Kontrakt API:**
  - Podgląd wyliczanych wartości matematycznych: prędkości chwilowe, rozstaw linii obrony/ataku, centroidy drużyn, poligon kontroli boiska.
  - Interaktywna dokumentacja schematu danych (JSON Schema) ułatwiająca integrację kolejnych modułów AI/Python.

---

### 3. Kluczowe Decyzje Architektoniczne i Kompromisy

1. **Bezkosztowa lokalna analiza CV vs Płatne API LLM:**
   - *Decyzja:* Pełna implementacja lokalnego potoku analizy obrazu w HTML5 Canvas / JS (filtry barwne HSV murawy, wykrywanie konturów obiektów na zielonym tle, progowanie Otsu/adaptive, algorytm k-means dla strojów oraz homograficzny rzut perspektywiczny na płaszczyznę 2D boiska).
   - *Dlaczego:* Zero kosztów, natychmiastowe działanie, prywatność danych wideo i brak limitów API. Dodatkowo zapewniamy opcjonalny przycisk "Gemini Flash Assist" dla chętnych, którzy chcą porównać wyniki.
2. **Standard Współrzędnych:**
   - *Decyzja:* Każdy wykryty obiekt posiada jednocześnie:
     1. Współrzędne klatki w pikselach `[bbox_x, bbox_y, bbox_w, bbox_h]` i znormalizowane `[0..1]`.
     2. Współrzędne taktyczne na standardowym boisku FIFA `[pitch_x, pitch_y]` w metrach (0..105, 0..68).
   - *Dlaczego:* Umożliwia to zarówno rysowanie nakładki na wideo, jak i bezpośrednie liczenie dystansów, wektorów prędkości w m/s oraz diagramów taktycznych.

---

### 4. Architektura Techniczna i Schemat Przepływu Danych

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ARCHITEKTURA PITCHLAB                           │
└────────────────────────────────────────────────────────────────────────┘

 [ Plik Wideo (Upload / Demo) ]
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  MODUŁ 1: VIDEO INGEST & FRAME SLICER                                 │
│  - Video Element & Canvas Frame Grabber                                │
│  - Zakres czasowy [Time In, Time Out]                                  │
│  - Krok próbkowania: co 60 klatek (lub własny)                         │
│  - Generowanie manifestu JSON ze znacznikami czasu                     │
│  - Eksport klatek do ZIP / Pamięci RAM                                 │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                  Klatki [ImageBitmap/DataURL + Metadane]
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  MODUŁ 2: DETECT & TRACK ENGINE (Lokalne CV + Homografia)              │
│  - Pitch Boundary & Field Masking (izolacja murawy)                   │
│  - Blob & Motion Tracking (zawodnicy, sędzia, piłka)                  │
│  - Color Clustering K-Means (przypisanie drużyn po koszulkach)         │
│  - Macierz Homografii H -> Rzut perspektywy na boisko 2D (105x68m)     │
│  - Interpolator i kalibracja pozycji                                   │
│  - Output: match_tracking_data.json & positions.csv                    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                     Znormalizowany JSON pozycji
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  MODUŁ 3: SYSTEM MATEMATYCZNY & KONTRAKT ANCHOR                        │
│  - Wektory prędkości (|v|, vx, vy w m/s)                               │
│  - Centroidy drużyn i zwartość formacji (compactness)                  │
│  - Matematyczny poligon kontroli przestrzeni (Voronoi)                 │
│  - Sformalizowany interfejs JSON Schema dla przyszłych agentów AI      │
└────────────────────────────────────────────────────────────────────────┘
```

#### Szczegółowy Kontrakt Danych (JSON Schema)

```typescript
interface FrameDetectionOutput {
  frameIndex: number;
  timestampSeconds: number;
  timecode: string; // "00:02.000"
  dimensions: { width: number; height: number };
  pitchDimensions: { lengthMeters: 105; widthMeters: 68 };
  ball: {
    frameX: number;
    frameY: number;
    pitchX: number; // 0..105 metrów
    pitchY: number; // 0..68 metrów
    confidence: number;
    isDetected: boolean;
  };
  players: Array<{
    trackId: string;
    team: 'home' | 'away' | 'referee' | 'unknown';
    jerseyColorHex: string;
    bbox: [number, number, number, number]; // [x, y, w, h] w pikselach
    pitchX: number; // w metrach
    pitchY: number; // w metrach
    velocityX?: number; // m/s (dla kolejnych klatek)
    velocityY?: number; // m/s
    speedKmh?: number;
  }>;
}
```

---

### Plan Realizacji Krok po Kroku

1. **Konfiguracja Projektu i Nazwy:**
   - Nadanie nazwy `PitchLab - Soccer Video Spatial Analytics` w `metadata.json` i `index.html`.
2. **Budowa Modułu 1 (Wideo & Krojenie klatek):**
   - Komponent odtwarzacza wideo HTML5 z precyzyjnym pozycjonowaniem klatek (`currentTime`).
   - Kontrolki zakresu `Trim IN` / `Trim OUT`.
   - Próbnik klatek Canvas z suwakiem interwału (domyślnie 60 klatek).
   - Generowanie galerii klatek z oznaczeniami czasu i możliwością pobrania paczki ZIP.
3. **Budowa Modułu 2 (Silnik Detekcji CV i Rzutowania 2D):**
   - Lokalny detektor Canvas: wykrywanie obiektów na boisku, bounding boxy, wyznaczanie środka masy stóp gracza.
   - Moduł kolorystyczny strojów (analiza próbki z górnej części sylwetki).
   - Moduł homografii: rzutowanie perspektywiczne współrzędnych obrazu na płaszczyznę 2D boiska piłkarskiego 105×68m.
   - Interaktywny radar taktyczny 2D obok wideo oraz tabela danych liczbowych.
   - Eksport danych do plików JSON i CSV.
4. **Budowa Modułu 3 (Kalkulator Metryk i Kontrakt Przyszłościowy):**
   - Wyliczanie prędkości pomiędzy próbkami co 60 klatek ($v = \Delta s / \Delta t$).
   - Wyliczanie środka ciężkości obu drużyn (centroid) oraz szerokości/długości formacji.
   - Zakładka dokumentacji kontraktu z podglądem na żywo schematu JSON dla dalszych integracji AI/Python.
5. **Weryfikacja i Testy:**
   - Uruchomienie `compile_applet` i weryfikacja stabilności kodu.
