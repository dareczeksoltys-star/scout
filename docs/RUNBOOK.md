# PITCHLAB — Podręcznik Operacyjny (RUNBOOK.md)

Instrukcje uruchamiania, konfiguracji środowiska i obsługi CLI w trybach `dev` i `gpu`.

---

## 1. Wymagania wstępne i konfiguracja środowiska

### Środowisko lokalne (tryb `dev`)
- Python 3.11
- Wirtualne środowisko (venv / uv / poetry)
- Instalacja zależności bazowych:
  ```bash
  pip install -e .
  ```
- Uruchomienie testów:
  ```bash
  pytest -v
  ```

---

## 2. Podstawowe komendy CLI

```bash
# Etap 1: Kalibracja i footprint
pitchlab shots <video_path>                     # M1: Podział na ujęcia i klasyfikacja widoku
pitchlab calib <video_path> --backend roboflow  # M2+M3: Kalibracja boiska
pitchlab footprint <video_path>                 # M4: Rzutowanie i generacja wideo kontrolnego
pitchlab eval --backends roboflow,pnlcalib      # M5: Zestawienie metryk kalibracji

# Etap 2: Detekcja, tracking, świat w metrach
pitchlab track <video_path>                     # M6+M7: Detekcja, śledzenie i przypisanie drużyn
pitchlab world <video_path>                     # M8: Projekcja do metrów i kinematyka

# Etap 3: Zdarzenia, taktyka, uczenie
pitchlab events <match_id>                      # M9: Detekcja posiadania i podań
pitchlab options <match_id>                     # M10: Opcje podań i pitch control
pitchlab learn --matches m1,m2                  # M11: Uczenie wag wyboru podań
pitchlab tag <match_id>                         # M12: Obsługa ręcznych tagów

# Cały pipeline dla jednego skrótu:
pitchlab run-all <video_path>
```

---

## 3. Integracja z Google Cloud Platform

### Zmienne środowiskowe
- `PITCHLAB_ENV`: `dev` lub `gpu` (domyślnie wykrywane automatycznie)
- `GCP_PROJECT_ID`: ID projektu w Google Cloud
- `GCS_BUCKET`: Nazwa bucketa GCS (np. `pitchlab-data`)
- `GOOGLE_APPLICATION_CREDENTIALS`: Ścieżka do pliku klucza Service Account (w środowisku deweloperskim) lub domyślne uwierzytelnianie GCP (Workload Identity w chmurze).
