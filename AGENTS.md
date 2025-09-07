Nach dem nach dem Programmieren, ändern, korrigieren, ergänzen des Codes sollst du eine saubere, reproduzierbare Qualitäts-Pipeline fahren – schnell zuerst, tiefgründig danach:
Also immer vor dem Abschluss!
1. Schnelle Auto-Korrektur & Formatierung

Ruff (lint + quick-fix + Imports):
ruff check . --fix
Fängt Syntax-/Style-Fehler (E/F/W …), sortiert Importe (Regelgruppe I), kann auch Modernisierungen (UP/pyupgrade-Regeln) und Docstring-Checks (D) übernehmen – extrem schnell. 

Black (finale Formatierung):
black .
Erzwingt einheitlichen Stil, keine Diskussionen über Formatdetails. (Black liest optional deine pyproject.toml-Settings.) 

Optional isort: nur wenn du nicht willst, dass Ruff die Importe sortiert: isort . (Ruff kann das bereits.) 



2. Statische Typprüfung (zwei Perspektiven, mehr Treffer)

basedpyright (strenger, sehr schnell): basedpyright
Vorteil: reines PyPI-Paket, keine Node-Abhängigkeit. 

mypy (zweite Meinung, anderes Regelset): mypy
Beide Checker finden unterschiedliche Klassen von Typ-Fehlern; Konfiguration bequem in pyproject.toml unter [tool.mypy]. 



3. Tiefenanalyse der Codequalität

Pylint: pylint src/
Deckt Naming-Konventionen, Code-Smells, Refactoring-Hinweise ab; findet anderes als Ruff. Pylint kann pyproject.toml lesen. 

Vulture: vulture src/
Sucht toten/ungenutzten Code (Funktionen, Variablen, Klassen). Konfigurierbar via [tool.vulture] in pyproject.toml. 



4. Abhängigkeits-Hygiene

Deptry: deptry .
Findet fehlende, ungenutzte, transitive und falsch genutzte Stdlib-Im-porte; versteht PEP 621-Metadaten aus pyproject.toml ([project], [project.optional-dependencies]). Konfigurierbar via [tool.deptry]. 



5. Dokumentationsdisziplin

pydocstyle: pydocstyle src/
Prüft PEP 257-konforme Docstrings. (Alternativ kannst du die D-Regeln über Ruff laufen lassen und pydocstyle weglassen.) 




---

Minimal sinnvolle Reihenfolge (CLI)

# 1) Schnell fixen & formatieren
ruff check . --fix
black .

# 2) Typchecks
basedpyright
mypy

# 3) Tiefenanalyse
pylint src/
vulture src/

# 4) Dependencies prüfen
deptry .

# 5) (optional, falls nicht via Ruff-D-Regeln)
pydocstyle src/
# DexiNed → SD 1.5 + ControlNet Pipeline - Implementierungsstatus

## 1. ARCHITEKTUR-REFACTORING
- [x] Monolithischen Code in `main.py` (reiner GUI-Entrypoint) und `src/pipeline.py` (komplette Pipeline-Logik) aufteilen (main.py + pipeline.py erstellt)
- [x] Alle DexiNed-Funktionen in pipeline.py verschieben (load_dexined, get_dexined, rescale_edge) (nach pipeline.py migriert)
- [x] Alle SD/ControlNet-Funktionen in pipeline.py verschieben (load_sd15_lineart, sd_refine) (nach pipeline.py migriert)
- [x] Bildverarbeitungs-Utilities in pipeline.py kapseln (resize_img, ensure_rgb, postprocess_lineart) (Utility-Funktionen hinzugefügt)
- [x] Process-Funktionen (process_one, process_folder) in pipeline.py implementieren (aus GUI ausgelagert)
- [x] GUI-Code komplett in main.py belassen, nur Imports aus pipeline.py nutzen (main.py nutzt pipeline)
- [x] Globale Variablen (_dexi, _pipe) durch Singleton-Pattern oder Lazy-Loading ersetzen (lru_cache verwendet)

## 2. PERFORMANCE-OPTIMIERUNGEN
- [x] `torch.inference_mode()` Kontext-Manager für alle Modell-Inferenzen einbauen
- [x] `torch.autocast()` mit device-spezifischen dtype (fp16 für CUDA, fp32 für CPU) implementieren
- [x] Device-Detection-Funktion schreiben: erst CUDA prüfen, dann MPS (Apple), dann CPU-Fallback
- [x] Memory-Settings für SD-Pipeline: `pipe.enable_attention_slicing(slice_size="auto")`
- [x] VAE-Optimierungen: `pipe.enable_vae_slicing()` und `pipe.enable_vae_tiling()`
- [x] CPU-Offloading aktivieren: `pipe.enable_model_cpu_offload()` wenn CUDA verfügbar
- [x] Sequential CPU-Offloading für low-memory: `pipe.enable_sequential_cpu_offload()`
- [x] Deterministische Seeds: `torch.Generator(device).manual_seed(seed)` für jeden Inference-Call
- [x] Dtype-Detection implementieren: `torch.float16` für CUDA, `torch.bfloat16` für neuere CPUs, sonst `torch.float32`
- [x] Batch-Processing für mehrere Bilder gleichzeitig (Mini-Batches mit OOM-Fallback)

## 3. THREAD-SAFETY UND GUI-STABILITÄT
- [x] Logging-Setup in main.py: `logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')`
- [x] Logger-Instanzen für jedes Modul: `logger = logging.getLogger(__name__)`
- [x] Queue für Thread-safe GUI-Updates: `self.log_queue = queue.Queue()`
- [x] Log-Queue-Processor als Timer-Event (alle 100ms): `self.after(100, self.process_log_queue)`
- [x] Threading.Event für Stop-Signal: `self.stop_event = threading.Event()`
- [x] Stop-Event in process_folder-Loop prüfen: `if self.stop_event.is_set(): break`
- [x] Button-States korrekt verwalten: Start/Stop/Prefetch disabled während Operationen
- [x] Alle langläufigen Operationen in daemon-Threads: `threading.Thread(target=job, daemon=True).start()`
- [x] Progress-Updates über Queue senden: aktuelle Bildnummer, Gesamtanzahl, Dateiname
- [x] GUI-Updates nur im Main-Thread via `self.after(0, lambda: widget.config(...))`

## 4. FEHLERBEHANDLUNG UND ROBUSTHEIT
- [x] Try-except um Modell-Downloads mit spezifischen Fehlermeldungen (ConnectionError, HTTPError, OSError) ✅
- [x] Bildvalidierung: Mindestgröße 64x64, Maximalgröße 4096x4096, unterstützte Formate (.jpg, .png, .webp)
- [x] Korrupte Bilder abfangen: try-except um PIL.Image.open() mit verify()
- [x] Schreibrechte prüfen: `os.access(output_dir, os.W_OK)` vor Verarbeitung
- [x] OOM-Handling: except torch.cuda.OutOfMemoryError mit Hinweis auf Bildgröße/Batch-Size ✅
- [x] Netzwerk-Fallback: Bei Download-Fehler auf lokale Modell-Pfade hinweisen ✅
- [x] CUDA-Verfügbarkeit: Explizite Meldung wenn nur CPU verfügbar (Warnung vor langer Laufzeit)
- [x] Disk-Space-Check: Prüfen ob genug Speicherplatz für Outputs vorhanden
- [x] Modell-Loading-Fallbacks: Bei Fehler alternative Modell-IDs oder lokale Pfade versuchen
- [x] Graceful Shutdown: Ressourcen in finally-Blöcken freigeben, Modelle aus VRAM entladen

## 5. CODE-QUALITÄT
- [x] Type Hints für ALLE Funktionsparameter: `def function(param: str, number: int) -> Optional[Path]:` ✅
- [x] Type Hints für Rückgabewerte, auch bei None: `-> None` ✅
- [x] Google-Style Docstrings mit Args, Returns, Raises Sections für jede Funktion
- [x] Klassen-Docstrings mit Attributes-Section für alle Instance-Variablen
- [x] Black-Formatierung: Zeilen max 88 Zeichen, konsistente Quotes ✅
- [x] Ruff-Linting: Import-Sortierung, unused imports entfernen ✅
- [x] Pathlib überall: keine String-Pfade, immer `Path` objects
- [x] Konstanten in UPPER_CASE am Dateianfang definieren
- [x] Magic Numbers durch benannte Konstanten ersetzen (z.B. DEFAULT_STEPS = 32)
- [x] F-Strings statt .format() oder %-Formatierung verwenden

## 6. DOKUMENTATION UND USABILITY
- [x] README.md Struktur: Installation → Quickstart → Features → Parameter → Troubleshooting → Lizenz
- [ ] Screenshots der GUI mit Annotationen für jeden Bereich einfügen
- [x] Modell-Links mit exakten Hugging Face URLs: `lllyasviel/Annotators`, `lllyasviel/control_v11p_sd15_lineart`
- [x] Lizenz-Section: CreativeML OpenRAIL-M für SD 1.5, Apache 2.0 für ControlNet
- [x] requirements.txt aufräumen: torch, diffusers, transformers, accelerate, controlnet-aux, pillow, opencv-python, scikit-image, numpy, xformers, vtracer
- [x] Python-Version spezifizieren: >=3.8,<3.12 (wegen Dependencies)
- [x] Troubleshooting-FAQ: CUDA nicht gefunden, OOM-Errors, langsame CPU-Inferenz, Modell-Download-Fehler
- [x] Beispiel-Input/Output Bilder im `examples/` Ordner
- [x] Default-Werte dokumentieren und begründen (z.B. warum Steps=32)
- [ ] Performance-Benchmarks: Zeiten für verschiedene Bildgrößen auf verschiedenen GPUs

## 7. GUI-SPEZIFISCHE VERBESSERUNGEN
- [x] Preset-Buttons mit Lambda-Functions: Quick (16 steps), Standard (32), Quality (50), Technical (40)
- [x] Tooltips via `CreateToolTip` Klasse für jeden Parameter mit Erklärung und Wertebereich (Implementiert)
- [x] ttk.Progressbar mit determinate mode: Maximum = Anzahl Bilder, Update nach jedem Bild
- [ ] tkinterdnd2 für Drag&Drop oder Fallback auf Browse-Button
- [x] Settings in JSON speichern: `~/.dexined_pipeline/settings.json` mit last_input, last_output, parameters (Erledigt)
- [ ] Status-Icons: ✓ für fertig, ⚡ für processing, ❌ für Fehler, ⏸ für pausiert
- [ ] Recent-Folders ComboBox: Letzte 10 verwendete Ordner speichern und anzeigen
- [x] Parameter-Gruppen in LabelFrames: "Eingabe/Ausgabe", "Qualität", "Performance", "Erweitert" (GUI strukturiert)
- [ ] Tastenkürzel: Ctrl+O (Open), Ctrl+S (Start), Ctrl+Q (Quit), ESC (Stop)
- [x] Statusleiste am unteren Rand: Aktuelles Bild, Geschwindigkeit (Bilder/Min), geschätzte Restzeit (Anzeige aktiv)

## 8. ZUSÄTZLICHE FEATURES
- [ ] Thumbnail-Grid mit tkinter.Canvas: 100x100 px Vorschauen der verarbeiteten Bilder
- [ ] Batch-Size Spinbox: 1-8 Bilder parallel (mit VRAM-Warnung)
- [x] Output-Ordner mit Zeitstempel: `output_2024-01-15_14-30-45/` (Standardpfad)
- [ ] SHA256-Checksum für Modelle nach Download mit gespeicherten Hashes vergleichen
- [ ] Pause/Resume via threading.Event: pause_event zusätzlich zu stop_event
- [ ] Bildstatistiken anzeigen: Anzahl verarbeitet, Durchschnittszeit, Erfolgsrate
- [ ] Export-Optionen: PNG, JPG (mit Qualität-Slider), WebP
- [ ] Vergleichsansicht: Original vs. DexiNed vs. Refined nebeneinander
- [ ] Undo-Funktion: Letzte Batch-Operation rückgängig machen (Dateien löschen)
- [ ] GPU-Memory-Monitor: Aktuelle VRAM-Nutzung in MB anzeigen

## 9. TESTING UND VALIDIERUNG
- [ ] Testbilder verschiedener Größen: 256x256, 512x512, 1024x1024, 2048x2048
- [ ] Edge-Cases testen: 1x1 Pixel, 10000x10000 Pixel, korrupte Dateien
- [ ] Verschiedene Formate: JPG, PNG, WebP, BMP, GIF (nur erstes Frame)
- [ ] Speicher-Monitoring während Batch-Verarbeitung (keine Memory Leaks)
- [ ] Threading-Tests: Start/Stop/Pause in schneller Folge
- [ ] GUI-Responsiveness: Bleibt UI während 100+ Bilder Batch reaktiv?
- [ ] Cross-Platform: Windows 10/11, Ubuntu 22.04, macOS (wenn MPS verfügbar)
- [ ] Performance-Messung: FPS bzw. Bilder pro Minute dokumentieren
- [ ] Fehler-Recovery: Kann nach Crash/Exception fortgesetzt werden?
- [ ] Modell-Cache: Werden Modelle korrekt zwischengespeichert?

## Nachweise

| ID | Punkt | Status | Nachweis |
|----|-------|--------|---------|
| AGENT-201 | torch.inference_mode & autocast | ✅ | 2be2d20, tests/test_utils.py |
| AGENT-301 | Logging & Queue in GUI | ✅ | 2be2d20 |
| AGENT-401 | Bildvalidierung & Schreibrechte | ✅ | 2be2d20 |
| AGENT-601 | README Struktur & Troubleshooting | ✅ | 2be2d20 |
| AGENT-402 | Modell-Download-Fehler behandeln | ✅ | 53bc7a6, tests/test_errors.py |
| AGENT-403 | OOM-Handling im SD-Refine | ✅ | 53bc7a6, tests/test_errors.py |
| AGENT-501 | Typisierung & Black-Format | ✅ | 53bc7a6 |
