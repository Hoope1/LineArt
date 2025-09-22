# Dexi LineArt Max

*Benötigt Python >=3.10,<3.13*

## Installation
```
python -m venv .venv
. .venv/bin/activate
pip install .[test]
```

## Quickstart
```
python main.py
```

## Entwicklung

Ruff ersetzt Flake8, Black, isort und Pylint; BasedPyright löst mypy ab. Vulture, Refurb und Deptry prüfen zusätzlich auf toten Code, Modernisierungspotenzial und saubere Abhängigkeiten.

### Setup (Windows 11, Python ≥ 3.10)

```
py -3.10 -m venv .venv
.\.venv\Scripts\activate
pip install .[dev,test]
```

Die Entwicklungs-Extras installieren Ruff, BasedPyright, Vulture, Refurb und Deptry; das Test-Extra bringt Pytest sowie Tomli für Python 3.10 mit.

### Prüfbefehle

```
ruff format .
ruff check . --fix
basedpyright --outputjson
vulture src tests --ignore-names 'use_sd,save_svg'
refurb src tests
deptry .
```

Alle Schritte lassen sich gebündelt ausführen:

```
make full-check
```

### Beispielausgabe

```
$ make full-check
ruff format .
ruff check . --fix
basedpyright --outputjson
vulture src tests --ignore-names 'use_sd,save_svg'
refurb src tests
deptry .
  Success! No dependency issues found.
```

### Lokale Hooks

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

### CI (report-only)

Der GitHub-Workflow `lint-advisory` führt die gleichen Checks aus, liefert aber nur Berichte als Artefakte und in der Job-Zusammenfassung. Branch-Protection-Regeln dürfen diesen Workflow nicht als "required" markieren; die Durchsetzung passiert lokal über die oben genannten Hooks bzw. `make full-check`.

## Features
- DexiNed edge detection
- SD 1.5 + ControlNet lineart refinement
- Optional SVG export via VTracer
- Automatische VRAM-Erkennung für GPUs mit <6 GB passt Auflösung und Offloading an
- Persistente Einstellungen in `~/.dexined_pipeline/settings.json`
- Statusleiste mit Dateiname, Fortschritt und ETA
- Standard-Ausgabeordner mit Zeitstempel `output_YYYY-MM-DD_HH-MM-SS`
- Drag & Drop für Eingabe- und Ausgabepfade (optional via tkinterdnd2)

## GUI

```
+---------------------------------------+
| Input Dir  [Browse] [Start] [Stop]    |
| Output Dir [Browse]                   |
|                                       |
| Steps: [32] Guidance: [6.0]           |
| Ctrl-Scale: [1.0] Strength: [0.70]    |
| Seed: [42] Max Edge: [896]            |
|                                       |
| [ Progress bar ------------------- ]  |
| Status: Idle                          |
+---------------------------------------+
```

## Modelle
- [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators) – DexiNed
- [lllyasviel/control_v11p_sd15_lineart](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart) – ControlNet

## Parameter
- Steps (default 32) – ausgewogen zwischen Qualität und Geschwindigkeit
- Guidance (6.0) – höhere Werte erzwingen stärkere Prompt-Treue
- Ctrl-Scale (1.0) – Einfluss der ControlNet-Linien
- Strength (0.70) – Stärke des Img2Img-Effekts
- Seed (42) – Reproduzierbarkeit
- Max lange Kante (bis 896px, 640px bei <5 GB VRAM) – begrenzt Rechenaufwand
- Batch-Size (1) – Anzahl Bilder pro Durchlauf; bei VRAM-Engpässen automatische Reduktion

## Beispiele
Beispiel-Eingabe und -Ausgabe befinden sich im Ordner `examples/`.

Da Binärdateien im Repository nicht enthalten sind, siehe `examples/README.md` für Downloadlinks zu Beispielbildern.

## Troubleshooting
- CUDA nicht gefunden: Installation von passenden Treibern prüfen.
- OOM-Error: Bildgröße oder Batch reduzieren.
- Langsame CPU: GPU nutzen oder Geduld haben.
- Modell-Download-Fehler: Modelle manuell in den Cache legen.

## Lizenz
- Stable Diffusion 1.5: CreativeML OpenRAIL-M
- ControlNet Lineart: Apache-2.0
