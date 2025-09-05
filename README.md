# Dexi LineArt Max

## Installation
```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart
```
python main.py
```

## Features
- DexiNed edge detection
- SD 1.5 + ControlNet lineart refinement
- Optional SVG export via VTracer

## Parameter
- Steps (default 32)
- Guidance (6.0)
- Ctrl-Scale (1.0)
- Strength (0.70)
- Seed (42)
- Max lange Kante (896px)

## Troubleshooting
- CUDA nicht gefunden: Installation von passenden Treibern prüfen.
- OOM-Error: Bildgröße oder Batch reduzieren.
- Langsame CPU: GPU nutzen oder Geduld haben.
- Modell-Download-Fehler: Modelle manuell in den Cache legen.

## Lizenz
- Stable Diffusion 1.5: CreativeML OpenRAIL-M
- ControlNet Lineart: Apache-2.0
