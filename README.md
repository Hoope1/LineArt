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

## Modelle
- [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators) – DexiNed
- [lllyasviel/control_v11p_sd15_lineart](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart) – ControlNet

## Parameter
- Steps (default 32) – ausgewogen zwischen Qualität und Geschwindigkeit
- Guidance (6.0) – höhere Werte erzwingen stärkere Prompt-Treue
- Ctrl-Scale (1.0) – Einfluss der ControlNet-Linien
- Strength (0.70) – Stärke des Img2Img-Effekts
- Seed (42) – Reproduzierbarkeit
- Max lange Kante (896px) – begrenzt Rechenaufwand

## Troubleshooting
- CUDA nicht gefunden: Installation von passenden Treibern prüfen.
- OOM-Error: Bildgröße oder Batch reduzieren.
- Langsame CPU: GPU nutzen oder Geduld haben.
- Modell-Download-Fehler: Modelle manuell in den Cache legen.

## Lizenz
- Stable Diffusion 1.5: CreativeML OpenRAIL-M
- ControlNet Lineart: Apache-2.0
