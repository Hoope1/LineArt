# Installation (Windows 11, Python ≥ 3.10)

```bash
python -m venv .venv
. .venv/Scripts/activate

# PyTorch (CUDA 12.1 wheels)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Diffusers-Stack
pip install diffusers transformers accelerate safetensors

# Edge & Bild
pip install controlnet-aux pillow opencv-python scikit-image numpy

# Optional (wenn verfügbar): schnellere Attention
pip install xformers

# Optional: SVG-Vektorisierung
pip install vtracer
```

Hinweis: Für den `guidedFilter` wird `opencv-contrib-python` benötigt. Fehlt es,
nutzt das Tool automatisch einen `bilateralFilter` als Fallback.

## Modelle vorab herunterladen

Starte das GUI und klicke **„Modelle jetzt herunterladen“**. Folgende Repositories
werden aus dem Hugging-Face-Hub in deinen lokalen Cache geladen:

- ControlNet Lineart (SD 1.5): `lllyasviel/control_v11p_sd15_lineart`
- Stable Diffusion 1.5: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- DexiNed Annotators: `lllyasviel/Annotators` via `controlnet-aux`

Kein HF-Token nötig, da es sich um offene Modelle handelt. Der Cache liegt
standardmäßig unter `%USERPROFILE%/.cache/huggingface`.

## Nutzung

```bash
python dexi_gui.py
```

1. Eingabeordner wählen → alle Bilder darin werden verarbeitet.
2. Ausgabeordner wählen → dort landen:
   - `preprocessed/<name>_dexi.png` (DexiNed-Kanten)
   - `<name>_refined.png` (optional: SD-Glättung)
   - `<name>_refined_bw.png` (hartes B/W für Vektor)
   - `<name>_refined_bw.svg` (optional: Vektor)
3. Optionen setzen oder Preset wählen (siehe unten).
4. **Start** drücken. Fortschritt und Laufzeiten stehen im Log.

## Presets

### Technische Strichzeichnung

- Steps: 36
- Guidance: 6.5
- Ctrl-Scale: 1.10
- Strength: 0.65
- SVG: an
- Max-Kante: 896 px

### Natürliche Lineart

- Steps: 32
- Guidance: 5.8
- Ctrl-Scale: 0.95
- Strength: 0.70
- SVG: an
- Max-Kante: 896 px

Beide Presets halten sich an VRAM-sichere Defaults. Erhöhe „Max-Kante“ nur
vorsichtig, wenn mehr VRAM verfügbar ist.

## Interner Ablauf

DexiNed (multi-scale): Vorab leichte Glättung (guided/bilateral) → DexiNed auf
0.75×, 1.0×, 1.25× → robuste Max-Fusion → Kontrastspreizung & leichter Blur →
dynamische Schwelle (Perzentile) → Closing → Remove-Small-Objects → Skeletonize
→ Hole-Remove → feine 1-Pixel-Konturen.

SD 1.5 + ControlNet(Lineart) (optional): Auf Vielfache von 8 skalieren (Default
896 px) → CPU-Offload, Attention-Slicing, VAE-Slicing/Tiling aktiv (Diffusers) →
Prompt erzwingt schwarz-weiße Lineart ohne Schattierung; Hintergrund bleibt.

B/W-Clipping & SVG: Hartes Schwellen auf Schwarz/Weiß → VTracer erzeugt SVG.

## Feintuning-Tipps

- Noch feinere Details: interne Dexi-Schwelle (`thr`) senken (z. B. 45.–48.
  Perzentil).
- Lücken schließen: `binary_closing(..., square(3))` +
  `remove_small_objects(min_size=8–12)` erhöhen.
- Linien zu dünn? Einmaliges Dilatieren (1‑px-Kernel) nach `skeletonize`.
- Zu sterile SD-Linien: `strength` auf 0.60–0.65, `ctrl-scale` auf 0.9–0.95.
- Nur deterministisch (ohne SD): SD-Refinement in der GUI deaktivieren; SVG
  direkt aus `*_dexi.png`.

## Quellen

- ControlNet Lineart (SD 1.5) – Modellkarte
- Stable Diffusion 1.5 – Modellkarte / Hub-Mirror
- ControlNet (allg.) – Hub-Ressourcen
- DexiNed – Paper & Repos
- Diffusers Memory-Optimierung – Offload/Slicing/Tiling
- VTracer – Raster→Vektor
