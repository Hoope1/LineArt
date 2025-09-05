# Dexi LineArt Max (SD1.5 + ControlNet)

**Ziel**  
Aus bestehenden Bildern hochqualitative **Schwarz-Weiß-Lineart** erzeugen – inkl. Hintergrund, wahlweise sehr „technisch“ oder „natürlich“. Ergebnis als **PNG** und optional **SVG** (Vektor) für perfekte Skalierung/Druck.

**Hardware-Profil**  
Entwickelt und getestet für **NVIDIA Quadro T1000 (4 GB VRAM)** + **16 GB RAM**. Die Pipeline ist so gewählt, dass sie **stabil** auf 4 GB läuft (Zeit ist zweitrangig).

---

## Warum genau dieser Stack?

1. **DexiNed** (Edge-Netz) liefert **feinste und scharfe** Kanten. Wir verwenden den **Annotator** aus `controlnet-aux` und fahren **multi-scale** mit Nachbearbeitung (Hysterese-ähnlich, Closing, Skeletonize). DexiNed ist für sehr feine, zusammenhängende Konturen bekannt. 0  
2. **Stable Diffusion 1.5 + ControlNet(Lineart)** glättet/normalisiert die aus DexiNed kommenden Linien **ohne** Schattierung oder Farbe – das **Lineart-ControlNet** ist für SD 1.5 trainiert und in der Community Standard. 1  
3. **SD 1.5** ist im Gegensatz zu **SDXL** auf **4 GB VRAM** verlässlich betreibbar (mit **CPU-Offload**, **Attention/ VAE-Slicing/-Tiling** in Diffusers). SDXL ist auf 4 GB realitätsfern langsam/instabil; daher setzen wir auf SD 1.5. 2  
4. **VTracer** erzeugt aus der binären Lineart **SVG-Vektoren**, ideal für Unterrichtsmaterial und Druck. 3

---

## Installation (Windows 11, Python ≥ 3.10)

```bash
python -m venv .venv
. .venv/Scripts/activate

# PyTorch (CUDA 12.1 Wheels)
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

## Nutzung

Starte die GUI nach Installation der Abhängigkeiten:

```bash
python dexi_gui.py
```

Ein Fenster ermöglicht das Auswählen von Eingabe- und Ausgabeordnern,
das Setzen der Optionen sowie den Start der Batch-Verarbeitung.
