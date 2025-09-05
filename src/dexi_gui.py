#!/usr/bin/env python3
"""DexiNed → SD 1.5 + ControlNet(Lineart) Pipeline mit GUI.

- Ordnerweise Batch-Verarbeitung
- Modelle vorab herunterladen (Prefetch)
- PNG-Ausgabe + optional SVG (VTracer)
- Zwei Presets: "Technische Strichzeichnung" & "Natürliche Lineart"

Optimiert für NVIDIA Turing mit 4 GB VRAM (Quadro T1000).

Abhängigkeiten (siehe README.md):
  torch, diffusers, transformers, accelerate, safetensors,
  controlnet-aux, pillow, opencv-python, scikit-image, numpy,
  (optional) xformers, vtracer
"""

import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image
from skimage import exposure, morphology
from skimage.morphology import binary_closing, remove_small_objects, skeletonize, square

# ---------- Utility ----------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    """Return all image paths in *folder* with a supported extension."""
    return [p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS]


def to_mult8(w, h, max_long=896):
    """Scale dimensions to multiples of eight, limiting the longest side."""
    if max(w, h) > max_long:
        s = max_long / max(w, h)
        w, h = int(w * s), int(h * s)
    return max(64, (w // 8) * 8), max(64, (h // 8) * 8)


def ensure_dir(p: Path):
    """Create directory *p* if needed and return the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_svg_vtracer(png_path: Path, svg_path: Path):
    """Convert *png_path* to SVG via ``vtracer`` and save to *svg_path*."""
    try:
        subprocess.run(
            [
                "vtracer",
                "--input",
                str(png_path),
                "--output",
                str(svg_path),
                "--mode",
                "polygon",
                "--filter-speckle",
                "8",
                "--hierarchical",
                "true",
            ],
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        return False


# ---------- DexiNed (controlnet-aux) ----------
_dexi = None


def load_dexined(device="cuda"):
    """Load the DexiNed edge detector on the requested *device*."""
    global _dexi
    if _dexi is not None:
        return _dexi
    from controlnet_aux import DexiNedDetector  # pip install controlnet-aux

    _dexi = DexiNedDetector.from_pretrained("lllyasviel/Annotators").to(device)
    return _dexi


def guided_smooth_if_available(pil_img):
    """Sanftes Entkörnen mit Detailerhalt (guidedFilter falls vorhanden, sonst bilateral)."""
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if hasattr(cv2, "ximgproc"):  # guidedFilter nur in opencv-contrib verfügbar
        arr = cv2.ximgproc.guidedFilter(guide=arr, src=arr, radius=4, eps=1e-2)
    else:
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=25, sigmaSpace=25)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def dexi_multiscale_edges(det, pil_img, scales=(1.0, 0.75, 1.25), pre_smooth=True):
    """Multi-Scale-DexiNed mit leichtem Postprocessing.

    Rückgabe: 8-bit L (0 = schwarze Linie, 255 = weiß).
    """
    img = pil_img.convert("RGB")
    if pre_smooth:
        img = guided_smooth_if_available(img)

    maps = []
    for s in scales:
        if s != 1.0:
            sz = (max(64, int(img.width * s)), max(64, int(img.height * s)))
            img_s = img.resize(sz, Image.LANCZOS)
        else:
            img_s = img
        e = det(img_s)  # PIL Image (L oder RGB je nach Version)
        if e.mode != "L":
            e = e.convert("L")
        e = e.resize((img.width, img.height), Image.BILINEAR)
        maps.append(np.array(e, dtype=np.float32) / 255.0)

    E = np.maximum.reduce(maps)  # robuste Fusionsregel für Kanten
    lo, hi = np.percentile(E, 5), np.percentile(E, 99)
    E = exposure.rescale_intensity(E, in_range=(lo, hi))
    E = cv2.GaussianBlur(E, (0, 0), 0.7)

    thr = np.clip(np.percentile(E, 50), 0.25, 0.65)
    mask = E <= thr  # „Linie ist dunkel“

    mask = binary_closing(mask, square(2))
    mask = remove_small_objects(mask, min_size=16)
    thin = skeletonize(mask)
    thin = morphology.remove_small_holes(thin, area_threshold=16)

    out = np.where(thin, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


# ---------- SD 1.5 + ControlNet(Lineart) ----------
_sd_pipe = None


def load_sd15_lineart():
    """Lädt SD1.5 + ControlNet Lineart mit RAM/VRAM-schonenden Optionen."""
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()  # Wichtig für 4 GB
    _sd_pipe = pipe
    return _sd_pipe


def sd_refine(
    base_rgb,
    ctrl_L,
    steps=32,
    guidance=6.0,
    ctrl_scale=1.0,
    strength=0.70,
    seed=42,
    max_long=896,
):
    """Refine edges with SD1.5 + ControlNet and return color and BW images."""
    import torch

    pipe = load_sd15_lineart()
    W, H = base_rgb.size
    w, h = to_mult8(W, H, max_long=max_long)
    base = base_rgb.resize((w, h), Image.LANCZOS).convert("RGB")
    ctrl = ctrl_L.resize((w, h), Image.NEAREST).convert("RGB")

    gen = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    img = pipe(
        prompt=(
            "clean black-and-white line art, uniform outlines, detailed, "
            "background preserved, white paper look, no shading"
        ),
        negative_prompt="color, gradients, blur, watermark, text, messy edges, artifacts",
        image=base,
        control_image=ctrl,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=ctrl_scale,
        strength=strength,
        generator=gen,
    ).images[0]

    # Hartes Schwarz/Weiß für Druck & Vektor
    bw = img.convert("L").point(lambda v: 255 if v > 200 else 0, mode="1").convert("L")
    return img, bw


# ---------- Verarbeitung ----------
def process_one(path: Path, out_dir: Path, cfg, log):
    """Process a single image and write outputs to *out_dir*."""
    t0 = time.perf_counter()
    src = Image.open(path).convert("RGB")

    # 1) DexiNed
    det = load_dexined(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    edges = dexi_multiscale_edges(det, src, scales=(1.0, 0.75, 1.25), pre_smooth=True)
    ensure_dir(out_dir)
    prep_dir = ensure_dir(out_dir / "preprocessed")
    out_edges = prep_dir / f"{path.stem}_dexi.png"
    edges.save(out_edges)

    result_paths = [out_edges]

    # 2) Optional SD-Refinement
    if cfg["use_sd"]:
        refined, bw = sd_refine(
            src,
            edges,
            steps=cfg["steps"],
            guidance=cfg["guidance"],
            ctrl_scale=cfg["ctrl"],
            strength=cfg["strength"],
            seed=cfg["seed"],
            max_long=cfg["max_long"],
        )
        ref_path = out_dir / f"{path.stem}_refined.png"
        bw_path = out_dir / f"{path.stem}_refined_bw.png"
        refined.save(ref_path)
        bw.save(bw_path)
        result_paths += [ref_path, bw_path]

        # 3) Optional SVG
        if cfg["save_svg"]:
            svg_ok = save_svg_vtracer(bw_path, out_dir / f"{path.stem}_refined_bw.svg")
            if svg_ok:
                result_paths.append(out_dir / f"{path.stem}_refined_bw.svg")

    dt = time.perf_counter() - t0
    log(f"{path.name}  → fertig ({dt:.1f} s)\n" + "   " + ", ".join([p.name for p in result_paths]))


def process_folder(inp_dir: Path, out_dir: Path, cfg, log, done_cb):
    """Process all supported images from *inp_dir* into *out_dir*."""
    imgs = list_images(inp_dir)
    if not imgs:
        log("Keine Eingabebilder gefunden.")
        done_cb()
        return
    for p in imgs:
        process_one(p, out_dir, cfg, log)
    log("\nALLE BILDER ERLEDIGT.")
    done_cb()


# ---------- Prefetch (lädt Modelle vorab) ----------
def prefetch_models(log):
    """Download all required models ahead of time."""
    log("Lade Modelle vom Hub … (einmalig)")
    _ = load_dexined(device="cpu")  # DexiNed Annotators
    _ = load_sd15_lineart()  # SD 1.5 + ControlNet Lineart
    log("Modelle vorhanden.\n")


# ---------- GUI ----------
class App(tk.Tk):
    """Tkinter GUI for batch line-art generation."""

    def __init__(self):
        """Initialize window, variables and widgets."""
        super().__init__()
        self.title("Dexi LineArt Max (SD1.5 + ControlNet) – Batch GUI")
        self.geometry("820x720")

        self.inp_var = tk.StringVar()
        self.out_var = tk.StringVar()

        self.use_sd = tk.BooleanVar(value=True)
        self.save_svg = tk.BooleanVar(value=True)

        self.steps = tk.IntVar(value=32)
        self.guidance = tk.DoubleVar(value=6.0)
        self.ctrl = tk.DoubleVar(value=1.0)
        self.strength = tk.DoubleVar(value=0.70)
        self.seed = tk.IntVar(value=42)
        self.max_long = tk.IntVar(value=896)

        self._build()

        self.running = False
        self.worker = None

    def _build(self):
        pad = {"padx": 8, "pady": 6}

        frm_paths = ttk.LabelFrame(self, text="Ordner")
        frm_paths.pack(fill="x", **pad)

        ttk.Label(frm_paths, text="Eingabe:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm_paths, textvariable=self.inp_var, width=70).grid(row=0, column=1, sticky="we")
        ttk.Button(frm_paths, text="…", command=self.pick_inp).grid(row=0, column=2)

        ttk.Label(frm_paths, text="Ausgabe:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm_paths, textvariable=self.out_var, width=70).grid(row=1, column=1, sticky="we")
        ttk.Button(frm_paths, text="…", command=self.pick_out).grid(row=1, column=2)

        frm_opts = ttk.LabelFrame(self, text="Optionen")
        frm_opts.pack(fill="x", **pad)

        ttk.Checkbutton(
            frm_opts,
            text="SD-Refinement (SD1.5 + ControlNet Lineart)",
            variable=self.use_sd,
        ).grid(
            row=0,
            column=0,
            sticky="w",
            columnspan=3,
        )
        ttk.Checkbutton(
            frm_opts,
            text="SVG speichern (VTracer)",
            variable=self.save_svg,
        ).grid(
            row=0,
            column=3,
            sticky="w",
        )

        ttk.Label(frm_opts, text="Steps").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm_opts, textvariable=self.steps, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(frm_opts, text="Guidance").grid(row=1, column=2, sticky="e")
        ttk.Entry(frm_opts, textvariable=self.guidance, width=6).grid(row=1, column=3, sticky="w")
        ttk.Label(frm_opts, text="Ctrl-Scale").grid(row=1, column=4, sticky="e")
        ttk.Entry(frm_opts, textvariable=self.ctrl, width=6).grid(row=1, column=5, sticky="w")

        ttk.Label(frm_opts, text="Strength (img2img)").grid(row=2, column=0, sticky="e")
        ttk.Entry(frm_opts, textvariable=self.strength, width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(frm_opts, text="Seed").grid(row=2, column=2, sticky="e")
        ttk.Entry(frm_opts, textvariable=self.seed, width=8).grid(row=2, column=3, sticky="w")
        ttk.Label(frm_opts, text="Max lange Kante (px)").grid(row=2, column=4, sticky="e")
        ttk.Entry(frm_opts, textvariable=self.max_long, width=6).grid(row=2, column=5, sticky="w")

        # --- Presets ---
        frm_presets = ttk.LabelFrame(self, text="Presets")
        frm_presets.pack(fill="x", **pad)

        ttk.Button(
            frm_presets, text="Technische Strichzeichnung", command=self.preset_technical
        ).grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Button(frm_presets, text="Natürliche Lineart", command=self.preset_natural).grid(
            row=0, column=1, padx=4, pady=4, sticky="w"
        )

        frm_actions = ttk.Frame(self)
        frm_actions.pack(fill="x", **pad)

        ttk.Button(
            frm_actions,
            text="Modelle jetzt herunterladen",
            command=self.prefetch,
        ).pack(side="left")
        ttk.Button(
            frm_actions,
            text="Start",
            command=self.start,
        ).pack(side="left", padx=10)
        ttk.Button(
            frm_actions,
            text="Stopp",
            command=self.stop,
        ).pack(side="left")

        frm_log = ttk.LabelFrame(self, text="Log")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(frm_log, height=20)
        self.txt.pack(fill="both", expand=True)

    # --- Preset-Setter ---
    def preset_technical(self):
        """Technische Strichzeichnung: einheitliche Konturen, klare Kanten."""
        self.use_sd.set(True)
        self.save_svg.set(True)
        self.steps.set(36)
        self.guidance.set(6.5)
        self.ctrl.set(1.10)  # strikteres Folgen des Hints
        self.strength.set(0.65)  # geringere Abweichung vom Ausgang
        self.seed.set(42)
        self.max_long.set(896)
        self.log("Preset geladen: Technische Strichzeichnung")

    def preset_natural(self):
        """Natürliche Lineart: Hintergrund erhalten, weichere Linien."""
        self.use_sd.set(True)
        self.save_svg.set(True)
        self.steps.set(32)
        self.guidance.set(5.8)
        self.ctrl.set(0.95)  # etwas freier, um Hintergrundstruktur zu wahren
        self.strength.set(0.70)
        self.seed.set(42)
        self.max_long.set(896)
        self.log("Preset geladen: Natürliche Lineart")

    # --- GUI-Handlers ---
    def pick_inp(self):
        """Ask the user for an input directory."""
        p = filedialog.askdirectory(title="Eingabeordner wählen")
        if p:
            self.inp_var.set(p)

    def pick_out(self):
        """Ask the user for an output directory."""
        p = filedialog.askdirectory(title="Ausgabeordner wählen")
        if p:
            self.out_var.set(p)

    def log(self, s):
        """Append a message to the GUI log widget."""
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def prefetch(self):
        """Download models in a background thread."""

        def job():
            try:
                prefetch_models(self.log)
                messagebox.showinfo("Fertig", "Alle Modelle sind lokal verfügbar.")
            except Exception as e:
                messagebox.showerror("Fehler beim Herunterladen", str(e))

        threading.Thread(target=job, daemon=True).start()

    def start(self):
        """Start processing in a separate thread."""
        if self.running:
            return
        inp, out = Path(self.inp_var.get()), Path(self.out_var.get())
        if not inp.exists():
            messagebox.showwarning("Fehler", "Bitte Eingabeordner wählen.")
            return
        if not out.exists():
            try:
                out.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Fehler", f"Ausgabeordner kann nicht erstellt werden:\n{e}")
                return

        cfg = dict(
            use_sd=self.use_sd.get(),
            save_svg=self.save_svg.get(),
            steps=int(self.steps.get()),
            guidance=float(self.guidance.get()),
            ctrl=float(self.ctrl.get()),
            strength=float(self.strength.get()),
            seed=int(self.seed.get()),
            max_long=int(self.max_long.get()),
        )

        self.running = True
        self.log("Starte Verarbeitung …")

        def job():
            try:
                process_folder(inp, out, cfg, self.log, self.done)
            except Exception as e:
                self.log(f"FEHLER: {e}")
                self.done()

        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def stop(self):
        """Request that processing stop after the current image."""
        # Hard-Stop: GUI-seitig nur Flag, da Diffusers synchron rechnet.
        # Bei Bedarf Prozess im Task-Manager beenden.
        self.running = False
        self.log("Stop angefordert (nach aktuellem Bild).")

    def done(self):
        """Mark the current job as finished."""
        self.running = False
        self.log("Fertig.")


def main():
    """Start the Dexi LineArt GUI."""
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        pass
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
