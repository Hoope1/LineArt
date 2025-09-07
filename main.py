#!/usr/bin/env python3
"""DexiNed → SD1.5 + ControlNet(Lineart) Pipeline GUI."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, cast

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    TKDND_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    DND_FILES = None  # type: ignore[assignment]
    TKDND_AVAILABLE = False
    TkinterDnD = tk  # type: ignore[assignment]

from src.pipeline import (
    DEFAULT_CTRL_SCALE,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_LONG,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_STRENGTH,
    Config,
    detect_device,
    prefetch_models,
    process_folder,
)

# GUI constants
WINDOW_TITLE = "Dexi LineArt Max (SD1.5 + ControlNet) – Batch GUI"
WINDOW_GEOMETRY = "820x720"
LOG_INTERVAL_MS = 100
PAD = {"padx": 8, "pady": 6}

SETTINGS_FILE = Path.home() / ".dexined_pipeline" / "settings.json"

ICON_DONE = "\u2713"  # ✓
ICON_WORK = "\u26a1"  # ⚡
ICON_ERROR = "\u274c"  # ❌
ICON_PAUSE = "\u23f8"  # ⏸

if TKDND_AVAILABLE:
    BaseTk = TkinterDnD.Tk  # type: ignore[attr-defined]
else:
    BaseTk = tk.Tk


class CreateToolTip:
    """Simple tooltip implementation for Tk widgets."""

    widget: tk.Widget
    text: str
    tipwindow: tk.Toplevel | None

    def __init__(self, widget: tk.Widget, text: str) -> None:
        """Store widget reference and register events."""
        self.widget = widget
        self.text = text
        self.tipwindow = None
        _ = widget.bind("<Enter>", self.show_tip)
        _ = widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, _event: tk.Event) -> None:  # type: ignore[override]
        """Display the tooltip window."""
        if self.tipwindow:
            return
        bbox = cast(Any, self.widget).bbox("insert") or (0, 0, 0, 0)
        x, y = bbox[0], bbox[1]
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("tahoma", 8, "normal"),  # type: ignore[arg-type]
        )
        label.pack(ipadx=1)

    def hide_tip(self, _event: tk.Event) -> None:  # type: ignore[override]
        """Hide the tooltip window."""
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


LogMessage = tuple[str, str]
ProgressMessage = tuple[str, int, int, str]
QueueItem = LogMessage | ProgressMessage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class App(BaseTk):
    """Tkinter GUI for batch line-art generation.

    Attributes:
        inp_var: Selected input directory.
        out_var: Selected output directory.
        use_sd: Whether to run SD refinement.
        save_svg: Whether to save SVG outputs.
        steps: Diffusion steps.
        guidance: Guidance scale.
        ctrl: ControlNet conditioning scale.
        strength: Img2Img strength.
        seed: Random seed for reproducibility.
        max_long: Maximum long edge in pixels.
        log_queue: Queue for thread-safe logging.
        stop_event: Event signalling a stop request.
        progress_total: Total number of images processed.
        running: Flag indicating active processing.
        worker: Background worker thread.

    """

    def __init__(self) -> None:
        """Initialize window, variables and widgets.

        Returns:
            None

        Raises:
            None

        """
        super().__init__()
        _ = self.title(WINDOW_TITLE)
        _ = self.geometry(WINDOW_GEOMETRY)

        self.inp_var: tk.StringVar = tk.StringVar()
        self.out_var: tk.StringVar = tk.StringVar()

        self.use_sd: tk.BooleanVar = tk.BooleanVar(value=True)
        self.save_svg: tk.BooleanVar = tk.BooleanVar(value=True)

        self.steps: tk.IntVar = tk.IntVar(value=DEFAULT_STEPS)
        self.guidance: tk.DoubleVar = tk.DoubleVar(value=DEFAULT_GUIDANCE)
        self.ctrl: tk.DoubleVar = tk.DoubleVar(value=DEFAULT_CTRL_SCALE)
        self.strength: tk.DoubleVar = tk.DoubleVar(value=DEFAULT_STRENGTH)
        self.seed: tk.IntVar = tk.IntVar(value=DEFAULT_SEED)
        self.max_long: tk.IntVar = tk.IntVar(value=DEFAULT_MAX_LONG)

        self.log_queue: queue.Queue[QueueItem] = queue.Queue()
        _ = self.after(LOG_INTERVAL_MS, self.process_log_queue)
        self.stop_event: threading.Event = threading.Event()
        self.progress_total: int = 0
        self.start_time: float = 0.0

        self.status_var: tk.StringVar = tk.StringVar(value="Bereit")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._load_settings()
        self._build()

        self.running: bool = False
        self.worker: threading.Thread | None = None

    def _build(self) -> None:
        """Create and lay out all widgets.

        Returns:
            None

        Raises:
            None

        """
        pad_x, pad_y = PAD["padx"], PAD["pady"]

        frm_io = ttk.LabelFrame(self, text="Eingabe/Ausgabe")
        frm_io.pack(fill="x", padx=pad_x, pady=pad_y)

        ttk.Label(frm_io, text="Eingabe:").grid(row=0, column=0, sticky="e")
        inp_entry = ttk.Entry(frm_io, textvariable=self.inp_var, width=70)
        inp_entry.grid(row=0, column=1, sticky="we")
        ttk.Button(frm_io, text="…", command=self.pick_inp).grid(row=0, column=2)
        _ = CreateToolTip(inp_entry, "Ordner mit Eingabebildern")
        if TKDND_AVAILABLE:
            inp_entry.drop_target_register(DND_FILES)
            inp_entry.dnd_bind("<<Drop>>", self._on_drop_inp)

        ttk.Label(frm_io, text="Ausgabe:").grid(row=1, column=0, sticky="e")
        out_entry = ttk.Entry(frm_io, textvariable=self.out_var, width=70)
        out_entry.grid(row=1, column=1, sticky="we")
        ttk.Button(frm_io, text="…", command=self.pick_out).grid(row=1, column=2)
        _ = CreateToolTip(out_entry, "Ordner für Ausgabebilder")
        if TKDND_AVAILABLE:
            out_entry.drop_target_register(DND_FILES)
            out_entry.dnd_bind("<<Drop>>", self._on_drop_out)

        frm_quality = ttk.LabelFrame(self, text="Qualität")
        frm_quality.pack(fill="x", padx=pad_x, pady=pad_y)

        ttk.Label(frm_quality, text="Steps").grid(row=0, column=0, sticky="e")
        steps_entry = ttk.Entry(frm_quality, textvariable=self.steps, width=6)
        steps_entry.grid(row=0, column=1, sticky="w")
        _ = CreateToolTip(steps_entry, "Diffusionsschritte (1-100, Standard 32)")
        ttk.Label(frm_quality, text="Guidance").grid(row=0, column=2, sticky="e")
        guidance_entry = ttk.Entry(frm_quality, textvariable=self.guidance, width=6)
        guidance_entry.grid(row=0, column=3, sticky="w")
        _ = CreateToolTip(guidance_entry, "Guidance Scale (0-20, Standard 6.0)")
        ttk.Label(frm_quality, text="Ctrl-Scale").grid(row=0, column=4, sticky="e")
        ctrl_entry = ttk.Entry(frm_quality, textvariable=self.ctrl, width=6)
        ctrl_entry.grid(row=0, column=5, sticky="w")
        _ = CreateToolTip(ctrl_entry, "ControlNet Einfluss (0-2, Standard 1.0)")
        ttk.Label(frm_quality, text="Strength (img2img)").grid(
            row=1, column=0, sticky="e"
        )
        strength_entry = ttk.Entry(frm_quality, textvariable=self.strength, width=6)
        strength_entry.grid(row=1, column=1, sticky="w")
        _ = CreateToolTip(strength_entry, "Img2Img Stärke (0-1, Standard 0.7)")

        frm_performance = ttk.LabelFrame(self, text="Performance")
        frm_performance.pack(fill="x", padx=pad_x, pady=pad_y)
        ttk.Label(frm_performance, text="Max lange Kante (px)").grid(
            row=0, column=0, sticky="e"
        )
        max_entry = ttk.Entry(frm_performance, textvariable=self.max_long, width=6)
        max_entry.grid(row=0, column=1, sticky="w")
        _ = CreateToolTip(max_entry, "Maximale Bildkante (64-4096, Standard 896)")

        frm_adv = ttk.LabelFrame(self, text="Erweitert")
        frm_adv.pack(fill="x", padx=pad_x, pady=pad_y)
        sd_chk = ttk.Checkbutton(
            frm_adv,
            text="SD-Refinement (SD1.5 + ControlNet Lineart)",
            variable=self.use_sd,
        )
        sd_chk.grid(row=0, column=0, sticky="w", columnspan=2)
        _ = CreateToolTip(sd_chk, "Stable Diffusion Verfeinerung aktivieren")
        svg_chk = ttk.Checkbutton(
            frm_adv,
            text="SVG speichern (VTracer)",
            variable=self.save_svg,
        )
        svg_chk.grid(row=0, column=2, sticky="w")
        _ = CreateToolTip(svg_chk, "SVG-Ausgabe erzeugen")
        ttk.Label(frm_adv, text="Seed").grid(row=1, column=0, sticky="e")
        seed_entry = ttk.Entry(frm_adv, textvariable=self.seed, width=8)
        seed_entry.grid(row=1, column=1, sticky="w")
        _ = CreateToolTip(seed_entry, "Zufalls-Seed (Standard 42)")

        frm_presets = ttk.LabelFrame(self, text="Presets")
        frm_presets.pack(fill="x", padx=pad_x, pady=pad_y)

        ttk.Button(
            frm_presets,
            text="Quick",
            command=lambda: self._apply_preset(16, "Quick"),
        ).grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Button(
            frm_presets,
            text="Standard",
            command=lambda: self._apply_preset(32, "Standard"),
        ).grid(row=0, column=1, padx=4, pady=4, sticky="w")
        ttk.Button(
            frm_presets,
            text="Quality",
            command=lambda: self._apply_preset(50, "Quality"),
        ).grid(row=0, column=2, padx=4, pady=4, sticky="w")
        ttk.Button(
            frm_presets,
            text="Technical",
            command=lambda: self._apply_preset(40, "Technical"),
        ).grid(row=0, column=3, padx=4, pady=4, sticky="w")

        frm_actions = ttk.Frame(self)
        frm_actions.pack(fill="x", padx=pad_x, pady=pad_y)

        self.btn_prefetch: ttk.Button = ttk.Button(
            frm_actions,
            text="Modelle jetzt herunterladen",
            command=self.prefetch,
        )
        self.btn_prefetch.pack(side="left")
        self.btn_start: ttk.Button = ttk.Button(
            frm_actions,
            text="Start",
            command=self.start,
        )
        self.btn_start.pack(side="left", padx=10)
        self.btn_stop: ttk.Button = ttk.Button(
            frm_actions,
            text="Stopp",
            command=self.stop,
        )
        self.btn_stop.pack(side="left")

        self.pbar: ttk.Progressbar = ttk.Progressbar(frm_actions, mode="determinate")
        self.pbar.pack(fill="x", expand=True, padx=10)

        frm_log = ttk.LabelFrame(self, text="Log")
        frm_log.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)
        self.txt: tk.Text = tk.Text(frm_log, height=20)
        self.txt.pack(fill="both", expand=True)

        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(
            fill="x", side="bottom"
        )

    def _load_settings(self) -> None:
        """Load persisted GUI settings from disk."""
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except json.JSONDecodeError:
            return
        self.inp_var.set(data.get("last_input", ""))
        self.out_var.set(data.get("last_output", ""))
        params = data.get("params", {})
        self.use_sd.set(params.get("use_sd", True))
        self.save_svg.set(params.get("save_svg", True))
        self.steps.set(params.get("steps", DEFAULT_STEPS))
        self.guidance.set(params.get("guidance", DEFAULT_GUIDANCE))
        self.ctrl.set(params.get("ctrl", DEFAULT_CTRL_SCALE))
        self.strength.set(params.get("strength", DEFAULT_STRENGTH))
        self.seed.set(params.get("seed", DEFAULT_SEED))
        self.max_long.set(params.get("max_long", DEFAULT_MAX_LONG))

    def _save_settings(self) -> None:
        """Persist current GUI settings to disk."""
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_input": self.inp_var.get(),
            "last_output": self.out_var.get(),
            "params": {
                "use_sd": bool(self.use_sd.get()),
                "save_svg": bool(self.save_svg.get()),
                "steps": int(self.steps.get()),
                "guidance": float(self.guidance.get()),
                "ctrl": float(self.ctrl.get()),
                "strength": float(self.strength.get()),
                "seed": int(self.seed.get()),
                "max_long": int(self.max_long.get()),
            },
        }
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def on_close(self) -> None:
        """Save settings and close the application."""
        self._save_settings()
        self.destroy()

    # --- Preset-Setter ---
    def _apply_preset(self, steps: int, name: str) -> None:
        """Update step count and log preset selection."""
        self.steps.set(steps)
        self.log(f"Preset geladen: {name}")

    def pick_inp(self) -> None:
        """Ask the user for an input directory.

        Returns:
            None

        Raises:
            None

        """
        p = filedialog.askdirectory(title="Eingabeordner wählen")
        if p:
            self.inp_var.set(p)

    def pick_out(self) -> None:
        """Ask the user for an output directory.

        Returns:
            None

        Raises:
            None

        """
        p = filedialog.askdirectory(title="Ausgabeordner wählen")
        if p:
            self.out_var.set(p)

    def _clean_drop_path(self, data: str) -> str:
        """Normalize drag-and-drop paths from TkDND.

        Args:
            data: Raw event string.

        Returns:
            str: Cleaned path without surrounding braces.

        Raises:
            None

        """
        return data.strip("{}")

    def _on_drop_inp(self, event: Any) -> None:  # type: ignore[override]
        """Handle files dropped onto the input entry.

        Args:
            event: Tkinter event carrying drop data.

        Returns:
            None

        Raises:
            None

        """
        if event.data:
            path = Path(self._clean_drop_path(event.data))
            if path.is_dir():
                self.inp_var.set(str(path))

    def _on_drop_out(self, event: Any) -> None:  # type: ignore[override]
        """Handle files dropped onto the output entry.

        Args:
            event: Tkinter event carrying drop data.

        Returns:
            None

        Raises:
            None

        """
        if event.data:
            path = Path(self._clean_drop_path(event.data))
            if path.is_dir():
                self.out_var.set(str(path))

    def log(self, s: str) -> None:
        """Enqueue a log message from any thread.

        Args:
            s: Message to append.

        Returns:
            None

        Raises:
            None

        """
        self.log_queue.put(("log", s))

    def progress(self, cur: int, total: int, path: Path) -> None:
        """Enqueue progress update.

        Args:
            cur: Current index.
            total: Total number of items.
            path: Current image path.

        Returns:
            None

        Raises:
            None

        """
        self.log_queue.put(("progress", cur, total, path.name))

    def process_log_queue(self) -> None:
        """Handle queued log and progress events.

        Returns:
            None

        Raises:
            None

        """
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            if len(msg) == 2:
                _, text = msg
                self.txt.insert("end", f"{text}\n")
                self.txt.see("end")
                if text.startswith("FEHLER"):
                    self.status_var.set(f"{ICON_ERROR} {text}")
            else:
                _, cur, total, name = msg
                self.pbar["maximum"] = total
                self.pbar["value"] = cur
                elapsed = time.time() - self.start_time
                spm = cur / elapsed * 60 if elapsed > 0 else 0
                eta = (total - cur) / (spm / 60) if spm > 0 else 0
                self.status_var.set(
                    f"{ICON_WORK} {name} – {cur}/{total} | "
                    f"{spm:.1f} img/min | ETA {eta/60:.1f}m"
                )
        _ = self.after(100, self.process_log_queue)

    def prefetch(self) -> None:
        """Download models in a background thread.

        Returns:
            None

        Raises:
            None

        """

        def job() -> None:
            try:
                prefetch_models(self.log)
                messagebox.showinfo("Fertig", "Alle Modelle sind lokal verfügbar.")
            except Exception as e:  # pylint: disable=broad-except
                messagebox.showerror("Fehler beim Herunterladen", str(e))

        threading.Thread(target=job, daemon=True).start()

    def start(self) -> None:
        """Start processing in a background thread.

        Returns:
            None

        Raises:
            None

        """
        if self.running:
            return
        inp = Path(self.inp_var.get())
        out_str = self.out_var.get().strip()
        if not out_str:
            out_str = time.strftime("output_%Y-%m-%d_%H-%M-%S")
            self.out_var.set(out_str)
        out = Path(out_str)
        if not inp.exists():
            messagebox.showwarning("Fehler", "Bitte Eingabeordner wählen.")
            return
        if not out.exists():
            try:
                out.mkdir(parents=True, exist_ok=True)
            except Exception as e:  # pylint: disable=broad-except
                messagebox.showerror(
                    "Fehler", f"Ausgabeordner kann nicht erstellt werden:\n{e}"
                )
                return
        if not os.access(out, os.W_OK):
            messagebox.showerror("Fehler", "Keine Schreibrechte im Ausgabeordner")
            return

        if detect_device() != "cuda":
            messagebox.showwarning(
                "Warnung", "CUDA nicht verfügbar – CPU wird langsam sein."
            )

        cfg: Config = {
            "use_sd": self.use_sd.get(),
            "save_svg": self.save_svg.get(),
            "steps": int(self.steps.get()),
            "guidance": float(self.guidance.get()),
            "ctrl": float(self.ctrl.get()),
            "strength": float(self.strength.get()),
            "seed": int(self.seed.get()),
            "max_long": int(self.max_long.get()),
            "batch_size": 1,
        }

        self._save_settings()
        self.start_time = time.time()
        self.status_var.set(f"{ICON_WORK} Verarbeitung gestartet")

        self.running = True
        self.stop_event.clear()
        self.pbar["value"] = 0
        self.btn_start["state"] = "disabled"
        self.btn_prefetch["state"] = "disabled"
        self.btn_stop["state"] = "normal"
        self.log("Starte Verarbeitung …")

        def job() -> None:
            try:
                process_folder(
                    inp, out, cfg, self.log, self.done, self.stop_event, self.progress
                )
            except Exception as e:  # pylint: disable=broad-except
                self.log(f"FEHLER: {e}")
                self.done()

        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        """Request that processing stop after the current image.

        Returns:
            None

        Raises:
            None

        """
        self.running = False
        self.stop_event.set()
        self.log("Stop angefordert (nach aktuellem Bild).")
        self.status_var.set(f"{ICON_PAUSE} Stop angefordert")

    def done(self) -> None:
        """Mark the current job as finished.

        Returns:
            None

        Raises:
            None

        """
        self.running = False
        self.btn_start.config(state="normal")
        self.btn_prefetch.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.log("Fertig.")
        self.status_var.set(f"{ICON_DONE} Fertig")


def main() -> None:
    """Start the Dexi LineArt GUI.

    Returns:
        None

    Raises:
        None

    """
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
