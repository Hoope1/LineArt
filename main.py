#!/usr/bin/env python3
"""DexiNed → SD1.5 + ControlNet(Lineart) Pipeline GUI."""

from __future__ import annotations

import logging
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.pipeline import (
    DEFAULT_CTRL_SCALE,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_LONG,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_STRENGTH,
    detect_device,
    prefetch_models,
    process_folder,
)

# GUI constants
WINDOW_TITLE = "Dexi LineArt Max (SD1.5 + ControlNet) – Batch GUI"
WINDOW_GEOMETRY = "820x720"
LOG_INTERVAL_MS = 100
PAD = {"padx": 8, "pady": 6}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class App(tk.Tk):
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
        self.title(WINDOW_TITLE)
        self.geometry(WINDOW_GEOMETRY)

        self.inp_var = tk.StringVar()
        self.out_var = tk.StringVar()

        self.use_sd = tk.BooleanVar(value=True)
        self.save_svg = tk.BooleanVar(value=True)

        self.steps = tk.IntVar(value=DEFAULT_STEPS)
        self.guidance = tk.DoubleVar(value=DEFAULT_GUIDANCE)
        self.ctrl = tk.DoubleVar(value=DEFAULT_CTRL_SCALE)
        self.strength = tk.DoubleVar(value=DEFAULT_STRENGTH)
        self.seed = tk.IntVar(value=DEFAULT_SEED)
        self.max_long = tk.IntVar(value=DEFAULT_MAX_LONG)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.after(LOG_INTERVAL_MS, self.process_log_queue)
        self.stop_event = threading.Event()
        self.progress_total = 0

        self._build()

        self.running = False
        self.worker: threading.Thread | None = None

    def _build(self) -> None:
        """Create and lay out all widgets.

        Returns:
            None

        Raises:
            None

        """
        pad = PAD

        frm_paths = ttk.LabelFrame(self, text="Ordner")
        frm_paths.pack(fill="x", **pad)

        ttk.Label(frm_paths, text="Eingabe:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm_paths, textvariable=self.inp_var, width=70).grid(
            row=0, column=1, sticky="we"
        )
        ttk.Button(frm_paths, text="…", command=self.pick_inp).grid(row=0, column=2)

        ttk.Label(frm_paths, text="Ausgabe:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm_paths, textvariable=self.out_var, width=70).grid(
            row=1, column=1, sticky="we"
        )
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
        ttk.Entry(frm_opts, textvariable=self.steps, width=6).grid(
            row=1, column=1, sticky="w"
        )
        ttk.Label(frm_opts, text="Guidance").grid(row=1, column=2, sticky="e")
        ttk.Entry(frm_opts, textvariable=self.guidance, width=6).grid(
            row=1, column=3, sticky="w"
        )
        ttk.Label(frm_opts, text="Ctrl-Scale").grid(row=1, column=4, sticky="e")
        ttk.Entry(frm_opts, textvariable=self.ctrl, width=6).grid(
            row=1, column=5, sticky="w"
        )

        ttk.Label(frm_opts, text="Strength (img2img)").grid(row=2, column=0, sticky="e")
        ttk.Entry(frm_opts, textvariable=self.strength, width=6).grid(
            row=2, column=1, sticky="w"
        )
        ttk.Label(frm_opts, text="Seed").grid(row=2, column=2, sticky="e")
        ttk.Entry(frm_opts, textvariable=self.seed, width=8).grid(
            row=2, column=3, sticky="w"
        )
        ttk.Label(frm_opts, text="Max lange Kante (px)").grid(
            row=2, column=4, sticky="e"
        )
        ttk.Entry(frm_opts, textvariable=self.max_long, width=6).grid(
            row=2, column=5, sticky="w"
        )

        frm_presets = ttk.LabelFrame(self, text="Presets")
        frm_presets.pack(fill="x", **pad)

        ttk.Button(
            frm_presets,
            text="Technische Strichzeichnung",
            command=self.preset_technical,
        ).grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Button(
            frm_presets, text="Natürliche Lineart", command=self.preset_natural
        ).grid(row=0, column=1, padx=4, pady=4, sticky="w")

        frm_actions = ttk.Frame(self)
        frm_actions.pack(fill="x", **pad)

        self.btn_prefetch = ttk.Button(
            frm_actions,
            text="Modelle jetzt herunterladen",
            command=self.prefetch,
        )
        self.btn_prefetch.pack(side="left")
        self.btn_start = ttk.Button(
            frm_actions,
            text="Start",
            command=self.start,
        )
        self.btn_start.pack(side="left", padx=10)
        self.btn_stop = ttk.Button(
            frm_actions,
            text="Stopp",
            command=self.stop,
        )
        self.btn_stop.pack(side="left")

        self.pbar = ttk.Progressbar(frm_actions, mode="determinate")
        self.pbar.pack(fill="x", expand=True, padx=10)

        frm_log = ttk.LabelFrame(self, text="Log")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt: tk.Text = tk.Text(frm_log, height=20)
        self.txt.pack(fill="both", expand=True)

    # --- Preset-Setter ---
    def preset_technical(self) -> None:
        """Set parameters for technical line art.

        Returns:
            None

        Raises:
            None

        """
        self.use_sd.set(True)
        self.save_svg.set(True)
        self.steps.set(36)
        self.guidance.set(6.5)
        self.ctrl.set(1.10)
        self.strength.set(0.65)
        self.seed.set(42)
        self.max_long.set(896)
        self.log("Preset geladen: Technische Strichzeichnung")

    def preset_natural(self) -> None:
        """Set parameters for natural line art.

        Returns:
            None

        Raises:
            None

        """
        self.use_sd.set(True)
        self.save_svg.set(True)
        self.steps.set(32)
        self.guidance.set(5.8)
        self.ctrl.set(0.95)
        self.strength.set(0.70)
        self.seed.set(42)
        self.max_long.set(896)
        self.log("Preset geladen: Natürliche Lineart")

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

    def progress(self, cur: int, total: int, _path: Path) -> None:
        """Enqueue progress update.

        Args:
            cur: Current index.
            total: Total number of items.
            _path: Current image path (unused).

        Returns:
            None

        Raises:
            None

        """
        self.log_queue.put(("progress", cur, total))

    def process_log_queue(self) -> None:
        """Handle queued log and progress events.

        Returns:
            None

        Raises:
            None

        """
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            if msg[0] == "log":
                self.txt.insert("end", msg[1] + "\n")
                self.txt.see("end")
            elif msg[0] == "progress":
                _, cur, total = msg
                self.pbar.config(maximum=total, value=cur)
        self.after(100, self.process_log_queue)

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
        inp, out = Path(self.inp_var.get()), Path(self.out_var.get())
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
        self.stop_event.clear()
        self.pbar.config(value=0)
        self.btn_start.config(state="disabled")
        self.btn_prefetch.config(state="disabled")
        self.btn_stop.config(state="normal")
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
