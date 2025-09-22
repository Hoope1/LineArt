"""SVG export helpers."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def save_svg_vtracer(png_path: Path, svg_path: Path) -> bool:
    """Convert *png_path* to SVG via ``vtracer`` and save to *svg_path*."""
    if shutil.which("vtracer") is None:
        logger.error("vtracer CLI not found in PATH")
        return False
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
    except FileNotFoundError as exc:
        logger.error("Unable to launch vtracer CLI: %s", exc)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external tool
        err = exc.stderr.decode().strip() if exc.stderr else exc
        logger.error("vtracer command failed: %s", err)
    except Exception as exc:  # pragma: no cover - unexpected
        logger.error("Unexpected SVG export error: %s", exc)
    return False
