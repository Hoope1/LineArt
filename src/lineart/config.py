"""Configuration objects for the Dexi LineArt pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .constants import (
    DEFAULT_CTRL_SCALE,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_LONG,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_STRENGTH,
)


@dataclass(slots=True)
class PipelineConfig:
    """Runtime configuration for batch processing."""

    use_sd: bool = True
    save_svg: bool = True
    steps: int = DEFAULT_STEPS
    guidance: float = DEFAULT_GUIDANCE
    ctrl: float = DEFAULT_CTRL_SCALE
    strength: float = DEFAULT_STRENGTH
    seed: int = DEFAULT_SEED
    max_long: int = DEFAULT_MAX_LONG
    batch_size: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PipelineConfig:
        """Create a configuration instance from a generic mapping."""
        return cls(
            use_sd=bool(data.get("use_sd", True)),
            save_svg=bool(data.get("save_svg", True)),
            steps=int(data.get("steps", DEFAULT_STEPS)),
            guidance=float(data.get("guidance", DEFAULT_GUIDANCE)),
            ctrl=float(data.get("ctrl", DEFAULT_CTRL_SCALE)),
            strength=float(data.get("strength", DEFAULT_STRENGTH)),
            seed=int(data.get("seed", DEFAULT_SEED)),
            max_long=int(data.get("max_long", DEFAULT_MAX_LONG)),
            batch_size=max(1, int(data.get("batch_size", 1))),
        )
