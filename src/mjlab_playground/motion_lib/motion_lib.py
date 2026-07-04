from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .astro_enrichment_adapter import AstroSimulatorEnrichmentAdapter
from .motion_loader import (
    MotionFormat,
    MotionLoader,
    ReferenceMotionState,
    _validate_integer_fps,
)
from .motion_resampler import MotionResamplingCfg, ReferenceMotionResampler

MotionLibRobot = Literal["astro"]


@dataclass(frozen=True, kw_only=True)
class MotionLibCfg:
    """Stable public configuration for the MotionLib stage pipeline."""

    source_format: MotionFormat = "pyroki"
    source_fps: float = 30.0
    output_fps: float = 30.0
    robot: MotionLibRobot = "astro"
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if self.robot != "astro":
            raise ValueError("Only robot='astro' is supported by MotionLib v1")
        if self.source_format == "proto":
            raise NotImplementedError("proto source format is reserved but not implemented")
        if self.source_format != "pyroki":
            raise ValueError("Only source_format='pyroki' is supported by MotionLib v1")
        object.__setattr__(
            self,
            "source_fps",
            _validate_integer_fps(self.source_fps, field_name="source_fps"),
        )
        object.__setattr__(
            self,
            "output_fps",
            _validate_integer_fps(self.output_fps, field_name="output_fps"),
        )
        object.__setattr__(self, "device", torch.device(self.device))


class MotionLib:
    """Public source-to-rich reference motion stage interface."""

    _ENRICH_INPUT_RICH_FIELDS = (
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
        "body_contacts",
    )
    _ENRICH_OUTPUT_BODY_FIELDS = (
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
    )

    def __init__(self, cfg: MotionLibCfg) -> None:
        self.cfg = cfg
        self._resampler = ReferenceMotionResampler(
            MotionResamplingCfg(output_fps=cfg.output_fps)
        )
        self._enrichment_adapter: Any | None = None

    def load(self, motion_files: str | Path) -> list[ReferenceMotionState]:
        """Load source generalized-coordinate clips with the configured source format."""
        return MotionLoader.load(
            motion_files,
            motion_format=self.cfg.source_format,
            fps=self.cfg.source_fps,
            device=self.cfg.device,
        )

    def resample(
        self,
        motions: list[ReferenceMotionState],
    ) -> list[ReferenceMotionState]:
        """Resample source clips to ``cfg.output_fps`` without simulator enrichment."""
        resampled = []
        for motion in motions:
            try:
                resampled.append(self._resampler.resample(motion))
            except Exception as exc:
                identifier = self._motion_identifier(motion)
                raise type(exc)(f"{identifier}: {exc}") from exc
        return resampled

    @staticmethod
    def _motion_identifier(motion: ReferenceMotionState) -> str:
        return motion.name or motion.display_name or "<unnamed motion>"

    def enrich(
        self,
        motions: list[ReferenceMotionState],
    ) -> list[ReferenceMotionState]:
        """Enrich already-resampled clips with simulator-derived body fields."""
        if not motions:
            return []

        for motion in motions:
            self._validate_enrich_input(motion)

        adapter = self._get_enrichment_adapter()
        rich_motions = []
        for motion in motions:
            try:
                rich_motion = adapter.enrich(motion)
                self._validate_enrich_output(motion, rich_motion)
            except Exception as exc:
                identifier = self._motion_identifier(motion)
                raise type(exc)(f"{identifier}: {exc}") from exc
            rich_motions.append(rich_motion)
        return rich_motions

    def _validate_enrich_input(self, motion: ReferenceMotionState) -> None:
        if motion.fps != self.cfg.output_fps:
            identifier = self._motion_identifier(motion)
            raise ValueError(
                f"{identifier}: MotionLib.enrich expects resampled clips at "
                f"cfg.output_fps={self.cfg.output_fps}; got motion.fps={motion.fps}"
            )

        for field_name in self._ENRICH_INPUT_RICH_FIELDS:
            if getattr(motion, field_name) is not None:
                identifier = self._motion_identifier(motion)
                raise ValueError(
                    f"{identifier}: MotionLib.enrich accepts resampled "
                    "generalized-coordinate clips only; got existing rich field "
                    f"{field_name}"
                )

        if motion.foot_contacts is not None:
            identifier = self._motion_identifier(motion)
            raise ValueError(
                f"{identifier}: MotionLib.enrich v1 does not support foot_contacts"
            )

    def _validate_enrich_output(
        self,
        source_motion: ReferenceMotionState,
        rich_motion: object,
    ) -> None:
        if not isinstance(rich_motion, ReferenceMotionState):
            raise TypeError(
                "Astro enrichment adapter must return ReferenceMotionState objects"
            )
        for field_name in self._ENRICH_OUTPUT_BODY_FIELDS:
            if getattr(rich_motion, field_name) is None:
                identifier = self._motion_identifier(source_motion)
                raise ValueError(
                    f"{identifier}: Astro enrichment adapter did not fill {field_name}"
                )

    def _get_enrichment_adapter(self) -> Any:
        if self._enrichment_adapter is None:
            self._enrichment_adapter = AstroSimulatorEnrichmentAdapter.create(
                device=self.cfg.device
            )
        return self._enrichment_adapter
