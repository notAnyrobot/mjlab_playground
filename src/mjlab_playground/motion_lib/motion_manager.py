from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .motion_loader import ReferenceMotionState

ClipWeighting = Literal["uniform", "duration", "explicit"]
TimeSampling = Literal["start", "uniform", "adaptive"]


@dataclass(frozen=True, kw_only=True)
class MotionManagerCfg:
    """Configuration for stateless AMP-style reference motion sampling."""

    clip_weighting: ClipWeighting = "duration"
    time_sampling: TimeSampling = "uniform"
    history_seconds: float = 0.0
    future_seconds: float = 0.0
    clip_weights: torch.Tensor | None = None
    adaptive_num_bins: int = 10
    adaptive_uniform_floor: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.clip_weighting not in ("uniform", "duration", "explicit"):
            raise ValueError("clip_weighting must be 'uniform', 'duration', or 'explicit'")
        if self.time_sampling not in ("start", "uniform", "adaptive"):
            raise ValueError("time_sampling must be 'start', 'uniform', or 'adaptive'")
        for field_name in ("history_seconds", "future_seconds"):
            value = float(getattr(self, field_name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be non-negative and finite")
            object.__setattr__(self, field_name, value)
        if self.adaptive_num_bins <= 0:
            raise ValueError("adaptive_num_bins must be positive")
        adaptive_uniform_floor = float(self.adaptive_uniform_floor)
        if adaptive_uniform_floor <= 0.0 or not math.isfinite(adaptive_uniform_floor):
            raise ValueError("adaptive_uniform_floor must be positive and finite")
        object.__setattr__(self, "adaptive_uniform_floor", adaptive_uniform_floor)


@dataclass(frozen=True)
class ReferenceMotionSample:
    """Clip IDs and clip-local times for querying reference motion tensors."""

    motion_ids: torch.Tensor
    motion_times: torch.Tensor


class MotionManager:
    """Stateless sampler for AMP-style expert reference motion batches."""

    def __init__(
        self,
        motion: ReferenceMotionState,
        cfg: MotionManagerCfg | None = None,
    ) -> None:
        self.cfg = cfg or MotionManagerCfg()
        clip_starts, clip_lengths, clip_fps = self._package_clip_metadata(motion)
        self.clip_starts = clip_starts
        self.clip_lengths = clip_lengths
        self.clip_fps = clip_fps

        dtype = clip_fps.dtype
        device = clip_fps.device
        clip_lengths_float = clip_lengths.to(dtype=dtype)
        clip_durations = (clip_lengths_float - 1.0) / clip_fps
        self.valid_start_times = torch.full(
            clip_durations.shape,
            self.cfg.history_seconds,
            dtype=dtype,
            device=device,
        )
        self.valid_end_times = clip_durations - self.cfg.future_seconds
        self.valid_clip_mask = self.valid_end_times >= self.valid_start_times
        if not self.valid_clip_mask.any().item():
            raise ValueError("No clips have a valid sampling window")

        self._clip_probabilities = self._build_clip_probabilities()
        self.adaptive_failure_pressure = torch.zeros(
            (int(self.valid_start_times.numel()), self.cfg.adaptive_num_bins),
            dtype=dtype,
            device=device,
        )
        self._generator = None
        if self.cfg.seed is not None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.cfg.seed)

    def sample_batch(self, batch_size: int) -> ReferenceMotionSample:
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        motion_ids = torch.multinomial(
            self._clip_probabilities,
            batch_size,
            replacement=True,
            generator=self._generator,
        )
        valid_starts = self.valid_start_times[motion_ids]
        if self.cfg.time_sampling == "start":
            motion_times = valid_starts
        elif self.cfg.time_sampling == "adaptive":
            motion_times = self._sample_adaptive_times(motion_ids)
        else:
            valid_ends = self.valid_end_times[motion_ids]
            fractions = torch.rand(
                batch_size,
                dtype=valid_starts.dtype,
                device=valid_starts.device,
                generator=self._generator,
            )
            motion_times = valid_starts + fractions * (valid_ends - valid_starts)
        return ReferenceMotionSample(motion_ids=motion_ids, motion_times=motion_times)

    def _sample_adaptive_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        if motion_ids.numel() == 0:
            return torch.empty(
                0,
                dtype=self.valid_start_times.dtype,
                device=self.valid_start_times.device,
            )
        bin_ids = torch.empty_like(motion_ids)
        for clip_id in torch.unique(motion_ids):
            clip_mask = motion_ids == clip_id
            bin_weights = (
                self.adaptive_failure_pressure[clip_id]
                + self.cfg.adaptive_uniform_floor
            )
            bin_ids[clip_mask] = torch.multinomial(
                bin_weights,
                int(clip_mask.sum().item()),
                replacement=True,
                generator=self._generator,
            )

        valid_starts = self.valid_start_times[motion_ids]
        valid_ends = self.valid_end_times[motion_ids]
        window_lengths = valid_ends - valid_starts
        bin_widths = window_lengths / self.cfg.adaptive_num_bins
        fractions = torch.rand(
            int(motion_ids.numel()),
            dtype=valid_starts.dtype,
            device=valid_starts.device,
            generator=self._generator,
        )
        bin_offsets = bin_ids.to(valid_starts.dtype) + fractions
        motion_times = valid_starts + bin_offsets * bin_widths
        return torch.minimum(motion_times, valid_ends)

    def _build_clip_probabilities(self) -> torch.Tensor:
        if self.cfg.clip_weighting == "uniform":
            weights = self.valid_clip_mask.to(dtype=self.clip_fps.dtype)
        elif self.cfg.clip_weighting == "duration":
            weights = (self.valid_end_times - self.valid_start_times).clamp_min(0.0)
        elif self.cfg.clip_weighting == "explicit":
            if self.cfg.clip_weights is None:
                raise ValueError(
                    "clip_weights must be provided when clip_weighting='explicit'"
                )
            weights = torch.as_tensor(
                self.cfg.clip_weights,
                dtype=self.clip_fps.dtype,
                device=self.clip_fps.device,
            )
            if weights.shape != self.clip_fps.shape:
                raise ValueError("clip_weights must match the package clip count")
            if not torch.isfinite(weights).all().item() or (weights < 0.0).any().item():
                raise ValueError("clip_weights must be non-negative and finite")
        else:
            raise NotImplementedError(
                f"clip_weighting={self.cfg.clip_weighting!r} is not implemented"
            )
        weights = torch.where(self.valid_clip_mask, weights, torch.zeros_like(weights))
        if weights.sum() <= 0.0:
            raise ValueError("clip weighting produced no positive probability mass")
        return weights / weights.sum()

    @staticmethod
    def _package_clip_metadata(
        motion: ReferenceMotionState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            motion.clip_starts is None
            and motion.clip_lengths is None
            and motion.clip_fps is None
        ):
            return (
                torch.tensor([0], dtype=torch.long, device=motion.root_pos.device),
                torch.tensor(
                    [motion.root_pos.shape[0]],
                    dtype=torch.long,
                    device=motion.root_pos.device,
                ),
                torch.tensor(
                    [motion.fps],
                    dtype=motion.root_pos.dtype,
                    device=motion.root_pos.device,
                ),
            )
        if (
            motion.clip_starts is None
            or motion.clip_lengths is None
            or motion.clip_fps is None
        ):
            raise ValueError(
                "clip_starts, clip_lengths, and clip_fps must be provided together"
            )
        return motion.clip_starts, motion.clip_lengths, motion.clip_fps


class MimicMotionManager(MotionManager):
    """Stateful sampler that owns per-environment mimic motion tracks."""

    def __init__(
        self,
        motion: ReferenceMotionState,
        num_envs: int,
        cfg: MotionManagerCfg | None = None,
    ) -> None:
        super().__init__(motion, cfg)
        if num_envs < 0:
            raise ValueError("num_envs must be non-negative")
        self.motion_ids = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=self.valid_start_times.device,
        )
        self.motion_times = torch.zeros(
            num_envs,
            dtype=self.valid_start_times.dtype,
            device=self.valid_start_times.device,
        )

    def sample_envs(self, env_ids: torch.Tensor) -> ReferenceMotionSample:
        env_ids = self._normalize_env_ids(env_ids)
        sample = self.sample_batch(int(env_ids.numel()))
        self.motion_ids[env_ids] = sample.motion_ids
        self.motion_times[env_ids] = sample.motion_times
        return sample

    def advance_envs(self, dt: float) -> None:
        dt = float(dt)
        if dt < 0.0 or not math.isfinite(dt):
            raise ValueError("dt must be non-negative and finite")
        active = self.motion_ids >= 0
        self.motion_times[active] += dt

    def done_envs(self, lookahead: float = 0.0) -> torch.Tensor:
        lookahead = float(lookahead)
        if lookahead < 0.0 or not math.isfinite(lookahead):
            raise ValueError("lookahead must be non-negative and finite")
        active = self.motion_ids >= 0
        done = torch.zeros_like(active)
        active_motion_ids = self.motion_ids[active]
        done[active] = (
            self.motion_times[active] + lookahead
            > self.valid_end_times[active_motion_ids]
        )
        return done

    def report_env_outcomes(self, env_ids: torch.Tensor, *, failed: torch.Tensor) -> None:
        env_ids = self._normalize_env_ids(env_ids)
        failed = torch.as_tensor(
            failed,
            dtype=torch.bool,
            device=self.motion_ids.device,
        )
        if failed.shape != env_ids.shape:
            raise ValueError("failed must match env_ids shape")
        if env_ids.numel() == 0:
            return

        failed_env_ids = env_ids[failed]
        if failed_env_ids.numel() == 0:
            return
        active = self.motion_ids[failed_env_ids] >= 0
        failed_env_ids = failed_env_ids[active]
        if failed_env_ids.numel() == 0:
            return

        motion_ids = self.motion_ids[failed_env_ids]
        motion_times = self.motion_times[failed_env_ids]
        bin_ids = self._adaptive_bin_ids(motion_ids, motion_times)
        for motion_id, bin_id in zip(motion_ids.tolist(), bin_ids.tolist(), strict=True):
            self.adaptive_failure_pressure[motion_id, bin_id] += 1.0

    def _adaptive_bin_ids(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> torch.Tensor:
        valid_starts = self.valid_start_times[motion_ids]
        valid_ends = self.valid_end_times[motion_ids]
        window_lengths = valid_ends - valid_starts
        safe_lengths = torch.where(
            window_lengths > 0.0,
            window_lengths,
            torch.ones_like(window_lengths),
        )
        fractions = ((motion_times - valid_starts) / safe_lengths).clamp(0.0, 1.0)
        bin_ids = torch.floor(fractions * self.cfg.adaptive_num_bins).to(torch.long)
        return torch.clamp(bin_ids, max=self.cfg.adaptive_num_bins - 1)

    def _normalize_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, device=self.motion_ids.device)
        integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
        if env_ids.dtype not in integer_dtypes:
            raise TypeError("env_ids must be an integer tensor")
        env_ids = env_ids.to(dtype=torch.long)
        if env_ids.ndim != 1:
            raise ValueError("env_ids must be a 1-D tensor")
        if env_ids.numel() == 0:
            return env_ids
        if ((env_ids < 0) | (env_ids >= self.motion_ids.numel())).any().item():
            raise ValueError("env_ids contains out-of-range environment IDs")
        return env_ids
