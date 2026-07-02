from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True, kw_only=True)
class RecordingResult:
    output_path: Path
    frame_count: int
    fps: float
    written: bool
    reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class RecordingTarget:
    source_path: Path
    output_path: Path


@dataclass(frozen=True, kw_only=True)
class RecordingOutputPlan:
    output_dir: Path
    targets: tuple[RecordingTarget, ...]


class RecordingAttachment:
    """Buffer rendered frames for one reference-motion video recording."""

    def __init__(self, *, write_video: Callable[..., None] | None = None) -> None:
        self._write_video = write_video or _write_video_with_mediapy
        self._output_path: Path | None = None
        self._fps: float | None = None
        self._frames: list[Any] = []

    @property
    def is_recording(self) -> bool:
        return self._output_path is not None

    def start(self, output_path: str | Path, *, fps: float) -> None:
        path = Path(output_path)
        if path.suffix != ".mp4":
            raise ValueError(f"recording output path must end with .mp4, got {path}")
        if path.exists() and path.is_dir():
            raise ValueError(f"recording output path is a directory: {path}")
        if fps <= 0.0 or not math.isfinite(fps):
            raise ValueError("recording fps must be positive and finite")

        path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = path
        self._fps = float(fps)
        self._frames = []

    def capture(self, frame: Any) -> None:
        if self._output_path is None:
            raise RuntimeError("recording has not started")
        self._frames.append(frame)

    def stop(self) -> RecordingResult:
        if self._output_path is None or self._fps is None:
            raise RuntimeError("recording has not started")

        output_path = self._output_path
        fps = self._fps
        frames = list(self._frames)
        self._output_path = None
        self._fps = None
        self._frames = []

        if not frames:
            return RecordingResult(
                output_path=output_path,
                frame_count=0,
                fps=fps,
                written=False,
                reason="no frames captured",
            )

        self._write_video(output_path, frames, fps=fps)
        return RecordingResult(
            output_path=output_path,
            frame_count=len(frames),
            fps=fps,
            written=True,
        )


def _write_video_with_mediapy(path: Path, frames: Sequence[Any], *, fps: float) -> None:
    import mediapy

    mediapy.write_video(path, frames, fps=fps)


def plan_recording_outputs(
    motion_source: str | Path,
    *,
    timestamp: str | None = None,
    output_dir: str | Path | None = None,
    motion_names: Sequence[str] | None = None,
) -> RecordingOutputPlan:
    source = Path(motion_source)
    if source.is_file():
        motion_dir = source.parent
        motion_paths = [source]
    elif source.is_dir():
        motion_dir = source
        if motion_names is None:
            raise ValueError(
                "motion_names are required when planning recordings for a directory"
            )
        motion_paths = sorted(path for path in source.iterdir() if path.is_file())
        if not motion_paths:
            raise ValueError(f"No direct child motion files found in {source}")
    else:
        raise FileNotFoundError(f"Motion source does not exist: {source}")

    if motion_names is not None:
        motion_paths = []
        for motion_name in motion_names:
            motion_path = motion_dir / Path(motion_name).name
            if source.is_file() and motion_path.name == source.name:
                motion_path = source
            if not motion_path.is_file():
                raise FileNotFoundError(
                    f"Loaded motion {motion_name!r} was not found under {motion_dir}"
                )
            motion_paths.append(motion_path)

    run_timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    planned_output_dir = Path(output_dir) if output_dir is not None else (
        motion_dir.parent / "renderings" / run_timestamp
    )
    planned_output_dir.mkdir(parents=True, exist_ok=True)
    return RecordingOutputPlan(
        output_dir=planned_output_dir,
        targets=tuple(
            RecordingTarget(
                source_path=motion_path,
                output_path=planned_output_dir / f"{motion_path.stem}.mp4",
            )
            for motion_path in motion_paths
        ),
    )
