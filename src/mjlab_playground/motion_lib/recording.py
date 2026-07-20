from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

from mjlab_playground.motion_lib.motion_viewer import RecordingStatus

if TYPE_CHECKING:
    from mjlab_playground.motion_lib.motion_loader import ReferenceMotionClipSpan


@dataclass(frozen=True, kw_only=True)
class RecordingResult:
    output_path: Path
    frame_count: int
    fps: float
    written: bool
    reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class RecordingTarget:
    reference_motion_index: int
    clip_id: int
    source_path: Path
    output_path: Path


@dataclass(frozen=True, kw_only=True)
class RecordingOutputPlan:
    output_dir: Path
    targets: tuple[RecordingTarget, ...]

    def as_request(self) -> RecordingRequest:
        """Bind planned outputs to the corresponding loaded reference motions."""
        return RecordingRequest(
            targets=tuple(
                RecordingRequestTarget(
                    reference_motion_index=target.reference_motion_index,
                    clip_id=target.clip_id,
                    source_path=target.source_path,
                    output_path=target.output_path,
                )
                for target in self.targets
            )
        )


@dataclass(frozen=True, kw_only=True)
class RecordingRequestTarget:
    """One already-planned deterministic recording target."""

    reference_motion_index: int
    clip_id: int
    source_path: Path
    output_path: Path


@dataclass(frozen=True, kw_only=True)
class RecordingRequest:
    """An immutable request consumed by deterministic recording."""

    targets: tuple[RecordingRequestTarget, ...]


class BackgroundProcess(Protocol):
    """Nonblocking child-process status boundary."""

    def poll(self) -> int | None:
        """Return ``None`` while running, otherwise the child exit code."""

    def wait(self) -> int:
        """Wait for the child to exit and return its exit code."""
        ...

    def terminate(self) -> None:
        """Request graceful child termination."""
        ...

    def kill(self) -> None:
        """Force child termination after graceful cancellation fails."""
        ...


class SubprocessBackgroundRecorder:
    """Run one immutable deterministic recording request in a child process."""

    def __init__(
        self,
        *,
        targets: Sequence[RecordingRequestTarget],
        process_factory: Callable[[RecordingRequest], BackgroundProcess],
    ) -> None:
        self._targets = tuple(targets)
        self._process_factory = process_factory
        self._process: BackgroundProcess | None = None
        self._status = RecordingStatus.IDLE
        self._output_path: Path | None = None
        self._error: str | None = None

    @property
    def status(self) -> RecordingStatus:
        return self._status

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    @property
    def error(self) -> str | None:
        return self._error

    def record_selected_clip(
        self,
        reference_motion_index: int,
        clip: ReferenceMotionClipSpan,
    ) -> None:
        if self._process is not None:
            self._error = f"recording already running: {self._output_path}"
            return
        identity = (reference_motion_index, clip.clip_id)
        matching_targets = tuple(
            target
            for target in self._targets
            if (target.reference_motion_index, target.clip_id) == identity
        )
        if len(matching_targets) != 1:
            raise ValueError(
                "expected exactly one recording target for reference motion "
                f"{reference_motion_index}, clip {clip.clip_id}; "
                f"found {len(matching_targets)}"
            )
        target = matching_targets[0]
        request = RecordingRequest(targets=(target,))
        self._output_path = target.output_path
        self._error = None
        try:
            self._process = self._process_factory(request)
        except Exception as exc:
            self._process = None
            self._status = RecordingStatus.FAILED
            self._error = f"failed to start recording child: {exc}"
            return
        self._status = RecordingStatus.RUNNING

    def poll(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            returncode = process.poll()
        except Exception as exc:
            self._status = RecordingStatus.RUNNING
            self._error = f"failed to poll recording child: {exc}"
            return
        if returncode is None:
            return
        self._finish(returncode)

    def wait(self) -> None:
        """Wait for an active recording child and capture its result."""
        process = self._process
        if process is None:
            return
        try:
            returncode = process.wait()
        except Exception as exc:
            self._status = RecordingStatus.RUNNING
            self._error = f"failed to wait for recording child: {exc}"
            return
        self._finish(returncode)

    def cancel(self) -> None:
        """Cancel an active child and remove only its incomplete output."""
        process = self._process
        if process is None:
            return
        try:
            try:
                returncode = process.poll()
            except Exception:
                returncode = None
            if returncode is not None:
                self._finish(returncode)
                return
            process.terminate()
            process.wait()
        except Exception as graceful_exc:
            try:
                process.kill()
                process.wait()
            except Exception as force_exc:
                self._status = RecordingStatus.RUNNING
                self._error = (
                    "failed to cancel recording child: "
                    f"graceful termination failed ({graceful_exc}); "
                    f"forced termination failed ({force_exc})"
                )
                return

        self._process = None
        self._status = RecordingStatus.FAILED
        self._error = "recording cancelled"
        output_path = self._output_path
        if output_path is not None:
            try:
                output_path.unlink(missing_ok=True)
            except OSError as exc:
                self._error = (
                    f"recording cancelled; failed to remove incomplete output: {exc}"
                )

    def close(self) -> None:
        """Release recorder ownership after no child remains active."""
        if self._process is None:
            return
        self.cancel()
        if self._process is not None:
            detail = self._error or "child exit could not be confirmed"
            raise RuntimeError(f"failed to close recording child: {detail}")

    def _finish(self, returncode: int) -> None:
        self._process = None
        if returncode == 0:
            self._status = RecordingStatus.SUCCEEDED
            self._error = None
        else:
            self._status = RecordingStatus.FAILED
            self._error = f"recording child process exited with code {returncode}"


def create_subprocess_background_recorder(
    *,
    targets: Sequence[RecordingRequestTarget],
    motion_format: str,
    fps: float,
    robot: str,
    device: str,
    process_launcher: Callable[[list[str]], BackgroundProcess] | None = None,
) -> SubprocessBackgroundRecorder:
    """Create isolated background recording for one captured logical clip."""
    if process_launcher is None:

        def launch_process(command: list[str]) -> BackgroundProcess:
            return subprocess.Popen(command)

        process_launcher = launch_process

    def start_process(request: RecordingRequest) -> BackgroundProcess:
        target = request.targets[0]
        command = [
            sys.executable,
            "-m",
            "mjlab_playground.motion_lib.scripts._record_selected_clip",
            "--motion-file",
            str(target.source_path),
            "--format",
            motion_format,
            "--fps",
            str(float(fps)),
            "--robot",
            robot,
            "--device",
            device,
            "--clip-id",
            str(target.clip_id),
            "--output",
            str(target.output_path),
        ]
        return process_launcher(command)

    return SubprocessBackgroundRecorder(
        targets=targets,
        process_factory=start_process,
    )


class RecordingScene(Protocol):
    def apply_reference_frame(self, motion: Any, frame_index: int) -> None:
        """Apply one frame of an already-loaded reference motion."""


class FrameRenderer(Protocol):
    def render_frame(self) -> Any:
        """Render the currently applied scene frame."""

    def close(self) -> None:
        """Release rendering resources."""


class DeterministicRecordingError(RuntimeError):
    """A deterministic recording failed at a labeled operation boundary."""


class DeterministicRecorder:
    """Record planned reference-motion targets independently of viewer state."""

    def __init__(
        self,
        *,
        scene: RecordingScene,
        renderer_factory: Callable[[], FrameRenderer],
        attachment_factory: Callable[[], RecordingAttachment],
        status_reporter: Callable[[str], None] | None = None,
    ) -> None:
        self._scene = scene
        self._renderer_factory = renderer_factory
        self._attachment_factory = attachment_factory
        self._status_reporter = status_reporter

    def record(
        self,
        motions: Sequence[Any],
        request: RecordingRequest,
        *,
        status_reporter: Callable[[str], None] | None = None,
    ) -> tuple[RecordingResult, ...]:
        """Record every requested clip from frame zero through its final frame."""
        if not request.targets:
            raise ValueError("recording request must contain at least one target")
        resolved_targets: list[tuple[RecordingRequestTarget, Any]] = []
        for target in request.targets:
            if (
                target.reference_motion_index < 0
                or target.reference_motion_index >= len(motions)
            ):
                raise IndexError(
                    "reference_motion_index out of range: "
                    f"{target.reference_motion_index}"
                )
            motion = motions[target.reference_motion_index]
            span = next(
                (
                    candidate
                    for candidate in motion.iter_clip_spans()
                    if candidate.clip_id == target.clip_id
                ),
                None,
            )
            if span is None:
                raise IndexError(
                    f"clip_id {target.clip_id} out of range for reference motion "
                    f"{target.reference_motion_index}"
                )
            resolved_targets.append((target, span))

        reporter = status_reporter or self._status_reporter
        renderer = self._renderer_factory()
        results: list[RecordingResult] = []
        try:
            for target_number, (target, span) in enumerate(resolved_targets, start=1):
                motion = span.parent
                clip_name = span.name or _motion_name(motion)
                clip_context = (
                    f"clip {target_number}/{len(request.targets)} ({clip_name})"
                )
                frame_count = span.frame_count
                fps = span.fps
                attachment = self._attachment_factory()
                _report(
                    reporter,
                    f"Recording {target_number}/{len(request.targets)}: "
                    f"{clip_name} -> {target.output_path}",
                )
                attachment.start(target.output_path, fps=fps)
                try:
                    for local_frame_index in range(frame_count):
                        frame_context = (
                            f"frame {local_frame_index + 1}/{frame_count} "
                            f"for {clip_context}"
                        )
                        try:
                            self._scene.apply_reference_frame(
                                motion,
                                span.to_packed_frame(local_frame_index),
                            )
                        except Exception as exc:
                            raise DeterministicRecordingError(
                                f"failed to apply {frame_context}"
                            ) from exc
                        try:
                            frame = renderer.render_frame()
                        except Exception as exc:
                            raise DeterministicRecordingError(
                                f"failed to render {frame_context}"
                            ) from exc
                        try:
                            attachment.capture(frame)
                        except Exception as exc:
                            raise DeterministicRecordingError(
                                f"failed to capture {frame_context}"
                            ) from exc
                        _report(
                            reporter,
                            f"Recording {clip_name}: frame "
                            f"{local_frame_index + 1}/{frame_count}",
                        )
                    try:
                        result = attachment.stop()
                    except Exception as exc:
                        raise DeterministicRecordingError(
                            f"failed to write recording for {clip_context} "
                            f"to {target.output_path}"
                        ) from exc
                except Exception:
                    attachment.abort()
                    raise
                results.append(result)
                if result.written:
                    _report(reporter, f"Saved recording: {result.output_path}")
                else:
                    _report(
                        reporter,
                        f"Skipped recording: {result.reason or 'no frames captured'}",
                    )
        finally:
            renderer.close()

        return tuple(results)


class _MjlabOffscreenFrameRenderer:
    """Adapter around mjlab's public offscreen-rendering modules."""

    def __init__(
        self,
        scene: Any,
        *,
        offscreen_renderer_cls: Any,
        viewer_config_cls: Any,
    ) -> None:
        cfg = viewer_config_cls(
            height=480,
            width=640,
            origin_type=viewer_config_cls.OriginType.ASSET_ROOT,
            entity_name="robot",
            distance=2.0,
            elevation=-5.0,
            azimuth=20.0,
        )
        sim = scene.sim
        self._renderer = offscreen_renderer_cls(
            model=scene.mj_model,
            cfg=cfg,
            scene=scene.scene,
            sim_model=getattr(sim, "model", None),
        )
        self._sim_data = sim.data
        self._renderer.initialize()

    def render_frame(self) -> Any:
        self._renderer.update(self._sim_data)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


def create_mjlab_deterministic_recorder(
    scene: RecordingScene,
    *,
    attachment_factory: Callable[[], RecordingAttachment] | None = None,
    status_reporter: Callable[[str], None] | None = None,
    offscreen_renderer_cls: Any | None = None,
    viewer_config_cls: Any | None = None,
) -> DeterministicRecorder:
    """Create a recorder whose renderer remains lazy until ``record()``."""
    if offscreen_renderer_cls is None or viewer_config_cls is None:
        from mjlab.viewer import ViewerConfig
        from mjlab.viewer.offscreen_renderer import OffscreenRenderer

        offscreen_renderer_cls = OffscreenRenderer
        viewer_config_cls = ViewerConfig
    if attachment_factory is None:
        attachment_factory = RecordingAttachment

    return DeterministicRecorder(
        scene=scene,
        renderer_factory=lambda: _MjlabOffscreenFrameRenderer(
            scene,
            offscreen_renderer_cls=offscreen_renderer_cls,
            viewer_config_cls=viewer_config_cls,
        ),
        attachment_factory=attachment_factory,
        status_reporter=status_reporter,
    )


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

        if not frames:
            self._output_path = None
            self._fps = None
            self._frames = []
            return RecordingResult(
                output_path=output_path,
                frame_count=0,
                fps=fps,
                written=False,
                reason="no frames captured",
            )

        self._write_video(output_path, frames, fps=fps)
        self._output_path = None
        self._fps = None
        self._frames = []
        return RecordingResult(
            output_path=output_path,
            frame_count=len(frames),
            fps=fps,
            written=True,
        )

    def abort(self) -> None:
        """Discard buffered frames and remove an incomplete output artifact."""
        output_path = self._output_path
        self._output_path = None
        self._fps = None
        self._frames = []
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def _report(reporter: Callable[[str], None] | None, message: str) -> None:
    if reporter is not None:
        reporter(message)


def _motion_name(motion: Any) -> str:
    name = getattr(motion, "name", None)
    if name:
        return str(name)
    return "unnamed"


def _write_video_with_mediapy(path: Path, frames: Sequence[Any], *, fps: float) -> None:
    import mediapy

    mediapy.write_video(path, frames, fps=fps)


def plan_recording_outputs(
    motion_source: str | Path,
    *,
    timestamp: str | None = None,
    output_dir: str | Path | None = None,
    motion_names: Sequence[str] | None = None,
    reference_motions: Sequence[Any] | None = None,
) -> RecordingOutputPlan:
    source = Path(motion_source)
    if source.is_file():
        motion_dir = source.parent
        motion_paths = [source]
    elif source.is_dir():
        motion_dir = source
        if motion_names is None and reference_motions is None:
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
    planned_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else (motion_dir.parent / "renderings" / run_timestamp)
    )
    if reference_motions is None:
        targets = tuple(
            RecordingTarget(
                reference_motion_index=reference_motion_index,
                clip_id=0,
                source_path=motion_path,
                output_path=planned_output_dir / f"{motion_path.stem}.mp4",
            )
            for reference_motion_index, motion_path in enumerate(motion_paths)
        )
    else:
        if motion_names is not None:
            raise ValueError(
                "reference_motions and motion_names cannot both be provided"
            )
        loaded_motions = tuple(reference_motions)
        if not loaded_motions:
            raise ValueError("reference_motions must contain at least one value")
        motion_paths = _loaded_reference_motion_paths(source, loaded_motions)
        targets = tuple(
            RecordingTarget(
                reference_motion_index=reference_motion_index,
                clip_id=span.clip_id,
                source_path=motion_path,
                output_path=planned_output_dir
                / f"{Path(span.name or motion_path.name).stem}.mp4",
            )
            for reference_motion_index, (motion, motion_path) in enumerate(
                zip(loaded_motions, motion_paths, strict=True)
            )
            for span in motion.iter_clip_spans()
        )

    output_paths = [target.output_path for target in targets]
    if len(set(output_paths)) != len(output_paths):
        duplicate_names = sorted(
            path.name for path in set(output_paths) if output_paths.count(path) > 1
        )
        raise ValueError(
            "duplicate recording output names: " + ", ".join(duplicate_names)
        )

    planned_output_dir.mkdir(parents=True, exist_ok=True)
    return RecordingOutputPlan(
        output_dir=planned_output_dir,
        targets=targets,
    )


def _loaded_reference_motion_paths(
    source: Path,
    reference_motions: Sequence[Any],
) -> tuple[Path, ...]:
    if source.is_file():
        if len(reference_motions) != 1:
            raise ValueError(
                "a motion file must load exactly one reference motion for recording"
            )
        return (source,)

    paths: list[Path] = []
    for motion in reference_motions:
        name = getattr(motion, "name", None)
        if not name:
            raise ValueError("loaded reference motions must expose names for recording")
        path = source / Path(str(name)).name
        if not path.is_file():
            raise FileNotFoundError(
                f"Loaded motion {str(name)!r} was not found under {source}"
            )
        paths.append(path)
    return tuple(paths)
