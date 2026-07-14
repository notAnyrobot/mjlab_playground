from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from mjlab_playground.motion_lib.motion_viewer import RecordingStatus


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

    def as_request(self) -> RecordingRequest:
        """Bind planned outputs to the corresponding loaded motion indices."""
        return RecordingRequest(
            targets=tuple(
                RecordingRequestTarget(
                    motion_index=motion_index,
                    source_path=target.source_path,
                    output_path=target.output_path,
                )
                for motion_index, target in enumerate(self.targets)
            )
        )


@dataclass(frozen=True, kw_only=True)
class RecordingRequestTarget:
    """One already-planned deterministic recording target."""

    motion_index: int
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

    def record_selected_clip(self, motion_index: int, motion: Any) -> None:
        del motion
        if self._process is not None:
            self._error = f"recording already running: {self._output_path}"
            return
        target = self._targets[motion_index]
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
    """Create isolated background recording that invokes deterministic headless mode."""
    if process_launcher is None:

        def launch_process(command: list[str]) -> BackgroundProcess:
            return subprocess.Popen(command)

        process_launcher = launch_process

    def start_process(request: RecordingRequest) -> BackgroundProcess:
        target = request.targets[0]
        command = [
            sys.executable,
            "-m",
            "mjlab_playground.motion_lib.scripts.launch_motion_viewer",
            "--motion-files",
            str(target.source_path),
            "--format",
            motion_format,
            "--fps",
            str(float(fps)),
            "--robot",
            robot,
            "--device",
            device,
            "--headless",
            "--record-video",
            "--output-dir",
            str(target.output_path.parent),
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
        for target in request.targets:
            if target.motion_index < 0 or target.motion_index >= len(motions):
                raise IndexError(f"motion_index out of range: {target.motion_index}")

        reporter = status_reporter or self._status_reporter
        renderer = self._renderer_factory()
        results: list[RecordingResult] = []
        try:
            for target_number, target in enumerate(request.targets, start=1):
                motion = motions[target.motion_index]
                motion_name = _motion_name(motion)
                motion_context = (
                    f"motion {target_number}/{len(request.targets)} ({motion_name})"
                )
                frame_count = _frame_count(motion)
                fps = _motion_fps(motion)
                attachment = self._attachment_factory()
                _report(
                    reporter,
                    f"Recording {target_number}/{len(request.targets)}: "
                    f"{motion_name} -> {target.output_path}",
                )
                attachment.start(target.output_path, fps=fps)
                try:
                    for frame_index in range(frame_count):
                        frame_context = f"frame {frame_index + 1}/{frame_count} for {motion_context}"
                        try:
                            self._scene.apply_reference_frame(motion, frame_index)
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
                            f"Recording {motion_name}: frame "
                            f"{frame_index + 1}/{frame_count}",
                        )
                    try:
                        result = attachment.stop()
                    except Exception as exc:
                        raise DeterministicRecordingError(
                            f"failed to write recording for {motion_context} "
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


def _frame_count(motion: Any) -> int:
    try:
        frame_count = int(motion.root_pos.shape[0])
    except AttributeError as exc:
        raise TypeError("reference motion must expose root_pos frames") from exc
    if frame_count <= 0:
        raise ValueError("reference motion must contain at least one frame")
    return frame_count


def _motion_name(motion: Any) -> str:
    display_name = getattr(motion, "display_name", None)
    if display_name:
        return str(display_name)
    name = getattr(motion, "name", None)
    if name:
        return str(name)
    return "unnamed"


def _motion_fps(motion: Any) -> float:
    fps = float(getattr(motion, "fps", 30.0))
    if fps <= 0.0 or not math.isfinite(fps):
        raise ValueError("reference motion fps must be positive and finite")
    return fps


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
    planned_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else (motion_dir.parent / "renderings" / run_timestamp)
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
