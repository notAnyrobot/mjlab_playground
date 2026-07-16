from __future__ import annotations

import math
import shutil
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, TextIO

if TYPE_CHECKING:
    from mjlab_playground.motion_lib.recording import RecordingRequest, RecordingResult

PLAYBACK_SPEEDS = (0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DEFAULT_PLAYBACK_SPEED_INDEX = PLAYBACK_SPEEDS.index(1.0)
STATUS_PROGRESS_BAR_WIDTH = 20


class PlaybackAction(Enum):
    """Viewer-independent user intent submitted with one session tick."""

    TOGGLE_PAUSE = "toggle_pause"
    PREVIOUS_MOTION = "previous_motion"
    NEXT_MOTION = "next_motion"
    SLOWER = "slower"
    FASTER = "faster"
    RECORD_SELECTED_CLIP = "record_selected_clip"
    STOP = "stop"


class RecordingStatus(Enum):
    """Presentation-ready state of interactive background recording."""

    DISABLED = "disabled"
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ViewerTick:
    """One ordered interaction submitted by a concrete viewer adapter."""

    elapsed_seconds: float
    actions: tuple[PlaybackAction, ...] = ()


@dataclass(frozen=True)
class ViewerSnapshot:
    """Immutable presentation state returned after one core update."""

    status: str
    selected_motion_index: int
    selected_motion_name: str
    motion_count: int
    frame_index: int
    frame_count: int
    playback_speed: float
    paused: bool
    recording_status: RecordingStatus
    stop_requested: bool
    recording_output_path: Path | None = None
    recording_error: str | None = None


class MotionScene(Protocol):
    """Narrow scene seam required by the presentation-agnostic core."""

    def apply_reference_frame(self, motion: Any, frame_index: int) -> None:
        """Apply one reference frame to the authoritative robot scene."""


class ViewerAdapter(Protocol):
    """Presentation lifecycle driven by the core update function."""

    def run(
        self,
        update: Callable[[ViewerTick], ViewerSnapshot],
    ) -> None:
        """Run the presentation loop until it stops or an update fails."""

    def close(self) -> None:
        """Close presentation resources."""


class BackgroundRecorder(Protocol):
    """Minimal recording status seam used during session expansion."""

    @property
    def status(self) -> RecordingStatus:
        """Return the latest presentation-ready recording status."""
        ...

    def record_selected_clip(self, motion_index: int, motion: Any) -> None:
        """Request deterministic recording of the selected reference clip."""
        ...

    def poll(self) -> None:
        """Refresh child status without blocking the presentation loop."""
        ...

    def wait(self) -> None:
        """Wait for active recording work to finish."""
        ...

    def cancel(self) -> None:
        """Cancel active recording work and remove its incomplete output."""
        ...

    def close(self) -> None:
        """Release recorder resources after work has finished."""
        ...

    @property
    def output_path(self) -> Path | None:
        """Return the captured output target, if any."""
        ...

    @property
    def error(self) -> str | None:
        """Return recoverable recording failure or rejection context."""
        ...


class DeterministicRecordingOperation(Protocol):
    """Recording-module interface consumed by the deep viewer."""

    def record(
        self,
        motions: Sequence[Any],
        request: RecordingRequest,
    ) -> tuple[RecordingResult, ...]:
        """Record an already-planned request."""
        ...


class ViewerError(RuntimeError):
    """Fatal reference-motion session failure."""


class PlaybackController:
    """Pure playback state for source-agnostic reference motion clips."""

    def __init__(self, *, motion_count: int) -> None:
        if motion_count <= 0:
            raise ValueError("motion_count must be positive")
        self.motion_count = motion_count
        self.selected_motion_index = 0
        self.frame_position = 0.0
        self.paused = False
        self.playback_speed_index = DEFAULT_PLAYBACK_SPEED_INDEX

    @property
    def playback_speed(self) -> float:
        return PLAYBACK_SPEEDS[self.playback_speed_index]

    def increase_speed(self) -> None:
        self.playback_speed_index = min(
            self.playback_speed_index + 1,
            len(PLAYBACK_SPEEDS) - 1,
        )

    def decrease_speed(self) -> None:
        self.playback_speed_index = max(self.playback_speed_index - 1, 0)

    def select_next_motion(self) -> None:
        self.selected_motion_index = (
            self.selected_motion_index + 1
        ) % self.motion_count
        self.reset_frame()

    def select_previous_motion(self) -> None:
        self.selected_motion_index = (
            self.selected_motion_index - 1
        ) % self.motion_count
        self.reset_frame()

    def reset_frame(self) -> None:
        self.frame_position = 0.0

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def advance(self, *, frame_count: int, frames: float = 1.0) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.paused:
            return
        if not math.isfinite(frames):
            raise ValueError("frames must be finite")
        self.frame_position = (
            self.frame_position + frames * self.playback_speed
        ) % frame_count

    def current_frame_index(self, *, frame_count: int) -> int:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        return int(self.frame_position) % frame_count


class TerminalStatusReporter:
    """Render playback status as a single updating terminal line."""

    def __init__(
        self, stream: TextIO | None = None, max_width: int | None = None
    ) -> None:
        self.stream = stream or sys.stdout
        self.max_width = max_width
        self._last_status: str | None = None

    def __call__(self, status: str) -> None:
        status = self._fit_status(status)
        if status == self._last_status:
            return
        padding = ""
        if self._last_status is not None:
            padding = " " * max(0, len(self._last_status) - len(status))
        print(f"\r{status}{padding}", end="", file=self.stream, flush=True)
        self._last_status = status

    def _fit_status(self, status: str) -> str:
        max_width = self.max_width
        if max_width is None:
            max_width = shutil.get_terminal_size(fallback=(120, 24)).columns
        if max_width <= 0 or len(status) <= max_width:
            return status

        ellipsis = " ... "
        if max_width <= len(ellipsis):
            return status[:max_width]

        min_head_width = min(16, max_width - len(ellipsis))
        tail_width = min(48, max_width - len(ellipsis) - min_head_width)
        head_width = max_width - len(ellipsis) - tail_width
        return f"{status[:head_width]}{ellipsis}{status[-tail_width:]}"

    def close(self) -> None:
        if self._last_status is not None:
            print(file=self.stream, flush=True)
            self._last_status = None


class MotionViewer:
    """Source-agnostic reference motion viewer."""

    def __init__(
        self,
        motions: Sequence[Any],
        motion_scene: MotionScene,
        *,
        viewer_adapter: ViewerAdapter | None = None,
        background_recorder: BackgroundRecorder | None = None,
        deterministic_recorder: DeterministicRecordingOperation | None = None,
        status_reporter: Callable[[str], None] | None = None,
    ) -> None:
        if not motions:
            raise ValueError("MotionViewer requires at least one reference motion")
        self.motions = list(motions)
        self._motion_scene = motion_scene
        self.controller = PlaybackController(motion_count=len(self.motions))
        self._viewer_adapter = viewer_adapter
        self._background_recorder = background_recorder
        self._deterministic_recorder = deterministic_recorder
        self._status_reporter = status_reporter
        self._stop_requested = False

    @property
    def selected_motion(self) -> Any:
        return self.motions[self.controller.selected_motion_index]

    def run(self) -> None:
        """Run one presentation-agnostic interactive playback session."""
        if self._viewer_adapter is None:
            raise ValueError("viewer_adapter is required for MotionViewer.run()")

        try:
            try:
                self._viewer_adapter.run(self._update_session)
            except KeyboardInterrupt:
                self._apply_session_action(PlaybackAction.STOP)
            except ViewerError:
                raise
            except Exception as exc:
                self._stop_requested = True
                raise ViewerError("reference motion viewer session failed") from exc
            finally:
                primary_error = sys.exception()
                try:
                    self._viewer_adapter.close()
                except Exception as exc:
                    if primary_error is None:
                        raise ViewerError("failed to close viewer adapter") from exc
                    warnings.warn(
                        f"failed to close viewer adapter: {exc}",
                        stacklevel=2,
                    )
        finally:
            try:
                self._wait_for_background_recording()
            finally:
                self._close_session_resources(primary_error=sys.exception())

    def _wait_for_background_recording(self) -> None:
        recorder = self._background_recorder
        if recorder is None or recorder.status is not RecordingStatus.RUNNING:
            return

        output_path = recorder.output_path
        context = f": {output_path}" if output_path is not None else ""
        self._report_shutdown_status(f"Waiting for recording{context}")
        try:
            recorder.wait()
        except KeyboardInterrupt:
            self._cancel_background_recording(recorder)
        except Exception as exc:
            warnings.warn(
                f"failed to wait for background recording: {exc}",
                stacklevel=2,
            )
            self._cancel_background_recording(recorder)
        else:
            if recorder.status is RecordingStatus.RUNNING:
                self._cancel_background_recording(recorder)

        if recorder.status is RecordingStatus.SUCCEEDED:
            self._report_shutdown_status(f"Recording succeeded{context}")
        elif recorder.status is RecordingStatus.FAILED:
            error_context = f": {recorder.error}" if recorder.error else ""
            self._report_shutdown_status(f"Recording failed{error_context}")

    @staticmethod
    def _cancel_background_recording(recorder: BackgroundRecorder) -> None:
        try:
            recorder.cancel()
        except Exception as exc:
            warnings.warn(
                f"failed to cancel background recording: {exc}",
                stacklevel=2,
            )

    def _report_shutdown_status(self, status: str) -> None:
        if self._status_reporter is None:
            return
        try:
            self._status_reporter(status)
        except Exception as exc:
            warnings.warn(
                f"failed to report viewer shutdown status: {exc}", stacklevel=2
            )

    def _close_session_resources(
        self,
        *,
        primary_error: BaseException | None,
    ) -> None:
        recorder_close_error: Exception | None = None
        for resource in (
            self._background_recorder,
            self._motion_scene,
            self._status_reporter,
        ):
            close = getattr(resource, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    if resource is self._background_recorder:
                        recorder_close_error = exc
                    else:
                        warnings.warn(
                            f"failed to close viewer session resource: {exc}",
                            stacklevel=2,
                        )

        if recorder_close_error is None:
            return
        if primary_error is None:
            raise ViewerError("failed to close background recorder") from (
                recorder_close_error
            )
        warnings.warn(
            f"failed to close background recorder: {recorder_close_error}",
            stacklevel=2,
        )

    def record(self, request: RecordingRequest) -> tuple[RecordingResult, ...]:
        """Execute one already-planned deterministic recording request."""
        if self._deterministic_recorder is None:
            raise ValueError(
                "deterministic_recorder is required for MotionViewer.record()"
            )
        return self._deterministic_recorder.record(self.motions, request)

    def _update_session(self, tick: ViewerTick) -> ViewerSnapshot:
        elapsed_seconds = float(tick.elapsed_seconds)
        if elapsed_seconds < 0.0 or not math.isfinite(elapsed_seconds):
            cause = ValueError("elapsed_seconds must be non-negative and finite")
            raise ViewerError(
                f"invalid viewer tick elapsed_seconds: {tick.elapsed_seconds!r}"
            ) from cause

        for action in tick.actions:
            self._apply_session_action(action)

        if self._background_recorder is not None:
            self._background_recorder.poll()

        if not self._stop_requested:
            self.controller.advance(
                frame_count=self._frame_count(self.selected_motion),
                frames=elapsed_seconds * self._motion_fps(self.selected_motion),
            )
            frame_count = self._frame_count(self.selected_motion)
            frame_index = self.controller.current_frame_index(frame_count=frame_count)
            try:
                self._motion_scene.apply_reference_frame(
                    self.selected_motion,
                    frame_index,
                )
            except Exception as exc:
                self._stop_requested = True
                motion_number = self.controller.selected_motion_index + 1
                motion_name = self._motion_name(self.selected_motion)
                raise ViewerError(
                    f"failed to apply frame {frame_index} from motion "
                    f"{motion_number}/{len(self.motions)} ({motion_name})"
                ) from exc

        return self._session_snapshot()

    def _apply_session_action(self, action: PlaybackAction) -> None:
        if action is PlaybackAction.TOGGLE_PAUSE:
            self.controller.toggle_pause()
        elif action is PlaybackAction.PREVIOUS_MOTION:
            self.controller.select_previous_motion()
        elif action is PlaybackAction.NEXT_MOTION:
            self.controller.select_next_motion()
        elif action is PlaybackAction.SLOWER:
            self.controller.decrease_speed()
        elif action is PlaybackAction.FASTER:
            self.controller.increase_speed()
        elif action is PlaybackAction.RECORD_SELECTED_CLIP:
            if self._background_recorder is not None:
                motion_index = self.controller.selected_motion_index
                self._background_recorder.record_selected_clip(
                    motion_index,
                    self.motions[motion_index],
                )
        elif action is PlaybackAction.STOP:
            self._stop_requested = True

    def _session_snapshot(self) -> ViewerSnapshot:
        motion_index = self.controller.selected_motion_index
        motion = self.selected_motion
        frame_count = self._frame_count(motion)
        frame_index = self.controller.current_frame_index(frame_count=frame_count)
        recording_status = RecordingStatus.DISABLED
        recording_output_path = None
        recording_error = None
        if self._background_recorder is not None:
            recording_status = self._background_recorder.status
            recording_output_path = self._background_recorder.output_path
            recording_error = self._background_recorder.error
        status = self._playback_status()
        if recording_status not in (RecordingStatus.DISABLED, RecordingStatus.IDLE):
            recording_context = recording_status.value
            if recording_output_path is not None:
                recording_context = f"{recording_context}: {recording_output_path}"
            if recording_error is not None:
                recording_context = f"{recording_context} ({recording_error})"
            status = f"{status} | recording {recording_context}"
        return ViewerSnapshot(
            status=status,
            selected_motion_index=motion_index,
            selected_motion_name=self._motion_name(motion),
            motion_count=len(self.motions),
            frame_index=frame_index,
            frame_count=frame_count,
            playback_speed=self.controller.playback_speed,
            paused=self.controller.paused,
            recording_status=recording_status,
            stop_requested=self._stop_requested,
            recording_output_path=recording_output_path,
            recording_error=recording_error,
        )

    def _playback_status(self) -> str:
        motion_index = self.controller.selected_motion_index
        motion = self.selected_motion
        frame_count = self._frame_count(motion)
        frame_index = self.controller.current_frame_index(frame_count=frame_count)
        state = "paused" if self.controller.paused else "playing"
        percent = 100.0 * frame_index / frame_count
        return (
            f"Motion {motion_index + 1}/{len(self.motions)} "
            f"| {self._motion_name(motion)} "
            f"| {self._progress_bar(frame_index=frame_index, frame_count=frame_count)} "
            f"{frame_index}/{frame_count} ({percent:.1f}%) "
            f"| speed {self.controller.playback_speed:g}x "
            f"| {state}"
        )

    @staticmethod
    def _frame_count(motion: Any) -> int:
        try:
            frame_count = int(motion.root_pos.shape[0])
        except AttributeError as exc:
            raise TypeError("reference motion must expose root_pos frames") from exc
        if frame_count <= 0:
            raise ValueError("reference motion must contain at least one frame")
        return frame_count

    @staticmethod
    def _motion_name(motion: Any) -> str:
        name = getattr(motion, "name", None)
        if name:
            return str(name)
        return "unnamed"

    @staticmethod
    def _progress_bar(*, frame_index: int, frame_count: int) -> str:
        filled = round(STATUS_PROGRESS_BAR_WIDTH * frame_index / frame_count)
        filled = max(0, min(STATUS_PROGRESS_BAR_WIDTH, filled))
        empty = STATUS_PROGRESS_BAR_WIDTH - filled
        return f"[{'=' * filled}{'-' * empty}]"

    @staticmethod
    def _motion_fps(motion: Any) -> float:
        fps = float(getattr(motion, "fps", 30.0))
        if fps <= 0.0 or not math.isfinite(fps):
            raise ValueError("reference motion fps must be positive and finite")
        return fps
