from __future__ import annotations

import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, TextIO

PLAYBACK_SPEEDS = (0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DEFAULT_PLAYBACK_SPEED_INDEX = PLAYBACK_SPEEDS.index(1.0)
STATUS_PROGRESS_BAR_WIDTH = 20
PlaybackAction = str
KEY_TO_PLAYBACK_ACTION: dict[int, PlaybackAction] = {
    32: "space",
    92: "toggle_recording",
    262: "right",
    263: "left",
    264: "down",
    265: "up",
}


class SceneAdapter(Protocol):
    """Minimal scene seam used by the source-agnostic reference motion viewer."""

    @property
    def mj_model(self) -> Any:
        """MuJoCo model rendered by the passive viewer."""

    @property
    def mj_data(self) -> Any:
        """MuJoCo data rendered by the passive viewer."""

    def apply(self, motion: Any, frame_index: int) -> None:
        """Apply one reference motion frame to the scene."""

    def sync_display_data(self) -> None:
        """Copy the latest applied scene state into ``mj_data`` for display."""


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
        self.selected_motion_index = (self.selected_motion_index + 1) % self.motion_count
        self.reset_frame()

    def select_previous_motion(self) -> None:
        self.selected_motion_index = (self.selected_motion_index - 1) % self.motion_count
        self.reset_frame()

    def reset_frame(self) -> None:
        self.frame_position = 0.0

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def handle_action(self, action: PlaybackAction) -> None:
        if action == "left":
            self.select_previous_motion()
        elif action == "right":
            self.select_next_motion()
        elif action == "up":
            self.increase_speed()
        elif action == "down":
            self.decrease_speed()
        elif action == "space":
            self.toggle_pause()

    def advance(self, *, frame_count: int, frames: float = 1.0) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.paused:
            return
        if not math.isfinite(frames):
            raise ValueError("frames must be finite")
        self.frame_position = (self.frame_position + frames * self.playback_speed) % frame_count

    def current_frame_index(self, *, frame_count: int) -> int:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        return int(self.frame_position) % frame_count


class TerminalStatusReporter:
    """Render playback status as a single updating terminal line."""

    def __init__(self, stream: TextIO | None = None, max_width: int | None = None) -> None:
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
        scene_adapter: SceneAdapter,
        *,
        frame_renderer_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not motions:
            raise ValueError("MotionViewer requires at least one reference motion")
        self.motions = list(motions)
        self.scene_adapter = scene_adapter
        self.controller = PlaybackController(motion_count=len(self.motions))
        self._default_frame_renderer_factory = frame_renderer_factory

    @property
    def selected_motion(self) -> Any:
        return self.motions[self.controller.selected_motion_index]

    def handle_action(self, action: PlaybackAction) -> None:
        self.controller.handle_action(action)

    def handle_key(self, key: int) -> None:
        action = KEY_TO_PLAYBACK_ACTION.get(key)
        if action is not None:
            self.handle_action(action)

    def playback_status(self) -> str:
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

    def render_current_frame(self) -> None:
        motion = self.selected_motion
        frame_index = self.controller.current_frame_index(
            frame_count=self._frame_count(motion),
        )
        self.scene_adapter.apply(motion, frame_index)

    def advance(self, *, frames: float = 1.0) -> None:
        self.controller.advance(
            frame_count=self._frame_count(self.selected_motion),
            frames=frames,
        )

    def record_headless(
        self,
        *,
        recording_output_paths: Sequence[Path],
        recording_attachment_factory: Callable[[], Any] | None = None,
        frame_renderer_factory: Callable[[], Any] | None = None,
        status_reporter: Callable[[str], None] | None = None,
    ) -> list[Any]:
        """Record every loaded reference motion once without a passive viewer."""
        if len(recording_output_paths) != len(self.motions):
            raise ValueError("recording_output_paths must match the loaded motion count")
        if recording_attachment_factory is None:
            raise ValueError("recording_attachment_factory is required for recording")
        frame_renderer_factory = self._resolve_frame_renderer_factory(
            frame_renderer_factory,
        )

        def report(message: str) -> None:
            if status_reporter is not None:
                status_reporter(message)
            else:
                print(message, file=sys.stderr, flush=True)

        renderer = frame_renderer_factory()
        results: list[Any] = []
        try:
            for motion_index, (motion, output_path) in enumerate(
                zip(self.motions, recording_output_paths, strict=True)
            ):
                motion_name = self._motion_name(motion)
                frame_count = self._frame_count(motion)
                fps = self._motion_fps(motion)
                attachment = recording_attachment_factory()
                report(
                    f"Recording {motion_index + 1}/{len(self.motions)}: "
                    f"{motion_name} -> {output_path}"
                )
                attachment.start(output_path, fps=fps)
                for frame_index in range(frame_count):
                    self.scene_adapter.apply(motion, frame_index)
                    attachment.capture(renderer.render_frame())
                    report(f"Recording {motion_name}: frame {frame_index + 1}/{frame_count}")
                result = attachment.stop()
                results.append(result)
                if getattr(result, "written", False):
                    report(f"Saved recording: {result.output_path}")
                else:
                    reason = getattr(result, "reason", "no frames captured")
                    report(f"Skipped recording: {reason}")
        finally:
            close_frame_renderer = getattr(renderer, "close", None)
            if close_frame_renderer is not None:
                close_frame_renderer()

        return results

    def record_single_headless(
        self,
        *,
        motion_index: int,
        output_path: Path,
        recording_attachment: Any | None = None,
        frame_renderer_factory: Callable[[], Any] | None = None,
        status_reporter: Callable[[str], None] | None = None,
    ) -> Any:
        """Record one reference motion after interactive viewing has stopped."""
        if motion_index < 0 or motion_index >= len(self.motions):
            raise IndexError(f"motion_index out of range: {motion_index}")
        if recording_attachment is None:
            raise ValueError("recording_attachment is required for recording")
        frame_renderer_factory = self._resolve_frame_renderer_factory(
            frame_renderer_factory,
        )

        def report(message: str) -> None:
            if status_reporter is not None:
                status_reporter(message)
            else:
                print(message, file=sys.stderr, flush=True)

        motion = self.motions[motion_index]
        motion_name = self._motion_name(motion)
        frame_count = self._frame_count(motion)
        renderer = frame_renderer_factory()

        def attachment_is_recording() -> bool:
            return bool(getattr(recording_attachment, "is_recording", True))

        try:
            recording_attachment.start(output_path, fps=self._motion_fps(motion))
            for frame_index in range(frame_count):
                self.scene_adapter.apply(motion, frame_index)
                frame = renderer.render_frame()
                if not attachment_is_recording():
                    report("Skipped recording: recorder stopped before capture")
                    return None
                recording_attachment.capture(frame)
                report(f"Recording {motion_name}: frame {frame_index + 1}/{frame_count}")
            if not attachment_is_recording():
                report("Skipped recording: recorder stopped before save")
                return None
            result = recording_attachment.stop()
            if getattr(result, "written", False):
                report(f"Saved recording: {result.output_path}")
            else:
                reason = getattr(result, "reason", "no frames captured")
                report(f"Skipped recording: {reason}")
            return result
        finally:
            close_frame_renderer = getattr(renderer, "close", None)
            if close_frame_renderer is not None:
                close_frame_renderer()

    def run_interactive(
        self,
        *,
        launch_passive: Callable[..., Any] | None = None,
        status_reporter: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        target_refresh_rate: float = 60.0,
        recording_attachment: Any | None = None,
        recording_output_paths: Sequence[Path] | None = None,
        frame_renderer_factory: Callable[[], Any] | None = None,
        recording_executor: Callable[[int, Path], Any] | None = None,
        viewer_handle_configurator: Callable[[Any], None] | None = None,
        viewer_handle_status_provider: Callable[[Any], str] | None = None,
    ) -> None:
        if launch_passive is None:
            import mujoco.viewer

            launch_passive = mujoco.viewer.launch_passive
        recording_enabled = (
            recording_attachment is not None
            or recording_output_paths is not None
            or recording_executor is not None
        )
        if recording_enabled:
            if recording_output_paths is None:
                raise ValueError("recording_output_paths are required when recording is enabled")
            if len(recording_output_paths) != len(self.motions):
                raise ValueError("recording_output_paths must match the loaded motion count")
            if recording_executor is None:
                if recording_attachment is None:
                    raise ValueError(
                        "recording_executor or recording_attachment is required "
                        "for recording"
                    )
                frame_renderer_factory = self._resolve_frame_renderer_factory(
                    frame_renderer_factory,
                )

        recording_motion_index: int | None = None
        pending_recording: tuple[int, Path] | None = None

        def report_recording_status(message: str) -> None:
            if status_reporter is not None:
                status_reporter(message)
            else:
                print(message, file=sys.stderr, flush=True)

        def execute_recording(motion_index: int, output_path: Path) -> Any:
            if recording_executor is not None:
                return recording_executor(motion_index, output_path)
            assert recording_attachment is not None
            assert frame_renderer_factory is not None
            return self.record_single_headless(
                motion_index=motion_index,
                output_path=output_path,
                recording_attachment=recording_attachment,
                frame_renderer_factory=frame_renderer_factory,
                status_reporter=status_reporter,
            )

        def stop_recording() -> None:
            nonlocal pending_recording, recording_motion_index
            if not recording_enabled or recording_motion_index is None:
                return
            assert recording_output_paths is not None
            output_path = recording_output_paths[recording_motion_index]
            pending_recording = (recording_motion_index, output_path)
            recording_motion_index = None
            report_recording_status(
                f"Recording queued: {output_path}. "
                "Pausing viewer render while video is recorded."
            )

        def toggle_recording() -> None:
            nonlocal recording_motion_index
            if not recording_enabled or recording_output_paths is None:
                return
            if recording_motion_index is not None:
                stop_recording()
                return

            recording_motion_index = self.controller.selected_motion_index
            output_path = recording_output_paths[recording_motion_index]
            report_recording_status(f"Recording: {output_path}")

        def key_callback(key: int) -> None:
            action = KEY_TO_PLAYBACK_ACTION.get(key)
            if action == "toggle_recording":
                toggle_recording()
                return
            if recording_motion_index is not None and action in {"left", "right"}:
                report_recording_status(
                    "Recording active; stop recording before switching motions."
                )
                return
            self.handle_key(key)

        passive_key_callback = key_callback if recording_enabled else self.handle_key

        handle = launch_passive(
            self.scene_adapter.mj_model,
            self.scene_adapter.mj_data,
            key_callback=passive_key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        if handle is None:
            raise RuntimeError("Failed to launch MuJoCo viewer")
        if viewer_handle_configurator is not None:
            viewer_handle_configurator(handle)

        previous_time = clock()
        min_frame_time = 1.0 / target_refresh_rate if target_refresh_rate > 0.0 else 0.0
        try:
            try:
                while handle.is_running():
                    self.render_current_frame()
                    self.scene_adapter.sync_display_data()
                    if status_reporter is not None:
                        status = self.playback_status()
                        if viewer_handle_status_provider is not None:
                            status_suffix = viewer_handle_status_provider(handle)
                            if status_suffix:
                                status = f"{status} | {status_suffix}"
                        status_reporter(status)
                    handle.sync()
                    if pending_recording is not None:
                        motion_index, output_path = pending_recording
                        pending_recording = None
                        execute_recording(motion_index, output_path)
                        previous_time = clock()
                        continue

                    current_time = clock()
                    elapsed_seconds = max(0.0, current_time - previous_time)
                    previous_time = current_time
                    self.advance(frames=elapsed_seconds * self._motion_fps(self.selected_motion))

                    sleep_for = min_frame_time - elapsed_seconds
                    if sleep_for > 0.0:
                        sleep(sleep_for)
            finally:
                if recording_motion_index is not None and pending_recording is None:
                    assert recording_output_paths is not None
                    pending_recording = (
                        recording_motion_index,
                        recording_output_paths[recording_motion_index],
                    )
                    recording_motion_index = None
                try:
                    if pending_recording is not None:
                        motion_index, output_path = pending_recording
                        pending_recording = None
                        execute_recording(motion_index, output_path)
                finally:
                    handle.close()
        finally:
            close_status_reporter = getattr(status_reporter, "close", None)
            if close_status_reporter is not None:
                close_status_reporter()

    def _resolve_frame_renderer_factory(
        self,
        frame_renderer_factory: Callable[[], Any] | None,
    ) -> Callable[[], Any]:
        if frame_renderer_factory is not None:
            return frame_renderer_factory
        if self._default_frame_renderer_factory is not None:
            return self._default_frame_renderer_factory
        raise ValueError("frame_renderer_factory is required for recording")

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
        display_name = getattr(motion, "display_name", None)
        if display_name:
            return str(display_name)
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
