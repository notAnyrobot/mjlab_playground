"""Native MuJoCo presentation adapter for reference-motion playback."""

from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..motion_viewer import (
    PlaybackAction,
    ViewerError,
    ViewerSnapshot,
    ViewerTick,
)

_KEY_ACTIONS = {
    32: PlaybackAction.TOGGLE_PAUSE,
    262: PlaybackAction.NEXT_CLIP,
    263: PlaybackAction.PREVIOUS_CLIP,
    264: PlaybackAction.SLOWER,
    265: PlaybackAction.FASTER,
    92: PlaybackAction.RECORD_SELECTED_CLIP,
}


@dataclass(frozen=True)
class NativeCameraConfig:
    distance: float = 2.0
    elevation: float = -5.0
    azimuth: float = 20.0


class NativeMujocoViewerAdapter:
    """Own the complete passive-viewer lifecycle around the core update seam."""

    def __init__(
        self,
        *,
        scene: Any,
        launch_passive: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        target_refresh_rate: float = 60.0,
        recording_enabled: bool = False,
        camera_config: NativeCameraConfig | None = None,
        status_reporter: Callable[[str], None] | None = None,
        handle_configurator: Callable[[Any], None] | None = None,
        handle_status_provider: Callable[[Any], str] | None = None,
    ) -> None:
        if target_refresh_rate < 0.0:
            raise ValueError("target_refresh_rate must be non-negative")
        self._scene = scene
        self._launch_passive = launch_passive
        self._clock = clock
        self._sleep = sleep
        self._target_refresh_rate = target_refresh_rate
        self._recording_enabled = recording_enabled
        self.camera_config = camera_config or NativeCameraConfig()
        self._status_reporter = status_reporter
        self._handle_configurator = handle_configurator
        self._handle_status_provider = handle_status_provider
        self._handle: Any | None = None
        self._closed = False
        self._pending_actions: list[PlaybackAction] = []

    def run(self, update: Callable[[ViewerTick], ViewerSnapshot]) -> None:
        try:
            handle = self._launch()
            self._handle = handle
            self._configure_camera(handle)
            previous_time = self._clock()
            stopped = False

            while handle.is_running():
                current_time = self._clock()
                elapsed_seconds = max(0.0, current_time - previous_time)
                previous_time = current_time
                actions = tuple(self._pending_actions)
                self._pending_actions.clear()
                snapshot = update(
                    ViewerTick(
                        elapsed_seconds=elapsed_seconds,
                        actions=actions,
                    )
                )
                stopped = snapshot.stop_requested
                if stopped:
                    break

                self._scene.sync_display_data()
                self._report_status(snapshot, handle)
                handle.sync()

                if self._target_refresh_rate > 0.0:
                    spent = max(0.0, self._clock() - current_time)
                    remaining = 1.0 / self._target_refresh_rate - spent
                    if remaining > 0.0:
                        self._sleep(remaining)

            if not stopped:
                update(ViewerTick(0.0, (PlaybackAction.STOP,)))
        except ViewerError:
            raise
        except Exception as exc:
            raise ViewerError("native viewer session failed") from exc
        finally:
            primary_error = sys.exception()
            try:
                self.close()
            except ViewerError as exc:
                if primary_error is None:
                    raise
                warnings.warn(
                    f"failed to close native viewer: {exc.__cause__ or exc}",
                    stacklevel=2,
                )

    def close(self) -> None:
        if self._closed:
            return
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception as exc:
                raise ViewerError("failed to close native viewer") from exc
        self._closed = True

    def _launch(self) -> Any:
        launch_passive = self._launch_passive
        if launch_passive is None:
            import mujoco.viewer

            launch_passive = mujoco.viewer.launch_passive
        handle = launch_passive(
            self._scene.mj_model,
            self._scene.mj_data,
            key_callback=self._handle_key,
            show_left_ui=False,
            show_right_ui=False,
        )
        if handle is None:
            raise RuntimeError("failed to launch native MuJoCo viewer")
        return handle

    def _handle_key(self, key: int) -> None:
        action = _KEY_ACTIONS.get(key)
        if action is None:
            return
        if (
            action is PlaybackAction.RECORD_SELECTED_CLIP
            and not self._recording_enabled
        ):
            return
        self._pending_actions.append(action)

    def _configure_camera(self, handle: Any) -> None:
        if self._handle_configurator is not None:
            self._handle_configurator(handle)
            return
        configure_tracking_camera = getattr(
            self._scene,
            "configure_tracking_camera",
            None,
        )
        if configure_tracking_camera is not None:
            configure_tracking_camera(handle, self.camera_config)

    def _report_status(self, snapshot: ViewerSnapshot, handle: Any) -> None:
        if self._status_reporter is None:
            return
        if self._handle_status_provider is not None:
            camera_status = self._handle_status_provider(handle)
        else:
            cam = handle.cam
            camera_status = (
                f"camera distance={_format_camera_value(cam.distance)} "
                f"elevation={_format_camera_value(cam.elevation)} "
                f"azimuth={_format_camera_value(cam.azimuth)}"
            )
        status = snapshot.status
        if camera_status:
            status = f"{status} | {camera_status}"
        self._status_reporter(status)


def _format_camera_value(value: Any) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")
