"""Browser-based Viser presentation adapter for reference-motion playback."""

from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable
from importlib import import_module
from threading import Lock
from typing import Any

from ..motion_viewer import PlaybackAction, ViewerError, ViewerSnapshot, ViewerTick


class ViserPlaybackControls:
    """Reference-motion controls built from Viser's public GUI API."""

    def __init__(
        self,
        server: Any,
        viser_scene: Any,
        *,
        recording_enabled: bool = False,
    ) -> None:
        self._action_handler: Callable[[PlaybackAction], None] | None = None
        self._handles: list[Any] = []
        gui = server.gui
        with gui.add_folder("Reference motion"):
            self._status = gui.add_html("Waiting for playback")
            self._add_button(gui, "Pause / Play", PlaybackAction.TOGGLE_PAUSE)
            self._add_button(gui, "Previous clip", PlaybackAction.PREVIOUS_CLIP)
            self._add_button(gui, "Next clip", PlaybackAction.NEXT_CLIP)
            self._add_button(gui, "Slower", PlaybackAction.SLOWER)
            self._add_button(gui, "Faster", PlaybackAction.FASTER)
            if recording_enabled:
                self._add_button(
                    gui,
                    "Record selected clip",
                    PlaybackAction.RECORD_SELECTED_CLIP,
                )
            self._add_button(gui, "Stop", PlaybackAction.STOP)
        create_scene_gui = getattr(viser_scene, "create_scene_gui", None)
        if create_scene_gui is not None:
            with gui.add_folder("Scene"):
                create_scene_gui(
                    camera_distance=2.0,
                    camera_azimuth=20.0,
                    camera_elevation=-5.0,
                )

    def set_action_handler(
        self,
        handler: Callable[[PlaybackAction], None],
    ) -> None:
        self._action_handler = handler

    def update(self, snapshot: ViewerSnapshot) -> None:
        self._status.content = snapshot.status

    def close(self) -> None:
        for handle in reversed(self._handles):
            remove = getattr(handle, "remove", None)
            if remove is not None:
                remove()
        self._handles.clear()

    def _add_button(self, gui: Any, label: str, action: PlaybackAction) -> None:
        button = gui.add_button(label)

        @button.on_click
        def _(_: Any) -> None:
            if self._action_handler is not None:
                self._action_handler(action)

        self._handles.append(button)


class ViserViewerAdapter:
    """Own a Viser session around the presentation-agnostic update seam."""

    def __init__(
        self,
        *,
        scene: Any,
        server: Any | None = None,
        controls: Any | None = None,
        viser_scene: Any | None = None,
        server_factory: Callable[[], Any] | None = None,
        controls_factory: Callable[[Any, Any], Any] | None = None,
        viser_scene_factory: Callable[[Any, Any], Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        target_refresh_rate: float = 30.0,
        recording_enabled: bool = False,
        status_reporter: Callable[[str], None] | None = None,
    ) -> None:
        if target_refresh_rate < 0.0:
            raise ValueError("target_refresh_rate must be non-negative")
        self._scene = scene
        self._server = server
        self._controls = controls
        self._viser_scene = viser_scene
        self._server_factory = server_factory or self._create_server
        self._controls_factory = controls_factory or (
            lambda server, viser_scene: ViserPlaybackControls(
                server,
                viser_scene,
                recording_enabled=recording_enabled,
            )
        )
        self._viser_scene_factory = viser_scene_factory or self._create_viser_scene
        self._clock = clock
        self._sleep = sleep
        self._target_refresh_rate = target_refresh_rate
        # The core owns terminal shutdown reporting; accepting the shared
        # reporter keeps adapter construction presentation-independent.
        self._status_reporter = status_reporter
        self._action_lock = Lock()
        self._pending_actions: list[PlaybackAction] = []
        self._controls_closed = False
        self._server_closed = False

    def run(self, update: Callable[[ViewerTick], ViewerSnapshot]) -> None:
        try:
            self._initialize()
            controls = self._controls
            viser_scene = self._viser_scene
            assert controls is not None
            assert viser_scene is not None
            previous_time = self._clock()
            while True:
                current_time = self._clock()
                elapsed_seconds = max(0.0, current_time - previous_time)
                previous_time = current_time
                snapshot = update(ViewerTick(elapsed_seconds, self._drain_actions()))
                if snapshot.stop_requested:
                    break

                self._scene.sync_display_data()
                viser_scene.update_from_mjdata(self._scene.mj_data)
                controls.update(snapshot)

                if self._target_refresh_rate > 0.0:
                    spent = max(0.0, self._clock() - current_time)
                    remaining = 1.0 / self._target_refresh_rate - spent
                    if remaining > 0.0:
                        self._sleep(remaining)
        except KeyboardInterrupt:
            update(ViewerTick(0.0, (PlaybackAction.STOP,)))
        except ViewerError:
            raise
        except Exception as exc:
            raise ViewerError("viser viewer session failed") from exc
        finally:
            primary_error = sys.exception()
            try:
                self.close()
            except ViewerError as exc:
                if primary_error is None:
                    raise
                warnings.warn(
                    f"failed to close Viser viewer: {exc.__cause__ or exc}",
                    stacklevel=2,
                )

    def close(self) -> None:
        errors: list[Exception] = []
        if self._controls is not None and not self._controls_closed:
            try:
                self._controls.close()
            except Exception as exc:
                errors.append(exc)
            else:
                self._controls_closed = True
        if self._server is not None and not self._server_closed:
            try:
                stop = getattr(self._server, "stop", None)
                if stop is None:
                    stop = getattr(self._server, "close", None)
                if stop is not None:
                    stop()
            except Exception as exc:
                errors.append(exc)
            else:
                self._server_closed = True

        if errors:
            error = ViewerError("failed to close Viser viewer")
            for additional_error in errors[1:]:
                error.add_note(f"additional cleanup failure: {additional_error}")
            raise error from errors[0]

    def _initialize(self) -> None:
        if self._server is None:
            self._server = self._server_factory()
        if self._viser_scene is None:
            self._viser_scene = self._viser_scene_factory(self._server, self._scene)
        if self._controls is None:
            self._controls = self._controls_factory(self._server, self._viser_scene)
        self._controls.set_action_handler(self._enqueue_action)

    @staticmethod
    def _create_server() -> Any:
        viser = import_module("viser")
        return viser.ViserServer(label="mjlab reference motion")

    @staticmethod
    def _create_viser_scene(server: Any, scene: Any) -> Any:
        MjlabViserScene = import_module("mjlab.viewer.viser").MjlabViserScene
        return MjlabViserScene(
            server=server,
            mj_model=scene.mj_model,
            num_envs=1,
        )

    def _enqueue_action(self, action: PlaybackAction) -> None:
        if not isinstance(action, PlaybackAction):
            raise TypeError("Viser controls must submit PlaybackAction values")
        with self._action_lock:
            self._pending_actions.append(action)

    def _drain_actions(self) -> tuple[PlaybackAction, ...]:
        with self._action_lock:
            actions = tuple(self._pending_actions)
            self._pending_actions.clear()
        return actions
