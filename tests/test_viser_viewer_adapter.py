from __future__ import annotations

import warnings
from contextlib import nullcontext

import numpy as np
import pytest
from mjlab_playground.motion_lib.motion_viewer import (
    PlaybackAction,
    RecordingStatus,
    ViewerError,
    ViewerSnapshot,
)
from mjlab_playground.motion_lib.viewers import create_viewer_adapter
from mjlab_playground.motion_lib.viewers.viser import ViserPlaybackControls


class FakeViserServer:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


class FakeViserControls:
    def __init__(self) -> None:
        self.action_handler = None
        self.snapshots: list[ViewerSnapshot] = []
        self.close_calls = 0

    def set_action_handler(self, handler) -> None:
        self.action_handler = handler

    def update(self, snapshot: ViewerSnapshot) -> None:
        self.snapshots.append(snapshot)
        if len(self.snapshots) == 1:
            assert self.action_handler is not None
            for action in (
                PlaybackAction.TOGGLE_PAUSE,
                PlaybackAction.PREVIOUS_MOTION,
                PlaybackAction.NEXT_MOTION,
                PlaybackAction.SLOWER,
                PlaybackAction.FASTER,
                PlaybackAction.RECORD_SELECTED_CLIP,
                PlaybackAction.STOP,
            ):
                self.action_handler(action)

    def close(self) -> None:
        self.close_calls += 1


class FakeViserScene:
    def __init__(self) -> None:
        self.mj_data_updates: list[object] = []

    def update_from_mjdata(self, mj_data: object) -> None:
        self.mj_data_updates.append(mj_data)


class FakeAuthoritativeScene:
    def __init__(self) -> None:
        self.mj_model = object()
        self.mj_data = object()
        self.display_syncs = 0

    def sync_display_data(self) -> None:
        self.display_syncs += 1


def test_production_viser_controls_map_every_common_playback_action() -> None:
    class Handle:
        def __init__(self) -> None:
            self.callback = None
            self.remove_calls = 0

        def on_click(self, callback):
            self.callback = callback
            return callback

        def remove(self) -> None:
            self.remove_calls += 1

    class Gui:
        def __init__(self) -> None:
            self.buttons: list[tuple[str, Handle]] = []

        def add_folder(self, _label: str):
            return nullcontext()

        def add_html(self, content: str):
            return type("Status", (), {"content": content})()

        def add_button(self, label: str) -> Handle:
            handle = Handle()
            self.buttons.append((label, handle))
            return handle

    gui = Gui()
    controls = ViserPlaybackControls(
        type("Server", (), {"gui": gui})(),
        object(),
        recording_enabled=True,
    )
    actions: list[PlaybackAction] = []
    controls.set_action_handler(actions.append)

    for _, button in gui.buttons:
        assert button.callback is not None
        button.callback(None)
    controls.close()

    assert actions == [
        PlaybackAction.TOGGLE_PAUSE,
        PlaybackAction.PREVIOUS_MOTION,
        PlaybackAction.NEXT_MOTION,
        PlaybackAction.SLOWER,
        PlaybackAction.FASTER,
        PlaybackAction.RECORD_SELECTED_CLIP,
        PlaybackAction.STOP,
    ]
    assert all(button.remove_calls == 1 for _, button in gui.buttons)


def test_production_viser_controls_hide_recording_when_disabled() -> None:
    class Gui:
        def __init__(self) -> None:
            self.buttons: list[str] = []

        def add_folder(self, _label: str):
            return nullcontext()

        def add_html(self, content: str):
            return type("Status", (), {"content": content})()

        def add_button(self, label: str):
            self.buttons.append(label)
            return type(
                "Button",
                (),
                {
                    "on_click": lambda self, callback: callback,
                    "remove": lambda self: None,
                },
            )()

    gui = Gui()
    ViserPlaybackControls(
        type("Server", (), {"gui": gui})(),
        object(),
        recording_enabled=False,
    )

    assert "Record selected clip" not in gui.buttons


def _snapshot(*, stop_requested: bool = False) -> ViewerSnapshot:
    return ViewerSnapshot(
        status="Motion 1/1 | walk | playing",
        selected_motion_index=0,
        selected_motion_name="walk",
        motion_count=1,
        frame_index=0,
        frame_count=3,
        playback_speed=1.0,
        paused=False,
        recording_status=RecordingStatus.DISABLED,
        stop_requested=stop_requested,
    )


def test_viser_factory_queues_typed_controls_and_syncs_authoritative_scene() -> None:
    scene = FakeAuthoritativeScene()
    server = FakeViserServer()
    controls = FakeViserControls()
    viser_scene = FakeViserScene()
    ticks = []
    clock = iter([10.0, 10.0, 10.25]).__next__
    adapter = create_viewer_adapter(
        "viser",
        scene=scene,
        server=server,
        controls=controls,
        viser_scene=viser_scene,
        clock=clock,
        sleep=lambda _: None,
        target_refresh_rate=0.0,
    )

    def update(tick):
        ticks.append(tick)
        return _snapshot(stop_requested=PlaybackAction.STOP in tick.actions)

    adapter.run(update)
    adapter.close()

    assert [tick.actions for tick in ticks] == [
        (),
        (
            PlaybackAction.TOGGLE_PAUSE,
            PlaybackAction.PREVIOUS_MOTION,
            PlaybackAction.NEXT_MOTION,
            PlaybackAction.SLOWER,
            PlaybackAction.FASTER,
            PlaybackAction.RECORD_SELECTED_CLIP,
            PlaybackAction.STOP,
        ),
    ]
    assert [tick.elapsed_seconds for tick in ticks] == [0.0, 0.25]
    assert scene.display_syncs == 1
    assert viser_scene.mj_data_updates == [scene.mj_data]
    assert len(controls.snapshots) == 1
    assert controls.close_calls == 1
    assert server.stop_calls == 1


def test_viser_factory_can_construct_production_adapter_lazily() -> None:
    adapter = create_viewer_adapter("viser", scene=FakeAuthoritativeScene())

    adapter.close()


def test_viser_interrupt_submits_stop_and_returns_cleanly() -> None:
    scene = FakeAuthoritativeScene()
    server = FakeViserServer()
    controls = FakeViserControls()
    viser_scene = FakeViserScene()
    adapter = create_viewer_adapter(
        "viser",
        scene=scene,
        server=server,
        controls=controls,
        viser_scene=viser_scene,
        clock=iter([10.0, 10.0, 10.25]).__next__,
        target_refresh_rate=0.0,
    )
    ticks = []

    def update(tick):
        ticks.append(tick)
        if len(ticks) == 1:
            raise KeyboardInterrupt
        return _snapshot(stop_requested=True)

    adapter.run(update)

    assert [tick.actions for tick in ticks] == [(), (PlaybackAction.STOP,)]
    assert controls.close_calls == 1
    assert server.stop_calls == 1


def test_viser_initialization_failure_closes_server_and_preserves_cause() -> None:
    scene = FakeAuthoritativeScene()
    server = FakeViserServer()
    cause = RuntimeError("Viser scene conversion failed")

    def fail_scene_creation(_server, _scene):
        raise cause

    adapter = create_viewer_adapter(
        "viser",
        scene=scene,
        server_factory=lambda: server,
        viser_scene_factory=fail_scene_creation,
    )

    with pytest.raises(ViewerError, match="viser viewer session failed") as exc_info:
        adapter.run(lambda _: _snapshot())
    adapter.close()

    assert exc_info.value.__cause__ is cause
    assert server.stop_calls == 1


def test_viser_update_failure_closes_resources_and_preserves_cause() -> None:
    scene = FakeAuthoritativeScene()
    server = FakeViserServer()
    controls = FakeViserControls()
    viser_scene = FakeViserScene()
    cause = RuntimeError("Viser websocket update failed")

    def fail_update(_mj_data: object) -> None:
        raise cause

    viser_scene.update_from_mjdata = fail_update  # type: ignore[method-assign]
    adapter = create_viewer_adapter(
        "viser",
        scene=scene,
        server=server,
        controls=controls,
        viser_scene=viser_scene,
        clock=lambda: 10.0,
        target_refresh_rate=0.0,
    )

    with pytest.raises(ViewerError, match="viser viewer session failed") as exc_info:
        adapter.run(lambda _: _snapshot())
    adapter.close()

    assert exc_info.value.__cause__ is cause
    assert controls.close_calls == 1
    assert server.stop_calls == 1


@pytest.mark.parametrize("cleanup_failure", ["controls", "server"])
@pytest.mark.parametrize("primary_failure", [False, True])
def test_viser_cleanup_failure_is_typed_retries_and_does_not_replace_primary(
    cleanup_failure: str,
    primary_failure: bool,
) -> None:
    scene = FakeAuthoritativeScene()
    primary_cause = RuntimeError("Viser update failed")
    cleanup_cause = RuntimeError(f"Viser {cleanup_failure} close failed")

    class RetryableControls(FakeViserControls):
        def close(self) -> None:
            self.close_calls += 1
            if cleanup_failure == "controls" and self.close_calls == 1:
                raise cleanup_cause

    class RetryableServer(FakeViserServer):
        def stop(self) -> None:
            self.stop_calls += 1
            if cleanup_failure == "server" and self.stop_calls == 1:
                raise cleanup_cause

    controls = RetryableControls()
    server = RetryableServer()
    adapter = create_viewer_adapter(
        "viser",
        scene=scene,
        server=server,
        controls=controls,
        viser_scene=FakeViserScene(),
        clock=lambda: 10.0,
        target_refresh_rate=0.0,
    )

    def update(_tick):
        if primary_failure:
            raise primary_cause
        return _snapshot(stop_requested=True)

    expected_message = (
        "viser viewer session failed"
        if primary_failure
        else "failed to close Viser viewer"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ViewerError, match=expected_message) as error:
            adapter.run(update)

    if primary_failure:
        assert error.value.__cause__ is primary_cause
        assert any(
            "failed to close Viser viewer" in str(warning.message) for warning in caught
        )
    else:
        assert error.value.__cause__ is cleanup_cause
        assert caught == []

    adapter.close()
    expected_controls_calls = 2 if cleanup_failure == "controls" else 1
    expected_server_calls = 2 if cleanup_failure == "server" else 1
    assert controls.close_calls == expected_controls_calls
    assert server.stop_calls == expected_server_calls


def test_viser_targets_30_hz_without_replacing_wall_clock_elapsed_time() -> None:
    scene = FakeAuthoritativeScene()
    controls = FakeViserControls()
    sleeps: list[float] = []
    ticks = []
    adapter = create_viewer_adapter(
        "viser",
        scene=scene,
        server=FakeViserServer(),
        controls=controls,
        viser_scene=FakeViserScene(),
        clock=iter([10.0, 10.0, 10.0, 10.0 + 1.0 / 30.0]).__next__,
        sleep=sleeps.append,
    )

    def update(tick):
        ticks.append(tick)
        return _snapshot(stop_requested=PlaybackAction.STOP in tick.actions)

    adapter.run(update)

    assert sleeps == pytest.approx([1.0 / 30.0])
    assert [tick.elapsed_seconds for tick in ticks] == pytest.approx([0.0, 1.0 / 30.0])


def test_real_mjlab_viser_scene_applies_one_authoritative_scene_update() -> None:
    mujoco = pytest.importorskip("mujoco", reason="MuJoCo integration unavailable")
    try:
        from mjlab.viewer.viser import MjlabViserScene
    except ImportError as exc:
        pytest.skip(f"public MjlabViserScene unavailable: {exc}")

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="robot" pos="0 0 1">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    class Handle:
        def __init__(self, **kwargs) -> None:
            self.visible = kwargs.get("visible", True)
            self.position = kwargs.get("position", np.zeros(3))
            self.wxyz = kwargs.get("wxyz", np.array([1.0, 0.0, 0.0, 0.0]))
            self.batched_positions = kwargs.get("batched_positions", np.zeros((0, 3)))
            self.batched_wxyzs = kwargs.get("batched_wxyzs", np.zeros((0, 4)))

        def remove(self) -> None:
            pass

    class SceneApi:
        def __init__(self) -> None:
            self.handles: list[Handle] = []

        def _handle(self, **kwargs) -> Handle:
            handle = Handle(**kwargs)
            self.handles.append(handle)
            return handle

        def configure_environment_map(self, **_kwargs) -> None:
            pass

        def add_frame(self, *_args, **kwargs) -> Handle:
            return self._handle(**kwargs)

        def add_grid(self, *_args, **kwargs) -> Handle:
            return self._handle(**kwargs)

        def add_mesh_trimesh(self, *_args, **kwargs) -> Handle:
            return self._handle(**kwargs)

        def add_batched_meshes_trimesh(self, *_args, **kwargs) -> Handle:
            return self._handle(**kwargs)

        def add_batched_meshes_simple(self, *_args, **kwargs) -> Handle:
            return self._handle(**kwargs)

    class ServerApi:
        def __init__(self) -> None:
            self.scene = SceneApi()
            self.flush_calls = 0

        def atomic(self):
            return nullcontext()

        def flush(self) -> None:
            self.flush_calls += 1

    server = ServerApi()
    scene = MjlabViserScene(server=server, mj_model=model, num_envs=1)  # type: ignore[arg-type]

    scene.update_from_mjdata(data)

    assert any(
        np.allclose(handle.batched_positions, [[0.0, 0.0, 1.0]])
        for handle in server.scene.handles
    )
