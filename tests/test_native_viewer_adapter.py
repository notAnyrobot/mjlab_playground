from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
from mjlab_playground.motion_lib.motion_viewer import (
    PlaybackAction,
    RecordingStatus,
    ViewerError,
    ViewerSnapshot,
)
from mjlab_playground.motion_lib.viewers import create_viewer_adapter


class FakeNativeScene:
    def __init__(self) -> None:
        self.mj_model = object()
        self.mj_data = object()
        self.display_syncs = 0
        self.camera_configurations: list[tuple[object, object]] = []

    def sync_display_data(self) -> None:
        self.display_syncs += 1

    def configure_tracking_camera(self, handle: object, camera_config: object) -> None:
        self.camera_configurations.append((handle, camera_config))


class FakePassiveHandle:
    def __init__(self, *, key_events: dict[int, int] | None = None) -> None:
        self.key_events = key_events or {}
        self.key_callback = None
        self.sync_calls = 0
        self.close_calls = 0
        self.running = True
        self.cam = SimpleNamespace(distance=2.0, elevation=-5.0, azimuth=20.0)

    def is_running(self) -> bool:
        return self.running

    def sync(self) -> None:
        self.sync_calls += 1
        key = self.key_events.get(self.sync_calls)
        if key is not None:
            assert self.key_callback is not None
            self.key_callback(key)
        if self.sync_calls == 3:
            self.running = False

    def close(self) -> None:
        self.close_calls += 1


def _snapshot(*, stop_requested: bool = False) -> ViewerSnapshot:
    return ViewerSnapshot(
        status="Clip 1/1 | walk | playing",
        selected_clip_index=0,
        selected_clip_name="walk",
        clip_count=1,
        frame_index=0,
        frame_count=3,
        playback_speed=1.0,
        paused=False,
        recording_status=RecordingStatus.DISABLED,
        stop_requested=stop_requested,
    )


def test_native_factory_drives_typed_actions_and_authoritative_scene_sync() -> None:
    scene = FakeNativeScene()
    handle = FakePassiveHandle(
        key_events={
            1: 32,
            2: 262,
        }
    )
    launch_calls: list[tuple[object, object, dict[str, object]]] = []
    ticks = []

    def launch_passive(model, data, **kwargs):
        launch_calls.append((model, data, kwargs))
        handle.key_callback = kwargs["key_callback"]
        return handle

    adapter = create_viewer_adapter(
        "native",
        scene=scene,
        launch_passive=launch_passive,
        clock=iter([10.0, 10.0, 10.25, 10.5]).__next__,
        sleep=lambda _: None,
        target_refresh_rate=0.0,
    )

    def update(tick):
        ticks.append(tick)
        return _snapshot(stop_requested=PlaybackAction.STOP in tick.actions)

    adapter.run(update)
    adapter.close()

    assert launch_calls == [
        (
            scene.mj_model,
            scene.mj_data,
            {
                "key_callback": handle.key_callback,
                "show_left_ui": False,
                "show_right_ui": False,
            },
        )
    ]
    assert [tick.actions for tick in ticks] == [
        (),
        (PlaybackAction.TOGGLE_PAUSE,),
        (PlaybackAction.NEXT_CLIP,),
        (PlaybackAction.STOP,),
    ]
    assert scene.display_syncs == 3
    assert scene.camera_configurations == [(handle, adapter.camera_config)]
    assert handle.sync_calls == 3
    assert handle.close_calls == 1


def test_native_adapter_ignores_recording_key_when_recording_is_disabled() -> None:
    scene = FakeNativeScene()
    handle = FakePassiveHandle(key_events={1: 92})
    actions: list[tuple[PlaybackAction, ...]] = []

    def launch_passive(*_, **kwargs):
        handle.key_callback = kwargs["key_callback"]
        return handle

    adapter = create_viewer_adapter(
        "native",
        scene=scene,
        recording_enabled=False,
        launch_passive=launch_passive,
        clock=lambda: 10.0,
        sleep=lambda _: None,
        target_refresh_rate=0.0,
    )

    def update(tick):
        actions.append(tick.actions)
        return _snapshot(stop_requested=PlaybackAction.STOP in tick.actions)

    adapter.run(update)

    assert PlaybackAction.RECORD_SELECTED_CLIP not in {
        action for tick_actions in actions for action in tick_actions
    }


def test_native_adapter_submits_one_shot_recording_actions_to_the_core() -> None:
    scene = FakeNativeScene()
    handle = FakePassiveHandle(key_events={1: 92, 2: 92})
    actions: list[tuple[PlaybackAction, ...]] = []

    def launch_passive(*_, **kwargs):
        handle.key_callback = kwargs["key_callback"]
        return handle

    adapter = create_viewer_adapter(
        "native",
        scene=scene,
        recording_enabled=True,
        launch_passive=launch_passive,
        clock=lambda: 10.0,
        sleep=lambda _: None,
        target_refresh_rate=0.0,
    )

    def update(tick):
        actions.append(tick.actions)
        return _snapshot(stop_requested=PlaybackAction.STOP in tick.actions)

    adapter.run(update)

    assert actions == [
        (),
        (PlaybackAction.RECORD_SELECTED_CLIP,),
        (PlaybackAction.RECORD_SELECTED_CLIP,),
        (PlaybackAction.STOP,),
    ]
    assert handle.close_calls == 1


def test_native_initialization_failure_preserves_cause_without_owned_handle() -> None:
    scene = FakeNativeScene()
    cause = RuntimeError("display unavailable")

    def launch_passive(*_, **__):
        raise cause

    adapter = create_viewer_adapter(
        "native",
        scene=scene,
        launch_passive=launch_passive,
    )

    with pytest.raises(ViewerError, match="native viewer session failed") as exc_info:
        adapter.run(lambda _: _snapshot())
    adapter.close()

    assert exc_info.value.__cause__ is cause


def test_native_loop_failure_closes_handle_exactly_once_and_preserves_cause() -> None:
    scene = FakeNativeScene()
    handle = FakePassiveHandle()
    cause = RuntimeError("native sync failed")

    def fail_sync() -> None:
        raise cause

    handle.sync = fail_sync  # type: ignore[method-assign]

    adapter = create_viewer_adapter(
        "native",
        scene=scene,
        launch_passive=lambda *_, **__: handle,
        clock=lambda: 10.0,
        target_refresh_rate=0.0,
    )

    with pytest.raises(ViewerError, match="native viewer session failed") as exc_info:
        adapter.run(lambda _: _snapshot())
    adapter.close()

    assert exc_info.value.__cause__ is cause
    assert handle.close_calls == 1


@pytest.mark.parametrize("primary_failure", [False, True])
def test_native_cleanup_failure_is_typed_and_does_not_replace_primary_error(
    primary_failure: bool,
) -> None:
    scene = FakeNativeScene()
    primary_cause = RuntimeError("native update failed")
    close_cause = RuntimeError("native handle close failed")

    class RetryableCloseHandle(FakePassiveHandle):
        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise close_cause

    handle = RetryableCloseHandle()
    adapter = create_viewer_adapter(
        "native",
        scene=scene,
        launch_passive=lambda *_, **__: handle,
        clock=lambda: 10.0,
        target_refresh_rate=0.0,
    )

    def update(_tick):
        if primary_failure:
            raise primary_cause
        return _snapshot(stop_requested=True)

    expected_message = (
        "native viewer session failed"
        if primary_failure
        else "failed to close native viewer"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ViewerError, match=expected_message) as error:
            adapter.run(update)

    if primary_failure:
        assert error.value.__cause__ is primary_cause
        assert any(
            "failed to close native viewer" in str(warning.message)
            for warning in caught
        )
    else:
        assert error.value.__cause__ is close_cause
        assert caught == []

    adapter.close()
    assert handle.close_calls == 2
