from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mjlab_playground.motion_lib.motion_viewer import (
    PLAYBACK_SPEEDS,
    MotionViewer,
    PlaybackController,
    TerminalStatusReporter,
)
from mjlab_playground.motion_lib.recording import RecordingAttachment


class FakeReferenceMotion:
    def __init__(self, frame_count: int, marker: float) -> None:
        self.root_pos = torch.full((frame_count, 3), marker)
        self.root_rot = torch.zeros(frame_count, 4)
        self.dof_pos = torch.zeros(frame_count, 29)


class FakeSceneAdapter:
    def __init__(self) -> None:
        self.applied: list[tuple[FakeReferenceMotion, int]] = []
        self.mj_model = object()
        self.mj_data = object()
        self.display_syncs = 0

    def apply(self, motion: FakeReferenceMotion, frame_index: int) -> None:
        self.applied.append((motion, frame_index))

    def sync_display_data(self) -> None:
        self.display_syncs += 1


class FakeViewerHandle:
    def __init__(self, *, running_checks: int) -> None:
        self.running_checks = running_checks
        self.sync_calls = 0
        self.closed = False

    def is_running(self) -> bool:
        self.running_checks -= 1
        return self.running_checks >= 0

    def sync(self) -> None:
        self.sync_calls += 1

    def close(self) -> None:
        self.closed = True


class KeyEventViewerHandle:
    def __init__(self, *, running_checks: int, key_events: dict[int, int]) -> None:
        self.running_checks = running_checks
        self.key_events = key_events
        self.sync_calls = 0
        self.closed = False
        self.key_callback = None

    def is_running(self) -> bool:
        self.running_checks -= 1
        return self.running_checks >= 0

    def sync(self) -> None:
        self.sync_calls += 1
        key = self.key_events.get(self.sync_calls)
        if key is not None:
            assert self.key_callback is not None
            self.key_callback(key)

    def close(self) -> None:
        self.closed = True


class FakeRecordingAttachment:
    def __init__(self) -> None:
        self.started: list[tuple[Path, float]] = []
        self.captured: list[object] = []
        self.stopped = 0

    def start(self, output_path: Path, *, fps: float) -> None:
        self.started.append((output_path, fps))

    def capture(self, frame: object) -> None:
        self.captured.append(frame)

    def stop(self):
        self.stopped += 1
        return SimpleNamespace(
            written=bool(self.captured),
            output_path=Path("unused.mp4"),
        )


class FakeHeadlessRecordingAttachment(FakeRecordingAttachment):
    def __init__(self) -> None:
        super().__init__()
        self.output_path: Path | None = None
        self.fps: float | None = None

    def start(self, output_path: Path, *, fps: float) -> None:
        super().start(output_path, fps=fps)
        self.output_path = output_path
        self.fps = fps

    def stop(self):
        self.stopped += 1
        return SimpleNamespace(
            written=bool(self.captured),
            output_path=self.output_path,
            frame_count=len(self.captured),
            fps=self.fps,
        )


class FakeFrameRenderer:
    def __init__(self) -> None:
        self.rendered = 0
        self.closed = False

    def render_frame(self) -> str:
        self.rendered += 1
        return f"frame-{self.rendered}"

    def close(self) -> None:
        self.closed = True


def test_playback_controller_uses_fixed_speed_ladder_and_clamps() -> None:
    controller = PlaybackController(motion_count=2)

    assert PLAYBACK_SPEEDS == (0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    assert controller.playback_speed == 1.0

    for _ in range(10):
        controller.decrease_speed()

    assert controller.playback_speed == 0.1
    assert controller.playback_speed_index == 0

    for _ in range(10):
        controller.increase_speed()

    assert controller.playback_speed == 2.0
    assert controller.playback_speed_index == len(PLAYBACK_SPEEDS) - 1


def test_playback_controller_wraps_motion_selection_and_resets_frame() -> None:
    controller = PlaybackController(motion_count=3)
    controller.advance(frame_count=5, frames=2.0)

    controller.select_previous_motion()

    assert controller.selected_motion_index == 2
    assert controller.frame_position == 0.0

    controller.advance(frame_count=5, frames=3.0)
    controller.select_next_motion()

    assert controller.selected_motion_index == 0
    assert controller.frame_position == 0.0


def test_playback_controller_space_action_toggles_pause() -> None:
    controller = PlaybackController(motion_count=1)

    controller.handle_action("space")

    assert controller.paused is True

    controller.handle_action("space")

    assert controller.paused is False


def test_motion_viewer_applies_reference_motion_without_source_metadata() -> None:
    motions = [
        FakeReferenceMotion(frame_count=2, marker=1.0),
        FakeReferenceMotion(frame_count=3, marker=2.0),
    ]
    scene_adapter = FakeSceneAdapter()
    viewer = MotionViewer(motions, scene_adapter)

    viewer.render_current_frame()
    viewer.handle_action("right")
    viewer.render_current_frame()
    viewer.controller.advance(frame_count=3, frames=1.0)
    viewer.render_current_frame()

    assert scene_adapter.applied == [
        (motions[0], 0),
        (motions[1], 0),
        (motions[1], 1),
    ]


def test_motion_viewer_import_surface_is_source_agnostic() -> None:
    from mjlab_playground.motion_lib.motion_viewer import (
        MotionViewer as BaseMotionViewer,
    )

    motions = [
        FakeReferenceMotion(frame_count=2, marker=1.0),
        FakeReferenceMotion(frame_count=3, marker=2.0),
    ]
    scene_adapter = FakeSceneAdapter()
    viewer = BaseMotionViewer(motions, scene_adapter)

    viewer.render_current_frame()
    viewer.handle_action("right")
    viewer.advance(frames=2.0)
    viewer.render_current_frame()

    assert scene_adapter.applied == [
        (motions[0], 0),
        (motions[1], 2),
    ]


def test_motion_viewer_requires_injected_headless_recording_attachment(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.motion_viewer import (
        MotionViewer as BaseMotionViewer,
    )

    viewer = BaseMotionViewer(
        [FakeReferenceMotion(frame_count=1, marker=1.0)],
        FakeSceneAdapter(),
    )

    with pytest.raises(ValueError, match="recording_attachment_factory is required"):
        viewer.record_headless(
            recording_output_paths=[tmp_path / "walk.mp4"],
            frame_renderer_factory=FakeFrameRenderer,
        )


def test_motion_viewer_requires_injected_interactive_recording_handler(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.motion_viewer import (
        MotionViewer as BaseMotionViewer,
    )

    viewer = BaseMotionViewer(
        [FakeReferenceMotion(frame_count=1, marker=1.0)],
        FakeSceneAdapter(),
    )

    with pytest.raises(
        ValueError,
        match="recording_executor or recording_attachment is required",
    ):
        viewer.run_interactive(
            launch_passive=lambda *_, **__: FakeViewerHandle(running_checks=0),
            recording_output_paths=[tmp_path / "walk.mp4"],
            frame_renderer_factory=FakeFrameRenderer,
        )


def test_motion_viewer_interactive_loop_applies_frames_and_syncs_passive_viewer() -> None:
    motion = FakeReferenceMotion(frame_count=3, marker=1.0)
    motion.fps = 2.0
    scene_adapter = FakeSceneAdapter()
    viewer = MotionViewer([motion], scene_adapter)
    handle = FakeViewerHandle(running_checks=3)
    launched_with = {}
    status_updates: list[str] = []
    clock_values = iter([10.0, 10.5, 11.0, 11.5])

    def launch_passive(model, data, *, key_callback, show_left_ui, show_right_ui):
        launched_with.update(
            model=model,
            data=data,
            key_callback=key_callback,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        )
        return handle

    viewer.run_interactive(
        launch_passive=launch_passive,
        clock=lambda: next(clock_values),
        sleep=lambda _: None,
        status_reporter=status_updates.append,
    )

    assert launched_with == {
        "model": scene_adapter.mj_model,
        "data": scene_adapter.mj_data,
        "key_callback": viewer.handle_key,
        "show_left_ui": False,
        "show_right_ui": False,
    }
    assert scene_adapter.applied == [(motion, 0), (motion, 1), (motion, 2)]
    assert scene_adapter.display_syncs == 3
    assert status_updates == [
        "Motion 1/1 | unnamed | [--------------------] 0/3 (0.0%) | speed 1x | playing",
        "Motion 1/1 | unnamed | [=======-------------] 1/3 (33.3%) | speed 1x | playing",
        "Motion 1/1 | unnamed | [=============-------] 2/3 (66.7%) | speed 1x | playing",
    ]
    assert handle.sync_calls == 3
    assert handle.closed is True


def test_motion_viewer_configures_passive_viewer_handle_before_playback_sync() -> None:
    motion = FakeReferenceMotion(frame_count=1, marker=1.0)
    scene_adapter = FakeSceneAdapter()
    viewer = MotionViewer([motion], scene_adapter)
    handle = FakeViewerHandle(running_checks=1)
    events: list[str] = []

    def launch_passive(model, data, *, key_callback, show_left_ui, show_right_ui):
        events.append("launch")
        return handle

    def configure_handle(viewer_handle) -> None:
        assert viewer_handle is handle
        events.append("configure")

    original_sync_display_data = scene_adapter.sync_display_data

    def sync_display_data() -> None:
        events.append("sync-display")
        original_sync_display_data()

    scene_adapter.sync_display_data = sync_display_data

    viewer.run_interactive(
        launch_passive=launch_passive,
        clock=lambda: 10.0,
        sleep=lambda _: None,
        viewer_handle_configurator=configure_handle,
    )

    assert events == ["launch", "configure", "sync-display"]


def test_motion_viewer_appends_viewer_handle_status_suffix() -> None:
    motion = FakeReferenceMotion(frame_count=1, marker=1.0)
    viewer = MotionViewer([motion], FakeSceneAdapter())
    handle = FakeViewerHandle(running_checks=1)
    status_updates: list[str] = []

    viewer.run_interactive(
        launch_passive=lambda *_, **__: handle,
        clock=lambda: 10.0,
        sleep=lambda _: None,
        status_reporter=status_updates.append,
        viewer_handle_status_provider=lambda viewer_handle: (
            "camera distance=2 elevation=-5 azimuth=20"
            if viewer_handle is handle
            else "wrong-handle"
        ),
    )

    assert status_updates == [
        "Motion 1/1 | unnamed | [--------------------] 0/1 (0.0%) "
        "| speed 1x | playing | camera distance=2 elevation=-5 azimuth=20"
    ]


def test_terminal_status_reporter_keeps_status_on_one_terminal_line() -> None:
    stream = io.StringIO()
    reporter = TerminalStatusReporter(stream=stream, max_width=80)
    long_status = (
        "Motion 1/32 | 20120731_StefanosTheodorou_Stefanos_1os_antrikos_"
        "karsilamas_C3D | [===-----------------] 313/2526 (12.4%) "
        "| speed 1x | playing | camera distance=2 elevation=-19.44 azimuth=11.94"
    )

    reporter(long_status)

    rendered = stream.getvalue()
    assert rendered.startswith("\rMotion 1/32 | ")
    assert rendered.endswith("distance=2 elevation=-19.44 azimuth=11.94")
    assert "\n" not in rendered
    assert len(rendered.removeprefix("\r")) == 80


def test_motion_viewer_recording_toggle_pauses_without_closing_viewer(
    tmp_path: Path,
) -> None:
    motion = FakeReferenceMotion(frame_count=3, marker=1.0)
    motion.fps = 2.0
    scene_adapter = FakeSceneAdapter()
    viewer = MotionViewer([motion], scene_adapter)
    handle = KeyEventViewerHandle(
        running_checks=4,
        key_events={1: ord("\\"), 2: ord("\\"), 3: 262},
    )
    recording_calls: list[tuple[int, Path]] = []
    output_path = tmp_path / "videos" / "run" / "walk.mp4"

    def launch_passive(model, data, *, key_callback, show_left_ui, show_right_ui):
        handle.key_callback = key_callback
        return handle

    def recording_executor(motion_index: int, path: Path) -> None:
        assert handle.closed is False
        recording_calls.append((motion_index, path))

    viewer.run_interactive(
        launch_passive=launch_passive,
        clock=lambda: 10.0,
        sleep=lambda _: None,
        recording_output_paths=[output_path],
        recording_executor=recording_executor,
    )

    assert recording_calls == [(0, output_path)]
    assert viewer.controller.selected_motion_index == 0
    assert handle.closed is True


def test_motion_viewer_continues_same_session_after_interactive_recording(
    tmp_path: Path,
) -> None:
    motions = [
        FakeReferenceMotion(frame_count=3, marker=1.0),
        FakeReferenceMotion(frame_count=3, marker=2.0),
    ]
    scene_adapter = FakeSceneAdapter()
    viewer = MotionViewer(motions, scene_adapter)
    handle = KeyEventViewerHandle(
        running_checks=4,
        key_events={1: ord("\\"), 2: ord("\\"), 3: 262},
    )
    recording_calls: list[tuple[int, Path]] = []
    output_path = tmp_path / "videos" / "run" / "first.mp4"

    def launch_passive(model, data, *, key_callback, show_left_ui, show_right_ui):
        handle.key_callback = key_callback
        return handle

    def recording_executor(motion_index: int, path: Path) -> None:
        assert handle.closed is False
        recording_calls.append((motion_index, path))

    viewer.run_interactive(
        launch_passive=launch_passive,
        clock=lambda: 10.0,
        sleep=lambda _: None,
        recording_output_paths=[
            output_path,
            tmp_path / "videos" / "run" / "second.mp4",
        ],
        recording_executor=recording_executor,
    )

    assert recording_calls == [(0, output_path)]
    assert handle.closed is True
    assert viewer.controller.selected_motion_index == 1


def test_motion_viewer_blocks_motion_switching_while_recording(
    tmp_path: Path,
) -> None:
    motions = [
        FakeReferenceMotion(frame_count=3, marker=1.0),
        FakeReferenceMotion(frame_count=3, marker=2.0),
    ]
    viewer = MotionViewer(motions, FakeSceneAdapter())
    handle = KeyEventViewerHandle(
        running_checks=4,
        key_events={
            1: ord("\\"),
            2: 262,
            3: ord("\\"),
        },
    )
    attachment = FakeRecordingAttachment()
    status_updates: list[str] = []

    def launch_passive(model, data, *, key_callback, show_left_ui, show_right_ui):
        handle.key_callback = key_callback
        return handle

    viewer.run_interactive(
        launch_passive=launch_passive,
        clock=lambda: 10.0,
        sleep=lambda _: None,
        status_reporter=status_updates.append,
        recording_attachment=attachment,
        recording_output_paths=[
            tmp_path / "videos" / "run" / "first.mp4",
            tmp_path / "videos" / "run" / "second.mp4",
        ],
        frame_renderer_factory=FakeFrameRenderer,
    )

    assert viewer.controller.selected_motion_index == 0
    assert attachment.started == [(tmp_path / "videos" / "run" / "first.mp4", 30.0)]
    assert attachment.captured == ["frame-1", "frame-2", "frame-3"]
    assert any(
        "stop recording before switching motions" in status
        for status in status_updates
    )


def test_motion_viewer_finalizes_recording_when_closed(tmp_path: Path) -> None:
    motion = FakeReferenceMotion(frame_count=3, marker=1.0)
    viewer = MotionViewer([motion], FakeSceneAdapter())
    handle = KeyEventViewerHandle(running_checks=3, key_events={1: ord("\\")})
    attachment = FakeRecordingAttachment()

    def launch_passive(model, data, *, key_callback, show_left_ui, show_right_ui):
        handle.key_callback = key_callback
        return handle

    viewer.run_interactive(
        launch_passive=launch_passive,
        clock=lambda: 10.0,
        sleep=lambda _: None,
        recording_attachment=attachment,
        recording_output_paths=[tmp_path / "videos" / "run" / "walk.mp4"],
        frame_renderer_factory=FakeFrameRenderer,
    )

    assert attachment.captured == ["frame-1", "frame-2", "frame-3"]
    assert attachment.stopped == 1
    assert handle.closed is True


def test_motion_viewer_does_not_capture_after_attachment_becomes_inactive(
    tmp_path: Path,
) -> None:
    motion = FakeReferenceMotion(frame_count=3, marker=1.0)
    viewer = MotionViewer([motion], FakeSceneAdapter())
    handle = KeyEventViewerHandle(running_checks=3, key_events={1: ord("\\")})
    attachment = RecordingAttachment(write_video=lambda *_, **__: None)

    class StoppingFrameRenderer:
        closed = False

        def render_frame(self):
            attachment.stop()
            return "stale-frame"

        def close(self) -> None:
            self.closed = True

    def launch_passive(model, data, *, key_callback, show_left_ui, show_right_ui):
        handle.key_callback = key_callback
        return handle

    viewer.run_interactive(
        launch_passive=launch_passive,
        clock=lambda: 10.0,
        sleep=lambda _: None,
        recording_attachment=attachment,
        recording_output_paths=[tmp_path / "videos" / "run" / "walk.mp4"],
        frame_renderer_factory=StoppingFrameRenderer,
    )

    assert handle.closed is True


def test_motion_viewer_allows_pause_and_speed_changes_while_recording(
    tmp_path: Path,
) -> None:
    motion = FakeReferenceMotion(frame_count=3, marker=1.0)
    viewer = MotionViewer([motion], FakeSceneAdapter())
    handle = KeyEventViewerHandle(
        running_checks=5,
        key_events={
            1: ord("\\"),
            2: 265,
            3: 32,
            4: ord("\\"),
        },
    )

    def launch_passive(model, data, *, key_callback, show_left_ui, show_right_ui):
        handle.key_callback = key_callback
        return handle

    viewer.run_interactive(
        launch_passive=launch_passive,
        clock=lambda: 10.0,
        sleep=lambda _: None,
        recording_attachment=FakeRecordingAttachment(),
        recording_output_paths=[tmp_path / "videos" / "run" / "walk.mp4"],
        frame_renderer_factory=FakeFrameRenderer,
    )

    assert viewer.controller.playback_speed == 1.25
    assert viewer.controller.paused is True


def test_motion_viewer_records_headless_batch_deterministically(
    tmp_path: Path,
) -> None:
    motions = [
        FakeReferenceMotion(frame_count=2, marker=1.0),
        FakeReferenceMotion(frame_count=3, marker=2.0),
    ]
    motions[0].name = "walk.npz"
    motions[0].fps = 24.0
    motions[1].name = "jump.npz"
    motions[1].fps = 50.0
    scene_adapter = FakeSceneAdapter()
    viewer = MotionViewer(motions, scene_adapter)
    attachments: list[FakeHeadlessRecordingAttachment] = []
    renderer = FakeFrameRenderer()
    status_updates: list[str] = []

    def recording_attachment_factory():
        attachment = FakeHeadlessRecordingAttachment()
        attachments.append(attachment)
        return attachment

    results = viewer.record_headless(
        recording_output_paths=[
            tmp_path / "videos" / "run" / "walk.mp4",
            tmp_path / "videos" / "run" / "jump.mp4",
        ],
        recording_attachment_factory=recording_attachment_factory,
        frame_renderer_factory=lambda: renderer,
        status_reporter=status_updates.append,
    )

    assert scene_adapter.applied == [
        (motions[0], 0),
        (motions[0], 1),
        (motions[1], 0),
        (motions[1], 1),
        (motions[1], 2),
    ]
    assert [attachment.started for attachment in attachments] == [
        [(tmp_path / "videos" / "run" / "walk.mp4", 24.0)],
        [(tmp_path / "videos" / "run" / "jump.mp4", 50.0)],
    ]
    assert [attachment.captured for attachment in attachments] == [
        ["frame-1", "frame-2"],
        ["frame-3", "frame-4", "frame-5"],
    ]
    assert [attachment.stopped for attachment in attachments] == [1, 1]
    assert [result.output_path for result in results] == [
        tmp_path / "videos" / "run" / "walk.mp4",
        tmp_path / "videos" / "run" / "jump.mp4",
    ]
    assert status_updates == [
        f"Recording 1/2: walk.npz -> {tmp_path / 'videos' / 'run' / 'walk.mp4'}",
        "Recording walk.npz: frame 1/2",
        "Recording walk.npz: frame 2/2",
        f"Saved recording: {tmp_path / 'videos' / 'run' / 'walk.mp4'}",
        f"Recording 2/2: jump.npz -> {tmp_path / 'videos' / 'run' / 'jump.mp4'}",
        "Recording jump.npz: frame 1/3",
        "Recording jump.npz: frame 2/3",
        "Recording jump.npz: frame 3/3",
        f"Saved recording: {tmp_path / 'videos' / 'run' / 'jump.mp4'}",
    ]
    assert renderer.closed is True


def test_motion_viewer_headless_recording_uses_motion_fps(
    tmp_path: Path,
) -> None:
    motion = FakeReferenceMotion(frame_count=1, marker=1.0)
    motion.fps = 24.0
    viewer = MotionViewer([motion], FakeSceneAdapter())
    attachment = FakeHeadlessRecordingAttachment()

    viewer.record_headless(
        recording_output_paths=[tmp_path / "walk.mp4"],
        recording_attachment_factory=lambda: attachment,
        frame_renderer_factory=FakeFrameRenderer,
    )

    assert attachment.started == [(tmp_path / "walk.mp4", 24.0)]


def test_motion_viewer_headless_recording_fails_fast_on_scene_error(
    tmp_path: Path,
) -> None:
    class FailingSceneAdapter(FakeSceneAdapter):
        def apply(self, motion: FakeReferenceMotion, frame_index: int) -> None:
            super().apply(motion, frame_index)
            if frame_index == 1:
                raise RuntimeError("scene apply failed")

    motions = [
        FakeReferenceMotion(frame_count=3, marker=1.0),
        FakeReferenceMotion(frame_count=2, marker=2.0),
    ]
    scene_adapter = FailingSceneAdapter()
    viewer = MotionViewer(motions, scene_adapter)
    attachments: list[FakeHeadlessRecordingAttachment] = []
    renderer = FakeFrameRenderer()

    def recording_attachment_factory():
        attachment = FakeHeadlessRecordingAttachment()
        attachments.append(attachment)
        return attachment

    with pytest.raises(RuntimeError, match="scene apply failed"):
        viewer.record_headless(
            recording_output_paths=[tmp_path / "first.mp4", tmp_path / "second.mp4"],
            recording_attachment_factory=recording_attachment_factory,
            frame_renderer_factory=lambda: renderer,
        )

    assert scene_adapter.applied == [(motions[0], 0), (motions[0], 1)]
    assert len(attachments) == 1
    assert attachments[0].captured == ["frame-1"]
    assert attachments[0].stopped == 0
    assert renderer.closed is True


def test_motion_viewer_key_callback_maps_controls_to_playback_actions() -> None:
    motions = [
        FakeReferenceMotion(frame_count=2, marker=1.0),
        FakeReferenceMotion(frame_count=2, marker=2.0),
    ]
    viewer = MotionViewer(motions, FakeSceneAdapter())

    viewer.handle_key(262)
    assert viewer.controller.selected_motion_index == 1

    viewer.handle_key(263)
    assert viewer.controller.selected_motion_index == 0

    viewer.handle_key(265)
    assert viewer.controller.playback_speed == 1.25

    viewer.handle_key(264)
    assert viewer.controller.playback_speed == 1.0

    viewer.handle_key(32)
    assert viewer.controller.paused is True


def test_motion_viewer_reports_current_motion_speed_and_progress() -> None:
    motions = [
        FakeReferenceMotion(frame_count=2, marker=1.0),
        FakeReferenceMotion(frame_count=3, marker=2.0),
    ]
    motions[1].name = "walking.npz"
    viewer = MotionViewer(motions, FakeSceneAdapter())

    viewer.handle_action("right")
    viewer.handle_action("up")
    viewer.advance(frames=1.0)

    assert (
        viewer.playback_status()
        == "Motion 2/2 | walking.npz | [=======-------------] 1/3 (33.3%) | speed 1.25x | playing"
    )

    viewer.handle_action("space")

    assert (
        viewer.playback_status()
        == "Motion 2/2 | walking.npz | [=======-------------] 1/3 (33.3%) | speed 1.25x | paused"
    )


def test_motion_viewer_status_preserves_supplied_motion_names() -> None:
    motion = FakeReferenceMotion(frame_count=984, marker=1.0)
    motion.name = "0007_0007_Walking001_poses_keypoints_retargeted.npz"
    viewer = MotionViewer([motion], FakeSceneAdapter())
    viewer.advance(frames=131.0)

    assert (
        viewer.playback_status()
        == "Motion 1/1 | 0007_0007_Walking001_poses_keypoints_retargeted.npz | "
        "[===-----------------] 131/984 (13.3%) | speed 1x | playing"
    )


def test_motion_viewer_status_uses_prepared_display_name_when_available() -> None:
    motion = FakeReferenceMotion(frame_count=984, marker=1.0)
    motion.name = "0007_0007_Walking001_poses_keypoints_retargeted.npz"
    motion.display_name = "0007_0007_Walking001"
    viewer = MotionViewer([motion], FakeSceneAdapter())
    viewer.advance(frames=131.0)

    assert (
        viewer.playback_status()
        == "Motion 1/1 | 0007_0007_Walking001 | "
        "[===-----------------] 131/984 (13.3%) | speed 1x | playing"
    )
