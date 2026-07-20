from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from mjlab_playground.motion_lib import ReferenceMotion, ReferenceMotionClipSpan
from mjlab_playground.motion_lib.motion_viewer import (
    MotionScene,
    MotionViewer,
    PlaybackAction,
    RecordingStatus,
    ViewerError,
    ViewerSnapshot,
    ViewerTick,
)
from mjlab_playground.motion_lib.recording import (
    RecordingRequest,
    RecordingRequestTarget,
    SubprocessBackgroundRecorder,
)


class FakeReferenceMotion:
    def __init__(self, *, name: str, frame_count: int, fps: float) -> None:
        self.name = name
        self.fps = fps
        self.root_pos = torch.zeros(frame_count, 3)
        self.clip_name_bytes = None

    def iter_clip_spans(self):
        yield ReferenceMotionClipSpan(
            parent=self,  # type: ignore[arg-type]
            clip_id=0,
            start_frame=0,
            frame_count=int(self.root_pos.shape[0]),
            fps=self.fps,
        )


class InMemoryMotionScene:
    def __init__(self) -> None:
        self.applications: list[tuple[FakeReferenceMotion, int]] = []
        self.close_calls = 0

    def apply_reference_frame(
        self,
        motion: FakeReferenceMotion,
        frame_index: int,
    ) -> None:
        self.applications.append((motion, frame_index))

    def close(self) -> None:
        self.close_calls += 1


class FailingMotionScene(InMemoryMotionScene):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def apply_reference_frame(
        self,
        motion: FakeReferenceMotion,
        frame_index: int,
    ) -> None:
        raise self.error


class ScriptedViewerAdapter:
    def __init__(self, ticks: list[ViewerTick]) -> None:
        self.ticks = ticks
        self.snapshots: list[ViewerSnapshot] = []
        self.close_calls = 0

    def run(self, update) -> None:
        for tick in self.ticks:
            snapshot = update(tick)
            self.snapshots.append(snapshot)
            if snapshot.stop_requested:
                break

    def close(self) -> None:
        self.close_calls += 1


class FakeBackgroundRecorder:
    def __init__(self) -> None:
        self.status = RecordingStatus.IDLE
        self.requests: list[tuple[int, ReferenceMotionClipSpan]] = []
        self.wait_calls = 0
        self.cancel_calls = 0
        self.close_calls = 0

    def record_selected_clip(
        self,
        clip_index: int,
        clip: ReferenceMotionClipSpan,
    ) -> None:
        self.requests.append((clip_index, clip))
        self.status = RecordingStatus.RUNNING

    def poll(self) -> None:
        pass

    def wait(self) -> None:
        self.wait_calls += 1
        self.status = RecordingStatus.SUCCEEDED

    def cancel(self) -> None:
        self.cancel_calls += 1
        self.status = RecordingStatus.FAILED

    def close(self) -> None:
        self.close_calls += 1

    @property
    def output_path(self) -> Path | None:
        return None

    @property
    def error(self) -> str | None:
        return None


class SpyStatusReporter:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.close_calls = 0

    def __call__(self, status: str) -> None:
        self.events.append(status)

    def close(self) -> None:
        self.close_calls += 1


class ActiveLifecycleRecorder(FakeBackgroundRecorder):
    def __init__(self, events: list[str], output_path: Path) -> None:
        super().__init__()
        self.events = events
        self.status = RecordingStatus.RUNNING
        self._output_path = output_path
        self._error: str | None = None

    def wait(self) -> None:
        self.wait_calls += 1
        self.events.append("recorder_wait")
        self.status = RecordingStatus.SUCCEEDED

    def close(self) -> None:
        self.close_calls += 1
        self.events.append("recorder_close")

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    @property
    def error(self) -> str | None:
        return self._error


class OrderedCloseAdapter(ScriptedViewerAdapter):
    def __init__(self, ticks: list[ViewerTick], events: list[str]) -> None:
        super().__init__(ticks)
        self.events = events

    def close(self) -> None:
        super().close()
        self.events.append("adapter_close")


def test_run_flattens_logical_clips_and_applies_packed_parent_frames() -> None:
    package = ReferenceMotion(
        name="package.npz",
        fps=30.0,
        root_pos=torch.zeros(5, 3),
        root_rot=torch.zeros(5, 4),
        dof_pos=torch.zeros(5, 2),
        clip_starts=torch.tensor([0, 2]),
        clip_lengths=torch.tensor([2, 3]),
        clip_fps=torch.tensor([2.0, 4.0]),
        clip_name_bytes=torch.tensor(list(b"walkjump"), dtype=torch.uint8),
        clip_name_offsets=torch.tensor([0, 4, 8]),
    )
    balance = ReferenceMotion(
        name="balance",
        fps=1.0,
        root_pos=torch.zeros(4, 3),
        root_rot=torch.zeros(4, 4),
        dof_pos=torch.zeros(4, 2),
    )
    scene = InMemoryMotionScene()
    adapter = ScriptedViewerAdapter(
        [
            ViewerTick(0.0),
            ViewerTick(0.0, (PlaybackAction.NEXT_CLIP,)),
            ViewerTick(0.25),
            ViewerTick(0.0, (PlaybackAction.NEXT_CLIP,)),
            ViewerTick(0.0, (PlaybackAction.NEXT_CLIP,)),
            ViewerTick(0.0, (PlaybackAction.PREVIOUS_CLIP,)),
        ]
    )

    MotionViewer([package, balance], scene, viewer_adapter=adapter).run()

    assert scene.applications == [
        (package, 0),
        (package, 2),
        (package, 3),
        (balance, 0),
        (package, 0),
        (balance, 0),
    ]
    assert [
        (
            snapshot.selected_clip_index,
            snapshot.selected_clip_name,
            snapshot.clip_count,
            snapshot.frame_index,
            snapshot.frame_count,
        )
        for snapshot in adapter.snapshots
    ] == [
        (0, "walk", 3, 0, 2),
        (1, "jump", 3, 0, 3),
        (1, "jump", 3, 1, 3),
        (2, "balance", 3, 0, 4),
        (0, "walk", 3, 0, 2),
        (2, "balance", 3, 0, 4),
    ]


def test_run_loops_within_each_selected_clip_boundary() -> None:
    package = ReferenceMotion(
        name="package.npz",
        root_pos=torch.zeros(5, 3),
        root_rot=torch.zeros(5, 4),
        dof_pos=torch.zeros(5, 2),
        clip_starts=torch.tensor([0, 2]),
        clip_lengths=torch.tensor([2, 3]),
        clip_fps=torch.tensor([2.0, 4.0]),
        clip_name_bytes=torch.tensor(list(b"walkjump"), dtype=torch.uint8),
        clip_name_offsets=torch.tensor([0, 4, 8]),
    )
    scene = InMemoryMotionScene()
    adapter = ScriptedViewerAdapter(
        [
            ViewerTick(1.5),
            ViewerTick(0.0, (PlaybackAction.NEXT_CLIP,)),
            ViewerTick(0.75),
        ]
    )

    MotionViewer([package], scene, viewer_adapter=adapter).run()

    assert scene.applications == [(package, 1), (package, 2), (package, 2)]
    assert [snapshot.status.split(" | ", 2)[:2] for snapshot in adapter.snapshots] == [
        ["Clip 1/2", "walk"],
        ["Clip 2/2", "jump"],
        ["Clip 2/2", "jump"],
    ]


def test_public_viewer_contract_is_limited_to_deep_operations() -> None:
    public_operations = {
        name
        for name, member in vars(MotionViewer).items()
        if not name.startswith("_") and callable(member)
    }
    assert public_operations == {"run", "record"}
    assert set(MotionScene.__annotations__) == set()
    assert not {"mj_model", "mj_data", "apply", "sync_display_data"}.intersection(
        vars(MotionScene)
    )


def test_stop_without_active_recording_closes_owned_resources_once() -> None:
    events: list[str] = []
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    scene = InMemoryMotionScene()
    recorder = FakeBackgroundRecorder()
    reporter = SpyStatusReporter(events)
    adapter = ScriptedViewerAdapter([ViewerTick(0.0, (PlaybackAction.STOP,))])

    MotionViewer(
        [motion],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
        status_reporter=reporter,
    ).run()

    assert adapter.close_calls == 1
    assert recorder.wait_calls == 0
    assert recorder.cancel_calls == 0
    assert recorder.close_calls == 1
    assert scene.close_calls == 1
    assert reporter.close_calls == 1
    assert events == []


def test_stop_with_active_recording_closes_adapter_before_waiting_and_reports_success(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    output_path = tmp_path / "walk.mp4"
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    scene = InMemoryMotionScene()
    recorder = ActiveLifecycleRecorder(events, output_path)
    reporter = SpyStatusReporter(events)
    adapter = OrderedCloseAdapter(
        [ViewerTick(0.0, (PlaybackAction.STOP,))],
        events,
    )

    MotionViewer(
        [motion],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
        status_reporter=reporter,
    ).run()

    assert events[:2] == ["adapter_close", f"Waiting for recording: {output_path}"]
    assert "recorder_wait" in events
    assert f"Recording succeeded: {output_path}" in events
    assert recorder.wait_calls == 1
    assert recorder.cancel_calls == 0
    assert adapter.close_calls == 1
    assert recorder.close_calls == 1
    assert scene.close_calls == 1
    assert reporter.close_calls == 1


def test_terminal_shutdown_reports_recording_failure_and_returns(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    recorder = ActiveLifecycleRecorder(events, tmp_path / "walk.mp4")

    def fail_wait() -> None:
        recorder.wait_calls += 1
        recorder.status = RecordingStatus.FAILED
        recorder._error = "recording child process exited with code 7"

    recorder.wait = fail_wait  # type: ignore[method-assign]
    reporter = SpyStatusReporter(events)
    adapter = ScriptedViewerAdapter([ViewerTick(0.0, (PlaybackAction.STOP,))])

    MotionViewer(
        [motion],
        InMemoryMotionScene(),
        viewer_adapter=adapter,
        background_recorder=recorder,
        status_reporter=reporter,
    ).run()

    assert recorder.wait_calls == 1
    assert recorder.cancel_calls == 0
    assert "Recording failed: recording child process exited with code 7" in events


def test_first_interrupt_stops_presentation_and_second_interrupt_cancels_wait(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    recorder = ActiveLifecycleRecorder(events, tmp_path / "walk.mp4")

    def interrupt_wait() -> None:
        recorder.wait_calls += 1
        events.append("recorder_wait_interrupted")
        raise KeyboardInterrupt

    def cancel() -> None:
        recorder.cancel_calls += 1
        events.append("recorder_cancel")
        recorder.status = RecordingStatus.FAILED
        recorder._error = "recording cancelled"

    recorder.wait = interrupt_wait  # type: ignore[method-assign]
    recorder.cancel = cancel  # type: ignore[method-assign]

    class InterruptingAdapter(OrderedCloseAdapter):
        def run(self, update) -> None:
            del update
            events.append("first_interrupt")
            raise KeyboardInterrupt

    adapter = InterruptingAdapter([], events)

    MotionViewer(
        [motion],
        InMemoryMotionScene(),
        viewer_adapter=adapter,
        background_recorder=recorder,
        status_reporter=SpyStatusReporter(events),
    ).run()

    assert events.index("first_interrupt") < events.index("adapter_close")
    assert events.index("adapter_close") < events.index("recorder_wait_interrupted")
    assert events.index("recorder_wait_interrupted") < events.index("recorder_cancel")
    assert recorder.wait_calls == 1
    assert recorder.cancel_calls == 1
    assert adapter.close_calls == 1


def test_run_applies_initial_frame_and_returns_immutable_snapshot() -> None:
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    motion.display_name = "legacy display label"
    scene = InMemoryMotionScene()
    adapter = ScriptedViewerAdapter([ViewerTick(elapsed_seconds=0.0)])
    viewer = MotionViewer(
        [motion],
        scene,
        viewer_adapter=adapter,
        background_recorder=FakeBackgroundRecorder(),
    )

    viewer.run()

    assert scene.applications == [(motion, 0)]
    assert adapter.close_calls == 1
    assert adapter.snapshots == [
        ViewerSnapshot(
            status=(
                "Clip 1/1 | walk | [--------------------] 0/4 (0.0%) "
                "| speed 1x | playing"
            ),
            selected_clip_index=0,
            selected_clip_name="walk",
            clip_count=1,
            frame_index=0,
            frame_count=4,
            playback_speed=1.0,
            paused=False,
            recording_status=RecordingStatus.IDLE,
            stop_requested=False,
        )
    ]
    with pytest.raises(FrozenInstanceError):
        adapter.snapshots[0].paused = True  # type: ignore[misc]


def test_record_action_captures_selection_at_its_ordered_position() -> None:
    walk = FakeReferenceMotion(name="walk", frame_count=2, fps=2.0)
    jump = FakeReferenceMotion(name="jump", frame_count=2, fps=2.0)
    scene = InMemoryMotionScene()
    recorder = FakeBackgroundRecorder()
    adapter = ScriptedViewerAdapter(
        [
            ViewerTick(
                elapsed_seconds=0.0,
                actions=(
                    PlaybackAction.NEXT_CLIP,
                    PlaybackAction.RECORD_SELECTED_CLIP,
                    PlaybackAction.PREVIOUS_CLIP,
                    PlaybackAction.STOP,
                ),
            )
        ]
    )
    viewer = MotionViewer(
        [walk, jump],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
    )

    viewer.run()

    assert len(recorder.requests) == 1
    requested_index, requested_clip = recorder.requests[0]
    assert (requested_index, requested_clip.parent, requested_clip.clip_id) == (
        1,
        jump,
        0,
    )
    assert adapter.snapshots[0].selected_clip_index == 0
    assert adapter.snapshots[0].recording_status is RecordingStatus.RUNNING


def test_record_action_keeps_nonzero_package_clip_identity_while_interaction_continues(
    tmp_path: Path,
) -> None:
    package = ReferenceMotion(
        name="package.npz",
        root_pos=torch.zeros(5, 3),
        root_rot=torch.zeros(5, 4),
        dof_pos=torch.zeros(5, 2),
        clip_starts=torch.tensor([0, 2]),
        clip_lengths=torch.tensor([2, 3]),
        clip_fps=torch.tensor([2.0, 4.0]),
        clip_name_bytes=torch.tensor(list(b"walkjump"), dtype=torch.uint8),
        clip_name_offsets=torch.tensor([0, 4, 8]),
    )
    targets = (
        RecordingRequestTarget(
            reference_motion_index=0,
            clip_id=0,
            source_path=tmp_path / "package.npz",
            output_path=tmp_path / "walk.mp4",
        ),
        RecordingRequestTarget(
            reference_motion_index=0,
            clip_id=1,
            source_path=tmp_path / "package.npz",
            output_path=tmp_path / "jump.mp4",
        ),
    )

    class Child:
        def poll(self) -> int | None:
            return None

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    requests: list[RecordingRequest] = []
    recorder = SubprocessBackgroundRecorder(
        targets=targets,
        process_factory=lambda request: (requests.append(request), Child())[1],
    )
    scene = InMemoryMotionScene()
    adapter = ScriptedViewerAdapter(
        [
            ViewerTick(
                0.0,
                (
                    PlaybackAction.NEXT_CLIP,
                    PlaybackAction.RECORD_SELECTED_CLIP,
                ),
            ),
            ViewerTick(
                0.5,
                (
                    PlaybackAction.PREVIOUS_CLIP,
                    PlaybackAction.FASTER,
                    PlaybackAction.TOGGLE_PAUSE,
                ),
            ),
            ViewerTick(0.0, (PlaybackAction.STOP,)),
        ]
    )

    MotionViewer(
        [package],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
    ).run()

    assert requests == [RecordingRequest(targets=(targets[1],))]
    assert requests[0].targets[0] == RecordingRequestTarget(
        reference_motion_index=0,
        clip_id=1,
        source_path=tmp_path / "package.npz",
        output_path=tmp_path / "jump.mp4",
    )
    assert [snapshot.selected_clip_index for snapshot in adapter.snapshots] == [
        1,
        0,
        0,
    ]
    assert scene.applications == [(package, 2), (package, 0)]


def test_background_recording_is_polled_while_playback_and_selection_continue(
    tmp_path: Path,
) -> None:
    walk = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    jump = FakeReferenceMotion(name="jump", frame_count=4, fps=2.0)
    targets = (
        RecordingRequestTarget(
            reference_motion_index=0,
            clip_id=0,
            source_path=tmp_path / "walk.npz",
            output_path=tmp_path / "walk.mp4",
        ),
        RecordingRequestTarget(
            reference_motion_index=1,
            clip_id=0,
            source_path=tmp_path / "jump.npz",
            output_path=tmp_path / "jump.mp4",
        ),
    )

    class Child:
        def __init__(self) -> None:
            self.results = iter((None, None, 0))

        def poll(self) -> int | None:
            return next(self.results)

    requests: list[RecordingRequest] = []
    recorder = SubprocessBackgroundRecorder(
        targets=targets,
        process_factory=lambda request: (requests.append(request), Child())[1],
    )
    scene = InMemoryMotionScene()
    adapter = ScriptedViewerAdapter(
        [
            ViewerTick(0.5, (PlaybackAction.RECORD_SELECTED_CLIP,)),
            ViewerTick(
                0.5,
                (
                    PlaybackAction.NEXT_CLIP,
                    PlaybackAction.FASTER,
                    PlaybackAction.TOGGLE_PAUSE,
                ),
            ),
            ViewerTick(0.5, (PlaybackAction.TOGGLE_PAUSE,)),
            ViewerTick(0.0, (PlaybackAction.STOP,)),
        ]
    )
    MotionViewer(
        [walk, jump],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
    ).run()

    assert requests == [RecordingRequest(targets=(targets[0],))]
    assert scene.applications == [(walk, 1), (jump, 0), (jump, 1)]
    assert [snapshot.recording_status for snapshot in adapter.snapshots] == [
        RecordingStatus.RUNNING,
        RecordingStatus.RUNNING,
        RecordingStatus.SUCCEEDED,
        RecordingStatus.SUCCEEDED,
    ]
    assert adapter.snapshots[1].selected_clip_index == 1
    assert adapter.snapshots[1].paused is True
    assert adapter.snapshots[1].playback_speed == 1.25
    assert adapter.snapshots[2].recording_output_path == tmp_path / "walk.mp4"
    assert adapter.snapshots[2].recording_error is None
    assert f"recording running: {tmp_path / 'walk.mp4'}" in adapter.snapshots[0].status
    assert (
        f"recording succeeded: {tmp_path / 'walk.mp4'}" in adapter.snapshots[2].status
    )


def test_second_active_recording_request_is_visible_without_stopping_playback(
    tmp_path: Path,
) -> None:
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    target = RecordingRequestTarget(
        reference_motion_index=0,
        clip_id=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )

    class RunningChild:
        def poll(self) -> None:
            return None

        def wait(self) -> int:
            return 0

    starts = 0

    def start_child(_: RecordingRequest) -> RunningChild:
        nonlocal starts
        starts += 1
        return RunningChild()

    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=start_child,
    )
    scene = InMemoryMotionScene()
    adapter = ScriptedViewerAdapter(
        [
            ViewerTick(
                0.5,
                (
                    PlaybackAction.RECORD_SELECTED_CLIP,
                    PlaybackAction.RECORD_SELECTED_CLIP,
                ),
            )
        ]
    )

    MotionViewer(
        [motion],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
    ).run()

    snapshot = adapter.snapshots[0]
    assert starts == 1
    assert scene.applications == [(motion, 1)]
    assert snapshot.recording_status is RecordingStatus.RUNNING
    assert (
        snapshot.recording_error == f"recording already running: {target.output_path}"
    )
    assert "recording already running" in snapshot.status


def test_playback_actions_are_typed() -> None:
    assert tuple(PlaybackAction) == (
        PlaybackAction.TOGGLE_PAUSE,
        PlaybackAction.PREVIOUS_CLIP,
        PlaybackAction.NEXT_CLIP,
        PlaybackAction.SLOWER,
        PlaybackAction.FASTER,
        PlaybackAction.RECORD_SELECTED_CLIP,
        PlaybackAction.STOP,
    )
    assert not hasattr(PlaybackAction, "PREVIOUS_MOTION")
    assert not hasattr(PlaybackAction, "NEXT_MOTION")

    viewer = MotionViewer(
        [FakeReferenceMotion(name="walk", frame_count=2, fps=2.0)],
        InMemoryMotionScene(),
    )
    assert hasattr(viewer.controller, "selected_clip_index")
    assert not hasattr(viewer.controller, "selected_motion_index")
    assert hasattr(viewer, "selected_clip")
    assert not hasattr(viewer, "selected_motion")


def test_run_preserves_fractional_timing_and_ordered_playback_actions() -> None:
    walk = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    jump = FakeReferenceMotion(name="jump", frame_count=3, fps=4.0)
    scene = InMemoryMotionScene()
    adapter = ScriptedViewerAdapter(
        [
            ViewerTick(elapsed_seconds=0.0),
            ViewerTick(elapsed_seconds=0.25),
            ViewerTick(elapsed_seconds=0.25),
            ViewerTick(
                elapsed_seconds=0.25,
                actions=(PlaybackAction.NEXT_CLIP, PlaybackAction.FASTER),
            ),
            ViewerTick(
                elapsed_seconds=10.0,
                actions=(PlaybackAction.TOGGLE_PAUSE,),
            ),
            ViewerTick(
                elapsed_seconds=0.0,
                actions=(PlaybackAction.PREVIOUS_CLIP,),
            ),
            ViewerTick(
                elapsed_seconds=0.5,
                actions=(PlaybackAction.TOGGLE_PAUSE, PlaybackAction.SLOWER),
            ),
            ViewerTick(elapsed_seconds=0.0, actions=(PlaybackAction.STOP,)),
        ]
    )
    viewer = MotionViewer(
        [walk, jump],
        scene,
        viewer_adapter=adapter,
        background_recorder=FakeBackgroundRecorder(),
    )

    viewer.run()

    assert scene.applications == [
        (walk, 0),
        (walk, 0),
        (walk, 1),
        (jump, 1),
        (jump, 1),
        (walk, 0),
        (walk, 1),
    ]
    assert [
        (
            snapshot.selected_clip_index,
            snapshot.frame_index,
            snapshot.playback_speed,
            snapshot.paused,
            snapshot.stop_requested,
        )
        for snapshot in adapter.snapshots
    ] == [
        (0, 0, 1.0, False, False),
        (0, 0, 1.0, False, False),
        (0, 1, 1.0, False, False),
        (1, 1, 1.25, False, False),
        (1, 1, 1.25, True, False),
        (0, 0, 1.25, True, False),
        (0, 1, 1.0, False, False),
        (0, 1, 1.0, False, True),
    ]
    assert adapter.close_calls == 1


def test_scene_failure_stops_closes_once_and_preserves_original_cause() -> None:
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    cause = RuntimeError("simulator forward failed")
    scene = FailingMotionScene(cause)
    adapter = ScriptedViewerAdapter([ViewerTick(elapsed_seconds=0.0)])
    viewer = MotionViewer([motion], scene, viewer_adapter=adapter)

    with pytest.raises(
        ViewerError,
        match=r"failed to apply frame 0 from clip 1/1 \(walk\)",
    ) as error:
        viewer.run()

    assert error.value.__cause__ is cause
    assert adapter.close_calls == 1
    assert adapter.snapshots == []
    assert viewer.controller.frame_position == 0.0


@pytest.mark.parametrize(
    ("failure_source", "message"),
    [
        ("scene", r"failed to apply frame 0 from clip 1/1 \(walk\)"),
        ("adapter", "reference motion viewer session failed"),
    ],
)
def test_fatal_session_failure_settles_active_recording_and_preserves_primary_error(
    tmp_path: Path,
    failure_source: str,
    message: str,
) -> None:
    events: list[str] = []
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    cause = RuntimeError(f"{failure_source} failed")
    scene = (
        FailingMotionScene(cause)
        if failure_source == "scene"
        else InMemoryMotionScene()
    )
    recorder = ActiveLifecycleRecorder(events, tmp_path / "walk.mp4")
    reporter = SpyStatusReporter(events)

    if failure_source == "adapter":

        class FailingAdapter(ScriptedViewerAdapter):
            def run(self, update) -> None:
                del update
                raise cause

        adapter = FailingAdapter([])
    else:
        adapter = ScriptedViewerAdapter([ViewerTick(0.0)])

    viewer = MotionViewer(
        [motion],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
        status_reporter=reporter,
    )

    with pytest.raises(ViewerError, match=message) as error:
        viewer.run()

    assert error.value.__cause__ is cause
    assert recorder.wait_calls == 1
    assert adapter.close_calls == 1
    assert recorder.close_calls == 1
    assert scene.close_calls == 1
    assert reporter.close_calls == 1


@pytest.mark.parametrize(
    ("failure_source", "message"),
    [
        ("scene", r"failed to apply frame 0 from clip 1/1 \(walk\)"),
        ("adapter", "reference motion viewer session failed"),
    ],
)
def test_adapter_close_failure_does_not_replace_primary_session_error(
    tmp_path: Path,
    failure_source: str,
    message: str,
) -> None:
    events: list[str] = []
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    cause = RuntimeError(f"{failure_source} failed")
    close_error = RuntimeError("adapter close failed")
    scene = (
        FailingMotionScene(cause)
        if failure_source == "scene"
        else InMemoryMotionScene()
    )
    recorder = ActiveLifecycleRecorder(events, tmp_path / "walk.mp4")
    reporter = SpyStatusReporter(events)

    class FailingCloseAdapter(ScriptedViewerAdapter):
        def run(self, update) -> None:
            if failure_source == "adapter":
                raise cause
            super().run(update)

        def close(self) -> None:
            super().close()
            raise close_error

    adapter = FailingCloseAdapter([ViewerTick(0.0)])
    viewer = MotionViewer(
        [motion],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
        status_reporter=reporter,
    )

    with (
        pytest.warns(UserWarning, match="failed to close viewer adapter"),
        pytest.raises(ViewerError, match=message) as error,
    ):
        viewer.run()

    assert error.value.__cause__ is cause
    assert adapter.close_calls == 1
    assert recorder.wait_calls == 1
    assert recorder.close_calls == 1
    assert scene.close_calls == 1
    assert reporter.close_calls == 1


@pytest.mark.parametrize("fatal_session_failure", [False, True])
def test_persistent_recording_ownership_close_failure_is_never_silent(
    tmp_path: Path,
    fatal_session_failure: bool,
) -> None:
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    primary_cause = RuntimeError("simulator forward failed")
    scene = (
        FailingMotionScene(primary_cause)
        if fatal_session_failure
        else InMemoryMotionScene()
    )
    target = RecordingRequestTarget(
        reference_motion_index=0,
        clip_id=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )

    class UnreachableChild:
        def poll(self) -> int | None:
            raise OSError("status unavailable")

        def wait(self) -> int:
            raise OSError("wait unavailable")

        def terminate(self) -> None:
            raise OSError("terminate unavailable")

        def kill(self) -> None:
            raise OSError("kill unavailable")

    starts = 0

    def start_child(_: RecordingRequest) -> UnreachableChild:
        nonlocal starts
        starts += 1
        return UnreachableChild()

    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=start_child,
    )
    target.output_path.write_bytes(b"partial")
    clip = next(motion.iter_clip_spans())
    recorder.record_selected_clip(0, clip)
    reporter = SpyStatusReporter([])
    adapter = ScriptedViewerAdapter(
        [
            ViewerTick(0.0)
            if fatal_session_failure
            else ViewerTick(0.0, (PlaybackAction.STOP,))
        ]
    )
    viewer = MotionViewer(
        [motion],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
        status_reporter=reporter,
    )
    expected_message = (
        r"failed to apply frame 0 from clip 1/1 \(walk\)"
        if fatal_session_failure
        else "failed to close background recorder"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ViewerError, match=expected_message) as error:
            viewer.run()

    if fatal_session_failure:
        assert error.value.__cause__ is primary_cause
        assert any(
            "failed to close background recorder" in str(warning.message)
            for warning in caught
        )
    else:
        assert isinstance(error.value.__cause__, RuntimeError)
        assert "failed to close recording child" in str(error.value.__cause__)
        assert caught == []

    recorder.record_selected_clip(0, next(motion.iter_clip_spans()))
    assert starts == 1
    assert recorder.status is RecordingStatus.RUNNING
    assert target.output_path.read_bytes() == b"partial"
    assert adapter.close_calls == 1
    assert scene.close_calls == 1
    assert reporter.close_calls == 1


def test_fatal_session_failure_is_primary_when_recording_settlement_also_fails(
    tmp_path: Path,
) -> None:
    motion = FakeReferenceMotion(name="walk", frame_count=4, fps=2.0)
    cause = RuntimeError("simulator forward failed")
    scene = FailingMotionScene(cause)

    class FailingSettlementRecorder(ActiveLifecycleRecorder):
        def wait(self) -> None:
            self.wait_calls += 1
            raise RuntimeError("wait observation failed")

        def cancel(self) -> None:
            self.cancel_calls += 1
            raise RuntimeError("cancel failed")

    class FailingReporter(SpyStatusReporter):
        def __call__(self, status: str) -> None:
            self.events.append(status)
            raise RuntimeError("reporting failed")

    events: list[str] = []
    recorder = FailingSettlementRecorder(events, tmp_path / "walk.mp4")
    reporter = FailingReporter(events)
    adapter = ScriptedViewerAdapter([ViewerTick(0.0)])
    viewer = MotionViewer(
        [motion],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
        status_reporter=reporter,
    )

    with (
        pytest.warns(UserWarning),
        pytest.raises(
            ViewerError,
            match=r"failed to apply frame 0 from clip 1/1 \(walk\)",
        ) as error,
    ):
        viewer.run()

    assert error.value.__cause__ is cause
    assert recorder.wait_calls == 1
    assert recorder.cancel_calls == 1
    assert adapter.close_calls == 1
    assert recorder.close_calls == 1
    assert scene.close_calls == 1
    assert reporter.close_calls == 1


@pytest.mark.parametrize("elapsed_seconds", [-1.0, float("nan"), float("inf")])
def test_invalid_elapsed_time_fails_before_actions_or_scene_mutation(
    elapsed_seconds: float,
) -> None:
    walk = FakeReferenceMotion(name="walk", frame_count=2, fps=2.0)
    jump = FakeReferenceMotion(name="jump", frame_count=2, fps=2.0)
    scene = InMemoryMotionScene()
    recorder = FakeBackgroundRecorder()
    adapter = ScriptedViewerAdapter(
        [
            ViewerTick(
                elapsed_seconds=elapsed_seconds,
                actions=(
                    PlaybackAction.NEXT_CLIP,
                    PlaybackAction.RECORD_SELECTED_CLIP,
                ),
            )
        ]
    )
    viewer = MotionViewer(
        [walk, jump],
        scene,
        viewer_adapter=adapter,
        background_recorder=recorder,
    )

    with pytest.raises(
        ViewerError, match="invalid viewer tick elapsed_seconds"
    ) as error:
        viewer.run()

    assert isinstance(error.value.__cause__, ValueError)
    assert viewer.controller.selected_clip_index == 0
    assert viewer.controller.frame_position == 0.0
    assert recorder.requests == []
    assert scene.applications == []
    assert adapter.close_calls == 1
