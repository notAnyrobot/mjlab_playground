from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mjlab_playground.motion_lib.motion_viewer import MotionViewer, RecordingStatus
from mjlab_playground.motion_lib.recording import (
    DeterministicRecorder,
    RecordingAttachment,
    RecordingRequest,
    RecordingRequestTarget,
    SubprocessBackgroundRecorder,
    create_mjlab_deterministic_recorder,
    create_subprocess_background_recorder,
    plan_recording_outputs,
)


class ControllableChildProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.poll_calls = 0
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        self.poll_calls += 1
        return self.returncode

    def wait(self) -> int:
        self.wait_calls += 1
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def test_subprocess_background_recorder_captures_request_and_polls_without_waiting(
    tmp_path: Path,
) -> None:
    motions = [SimpleNamespace(name="walk.npz"), SimpleNamespace(name="jump.npz")]
    targets = (
        RecordingRequestTarget(
            motion_index=0,
            source_path=tmp_path / "walk.npz",
            output_path=tmp_path / "videos" / "walk.mp4",
        ),
        RecordingRequestTarget(
            motion_index=1,
            source_path=tmp_path / "jump.npz",
            output_path=tmp_path / "videos" / "jump.mp4",
        ),
    )
    child = ControllableChildProcess()
    requests: list[RecordingRequest] = []
    recorder = SubprocessBackgroundRecorder(
        targets=targets,
        process_factory=lambda request: (requests.append(request), child)[1],
    )

    recorder.record_selected_clip(1, motions[1])

    assert requests == [RecordingRequest(targets=(targets[1],))]
    assert recorder.status is RecordingStatus.RUNNING
    assert recorder.output_path == tmp_path / "videos" / "jump.mp4"
    assert recorder.error is None

    recorder.poll()
    assert child.poll_calls == 1
    assert recorder.status is RecordingStatus.RUNNING

    child.returncode = 0
    recorder.poll()
    assert child.poll_calls == 2
    assert recorder.status is RecordingStatus.SUCCEEDED
    assert recorder.output_path == tmp_path / "videos" / "jump.mp4"


def test_subprocess_background_recorder_waits_for_active_child_completion(
    tmp_path: Path,
) -> None:
    target = RecordingRequestTarget(
        motion_index=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )
    child = ControllableChildProcess()
    child.returncode = 0
    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=lambda _: child,
    )
    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))

    recorder.wait()

    assert child.wait_calls == 1
    assert recorder.status is RecordingStatus.SUCCEEDED
    assert recorder.output_path == target.output_path
    assert recorder.error is None


def test_subprocess_background_recorder_wait_reports_child_failure(
    tmp_path: Path,
) -> None:
    target = RecordingRequestTarget(
        motion_index=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )
    child = ControllableChildProcess()
    child.returncode = 7
    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=lambda _: child,
    )
    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))

    recorder.wait()

    assert child.wait_calls == 1
    assert recorder.status is RecordingStatus.FAILED
    assert recorder.error == "recording child process exited with code 7"


def test_subprocess_background_recorder_cancel_removes_only_current_incomplete_output(
    tmp_path: Path,
) -> None:
    completed_output = tmp_path / "completed.mp4"
    incomplete_output = tmp_path / "incomplete.mp4"
    targets = (
        RecordingRequestTarget(
            motion_index=0,
            source_path=tmp_path / "completed.npz",
            output_path=completed_output,
        ),
        RecordingRequestTarget(
            motion_index=1,
            source_path=tmp_path / "incomplete.npz",
            output_path=incomplete_output,
        ),
    )
    first_child = ControllableChildProcess()
    first_child.returncode = 0
    second_child = ControllableChildProcess()
    children = iter((first_child, second_child))
    recorder = SubprocessBackgroundRecorder(
        targets=targets,
        process_factory=lambda _: next(children),
    )
    completed_output.write_bytes(b"complete")
    incomplete_output.write_bytes(b"partial")
    recorder.record_selected_clip(0, SimpleNamespace(name="completed.npz"))
    recorder.wait()
    recorder.record_selected_clip(1, SimpleNamespace(name="incomplete.npz"))

    recorder.cancel()

    assert second_child.poll_calls == 1
    assert second_child.terminate_calls == 1
    assert second_child.wait_calls == 1
    assert recorder.status is RecordingStatus.FAILED
    assert recorder.error == "recording cancelled"
    assert completed_output.read_bytes() == b"complete"
    assert not incomplete_output.exists()


def test_subprocess_background_recorder_cancel_preserves_output_if_child_just_completed(
    tmp_path: Path,
) -> None:
    target = RecordingRequestTarget(
        motion_index=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )
    child = ControllableChildProcess()
    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=lambda _: child,
    )
    target.output_path.write_bytes(b"complete")
    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    child.returncode = 0

    recorder.cancel()

    assert child.poll_calls == 1
    assert child.terminate_calls == 0
    assert child.wait_calls == 0
    assert recorder.status is RecordingStatus.SUCCEEDED
    assert recorder.error is None
    assert target.output_path.read_bytes() == b"complete"


@pytest.mark.parametrize("failure_stage", ["terminate", "wait"])
def test_subprocess_background_recorder_cancel_uses_kill_fallback_before_releasing_ownership(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    target = RecordingRequestTarget(
        motion_index=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )

    class FallbackChild(ControllableChildProcess):
        def terminate(self) -> None:
            self.terminate_calls += 1
            if failure_stage == "terminate":
                raise OSError("terminate failed")
            self.returncode = -15

        def wait(self) -> int:
            self.wait_calls += 1
            if failure_stage == "wait" and self.wait_calls == 1:
                raise OSError("wait failed")
            assert self.returncode is not None
            return self.returncode

    first_child = FallbackChild()
    second_child = ControllableChildProcess()
    starts = iter((first_child, second_child))
    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=lambda _: next(starts),
    )
    target.output_path.write_bytes(b"partial")
    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))

    recorder.cancel()

    assert first_child.terminate_calls == 1
    assert first_child.kill_calls == 1
    assert first_child.wait_calls == (1 if failure_stage == "terminate" else 2)
    assert recorder.status is RecordingStatus.FAILED
    assert recorder.error == "recording cancelled"
    assert not target.output_path.exists()

    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    assert recorder.status is RecordingStatus.RUNNING
    second_child.returncode = 0
    recorder.wait()
    assert recorder.status is RecordingStatus.SUCCEEDED


def test_subprocess_background_recorder_close_retries_retained_child_cancellation(
    tmp_path: Path,
) -> None:
    target = RecordingRequestTarget(
        motion_index=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )

    class RecoverableChild(ControllableChildProcess):
        def terminate(self) -> None:
            self.terminate_calls += 1
            if self.terminate_calls == 1:
                raise OSError("terminate temporarily failed")
            self.returncode = -15

        def kill(self) -> None:
            self.kill_calls += 1
            if self.kill_calls == 1:
                raise OSError("kill temporarily failed")
            self.returncode = -9

    retained_child = RecoverableChild()
    later_child = ControllableChildProcess()
    children = iter((retained_child, later_child))
    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=lambda _: next(children),
    )
    target.output_path.write_bytes(b"partial")
    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    recorder.cancel()
    assert recorder.status is RecordingStatus.RUNNING

    recorder.close()

    assert retained_child.terminate_calls == 2
    assert retained_child.kill_calls == 1
    assert retained_child.wait_calls == 1
    assert recorder.status is RecordingStatus.FAILED
    assert recorder.error == "recording cancelled"
    assert not target.output_path.exists()

    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    assert recorder.status is RecordingStatus.RUNNING
    later_child.returncode = 0
    recorder.wait()
    assert recorder.status is RecordingStatus.SUCCEEDED


def test_subprocess_background_recorder_close_reports_persistent_child_ownership(
    tmp_path: Path,
) -> None:
    target = RecordingRequestTarget(
        motion_index=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )

    class UnreachableChild(ControllableChildProcess):
        def poll(self) -> int | None:
            self.poll_calls += 1
            raise OSError("status unavailable")

        def terminate(self) -> None:
            self.terminate_calls += 1
            raise OSError("terminate unavailable")

        def kill(self) -> None:
            self.kill_calls += 1
            raise OSError("kill unavailable")

    child = UnreachableChild()
    starts = 0

    def start_child(_: RecordingRequest) -> UnreachableChild:
        nonlocal starts
        starts += 1
        return child

    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=start_child,
    )
    target.output_path.write_bytes(b"partial")
    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))

    with pytest.raises(RuntimeError, match="failed to close recording child"):
        recorder.close()

    assert recorder.status is RecordingStatus.RUNNING
    assert "forced termination failed" in (recorder.error or "")
    assert target.output_path.read_bytes() == b"partial"
    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    assert starts == 1
    assert recorder.status is RecordingStatus.RUNNING


def test_subprocess_background_recorder_rejects_active_request_and_recovers_for_retry(
    tmp_path: Path,
) -> None:
    target = RecordingRequestTarget(
        motion_index=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )
    first_child = ControllableChildProcess()
    second_child = ControllableChildProcess()
    children = iter((first_child, second_child))
    requests: list[RecordingRequest] = []
    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=lambda request: (requests.append(request), next(children))[1],
    )

    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))

    assert len(requests) == 1
    assert recorder.status is RecordingStatus.RUNNING
    assert recorder.error == f"recording already running: {target.output_path}"

    first_child.returncode = 7
    recorder.poll()
    assert recorder.status is RecordingStatus.FAILED
    assert recorder.error == "recording child process exited with code 7"

    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    assert len(requests) == 2
    assert recorder.status is RecordingStatus.RUNNING
    assert recorder.error is None

    second_child.returncode = 0
    recorder.poll()
    assert recorder.status is RecordingStatus.SUCCEEDED


def test_subprocess_background_recorder_reports_child_start_failure_and_allows_retry(
    tmp_path: Path,
) -> None:
    target = RecordingRequestTarget(
        motion_index=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )
    child = ControllableChildProcess()
    attempts = 0

    def start_child(request: RecordingRequest) -> ControllableChildProcess:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("process table full")
        return child

    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=start_child,
    )

    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    assert recorder.status is RecordingStatus.FAILED
    assert recorder.error == "failed to start recording child: process table full"

    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    assert recorder.status is RecordingStatus.RUNNING
    assert recorder.error is None


def test_subprocess_background_recorder_factory_reuses_headless_recording_command(
    tmp_path: Path,
) -> None:
    target = RecordingRequestTarget(
        motion_index=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "videos" / "walk.mp4",
    )
    child = ControllableChildProcess()
    commands: list[list[str]] = []
    recorder = create_subprocess_background_recorder(
        targets=(target,),
        motion_format="mjlab",
        fps=50.0,
        robot="astro",
        device="cpu",
        process_launcher=lambda command: (commands.append(command), child)[1],
    )

    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))

    assert commands == [
        [
            sys.executable,
            "-m",
            "mjlab_playground.motion_lib.scripts.launch_motion_viewer",
            "--motion-files",
            str(target.source_path),
            "--format",
            "mjlab",
            "--fps",
            "50.0",
            "--robot",
            "astro",
            "--device",
            "cpu",
            "--headless",
            "--record-video",
            "--output-dir",
            str(target.output_path.parent),
        ]
    ]


@pytest.mark.parametrize("failure_method", ["poll", "wait"])
@pytest.mark.parametrize("recovery_method", ["wait", "cancel"])
def test_subprocess_background_recorder_retains_child_ownership_after_observation_failure(
    tmp_path: Path,
    failure_method: str,
    recovery_method: str,
) -> None:
    target = RecordingRequestTarget(
        motion_index=0,
        source_path=tmp_path / "walk.npz",
        output_path=tmp_path / "walk.mp4",
    )

    class FlakyChild(ControllableChildProcess):
        def __init__(self) -> None:
            super().__init__()
            self.failure_pending = True

        def poll(self) -> int | None:
            self.poll_calls += 1
            if failure_method == "poll" and self.failure_pending:
                self.failure_pending = False
                raise OSError("child status unavailable")
            return self.returncode

        def wait(self) -> int:
            self.wait_calls += 1
            if failure_method == "wait" and self.failure_pending:
                self.failure_pending = False
                raise OSError("child wait unavailable")
            assert self.returncode is not None
            return self.returncode

    child = FlakyChild()
    starts = 0

    def start_child(_: RecordingRequest) -> FlakyChild:
        nonlocal starts
        starts += 1
        return child

    recorder = SubprocessBackgroundRecorder(
        targets=(target,),
        process_factory=start_child,
    )
    target.output_path.write_bytes(b"partial")

    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    getattr(recorder, failure_method)()

    assert recorder.status is RecordingStatus.RUNNING
    assert recorder.error == (
        "failed to poll recording child: child status unavailable"
        if failure_method == "poll"
        else "failed to wait for recording child: child wait unavailable"
    )

    recorder.record_selected_clip(0, SimpleNamespace(name="walk.npz"))
    assert starts == 1
    assert recorder.status is RecordingStatus.RUNNING
    assert recorder.error == f"recording already running: {target.output_path}"

    if recovery_method == "wait":
        child.returncode = 0
        recorder.wait()
        assert recorder.status is RecordingStatus.SUCCEEDED
        assert target.output_path.read_bytes() == b"partial"
    else:
        recorder.cancel()
        assert child.terminate_calls == 1
        assert recorder.status is RecordingStatus.FAILED
        assert recorder.error == "recording cancelled"
        assert not target.output_path.exists()


def test_recording_attachment_writes_captured_frames_when_stopped(
    tmp_path: Path,
) -> None:
    calls = []

    def write_video(path, frames, *, fps):
        calls.append((Path(path), list(frames), fps))

    recorder = RecordingAttachment(write_video=write_video)
    output_path = tmp_path / "videos" / "run" / "walk.mp4"
    first_frame = np.zeros((2, 3, 3), dtype=np.uint8)
    second_frame = np.full((2, 3, 3), 255, dtype=np.uint8)

    recorder.start(output_path, fps=30.0)
    recorder.capture(first_frame)
    recorder.capture(second_frame)
    result = recorder.stop()

    assert result.written is True
    assert result.output_path == output_path
    assert result.frame_count == 2
    assert result.fps == 30.0
    assert result.reason is None
    assert calls == [(output_path, [first_frame, second_frame], 30.0)]


def test_recording_attachment_default_writer_uses_mediapy_write_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "mediapy",
        SimpleNamespace(
            write_video=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
    )
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    output_path = tmp_path / "video.mp4"

    recorder = RecordingAttachment()
    recorder.start(output_path, fps=60.0)
    recorder.capture(frame)
    result = recorder.stop()

    assert result.written is True
    assert calls == [((output_path, [frame]), {"fps": 60.0})]


def test_recording_attachment_stop_with_no_frames_returns_no_output(
    tmp_path: Path,
) -> None:
    calls = []
    output_path = tmp_path / "videos" / "run" / "empty.mp4"
    recorder = RecordingAttachment(
        write_video=lambda *args, **kwargs: calls.append((args, kwargs))
    )

    recorder.start(output_path, fps=24.0)
    result = recorder.stop()

    assert result.written is False
    assert result.output_path == output_path
    assert result.frame_count == 0
    assert result.fps == 24.0
    assert result.reason == "no frames captured"
    assert calls == []
    assert not output_path.exists()


def test_plan_recording_outputs_uses_one_timestamped_run_directory(
    tmp_path: Path,
) -> None:
    single_motion = tmp_path / "single" / "walk.npz"
    single_motion.parent.mkdir()
    single_motion.write_bytes(b"motion")

    single_plan = plan_recording_outputs(single_motion, timestamp="20260702-101112")

    assert single_plan.output_dir == tmp_path / "renderings" / "20260702-101112"
    assert single_plan.output_dir.is_dir()
    assert single_plan.targets[0].source_path == single_motion
    assert single_plan.targets[0].output_path == (
        tmp_path / "renderings" / "20260702-101112" / "walk.mp4"
    )

    motion_dir = tmp_path / "batch"
    motion_dir.mkdir()
    (motion_dir / "run.motion").write_bytes(b"motion")
    (motion_dir / "jump.npz").write_bytes(b"motion")
    (motion_dir / "nested").mkdir()

    batch_plan = plan_recording_outputs(
        motion_dir,
        timestamp="20260702-101112",
        motion_names=["jump.npz", "run.motion"],
    )

    assert batch_plan.output_dir == tmp_path / "renderings" / "20260702-101112"
    assert batch_plan.output_dir.is_dir()
    assert [
        (target.source_path.name, target.output_path.name)
        for target in batch_plan.targets
    ] == [
        ("jump.npz", "jump.mp4"),
        ("run.motion", "run.mp4"),
    ]


def test_plan_recording_outputs_rejects_raw_directory_without_motion_names(
    tmp_path: Path,
) -> None:
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    (motion_dir / "walk.npz").write_bytes(b"motion")

    with pytest.raises(ValueError, match="motion_names are required"):
        plan_recording_outputs(motion_dir, timestamp="20260702-101112")


def test_plan_recording_outputs_uses_loader_selected_motion_names(
    tmp_path: Path,
) -> None:
    motion_dir = tmp_path / "accad" / "pyroki-retargeted-astro"
    motion_dir.mkdir(parents=True)
    (motion_dir / "jump.npz").write_bytes(b"motion")
    (motion_dir / "walk.npz").write_bytes(b"motion")
    (motion_dir / "README.txt").write_text("sidecar")
    (motion_dir / "legacy.motion").write_bytes(b"proto")

    plan = plan_recording_outputs(
        motion_dir,
        timestamp="20260702-101112",
        motion_names=["jump.npz", "walk.npz"],
    )

    assert plan.output_dir == tmp_path / "accad" / "renderings" / "20260702-101112"
    assert [
        (target.source_path.name, target.output_path.name) for target in plan.targets
    ] == [
        ("jump.npz", "jump.mp4"),
        ("walk.npz", "walk.mp4"),
    ]


def test_plan_recording_outputs_accepts_output_dir_override(tmp_path: Path) -> None:
    motion_path = tmp_path / "motions" / "walk.npz"
    motion_path.parent.mkdir()
    motion_path.write_bytes(b"motion")
    output_dir = tmp_path / "custom-videos"

    plan = plan_recording_outputs(
        motion_path,
        timestamp="20260702-101112",
        output_dir=output_dir,
    )

    assert plan.output_dir == output_dir
    assert plan.output_dir.is_dir()
    assert plan.targets[0].output_path == output_dir / "walk.mp4"


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan")])
def test_recording_attachment_rejects_invalid_fps_before_recording(
    tmp_path: Path,
    fps: float,
) -> None:
    recorder = RecordingAttachment(write_video=lambda *_, **__: None)

    with pytest.raises(ValueError, match="fps must be positive and finite"):
        recorder.start(tmp_path / "motion.mp4", fps=fps)

    with pytest.raises(RuntimeError, match="recording has not started"):
        recorder.capture(np.zeros((1, 1, 3), dtype=np.uint8))


def test_recording_attachment_rejects_invalid_output_targets_before_recording(
    tmp_path: Path,
) -> None:
    recorder = RecordingAttachment(write_video=lambda *_, **__: None)

    with pytest.raises(ValueError, match="must end with .mp4"):
        recorder.start(tmp_path / "motion.mov", fps=30.0)

    output_dir = tmp_path / "directory.mp4"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="is a directory"):
        recorder.start(output_dir, fps=30.0)


def test_deterministic_recorder_traverses_planned_targets_at_motion_fps(
    tmp_path: Path,
) -> None:
    motions = [
        SimpleNamespace(
            name="walk.npz",
            fps=24.0,
            root_pos=np.zeros((2, 3)),
        ),
        SimpleNamespace(
            name="jump.npz",
            fps=50.0,
            root_pos=np.zeros((3, 3)),
        ),
    ]

    class FakeScene:
        def __init__(self) -> None:
            self.applied: list[tuple[object, int]] = []

        def apply_reference_frame(self, motion: object, frame_index: int) -> None:
            self.applied.append((motion, frame_index))

    class FakeRenderer:
        def __init__(self) -> None:
            self.frames = iter(["walk-0", "walk-1", "jump-0", "jump-1", "jump-2"])
            self.closed = False

        def render_frame(self) -> str:
            return next(self.frames)

        def close(self) -> None:
            self.closed = True

    attachments: list[RecordingAttachment] = []
    writes: list[tuple[Path, list[str], float]] = []

    def make_attachment() -> RecordingAttachment:
        attachment = RecordingAttachment(
            write_video=lambda path, frames, *, fps: writes.append(
                (path, list(frames), fps)
            )
        )
        attachments.append(attachment)
        return attachment

    scene = FakeScene()
    renderer = FakeRenderer()
    request = RecordingRequest(
        targets=(
            RecordingRequestTarget(
                motion_index=0,
                source_path=tmp_path / "walk.npz",
                output_path=tmp_path / "videos" / "walk.mp4",
            ),
            RecordingRequestTarget(
                motion_index=1,
                source_path=tmp_path / "jump.npz",
                output_path=tmp_path / "videos" / "jump.mp4",
            ),
        )
    )

    results = DeterministicRecorder(
        scene=scene,
        renderer_factory=lambda: renderer,
        attachment_factory=make_attachment,
    ).record(motions, request)

    assert scene.applied == [
        (motions[0], 0),
        (motions[0], 1),
        (motions[1], 0),
        (motions[1], 1),
        (motions[1], 2),
    ]
    assert writes == [
        (tmp_path / "videos" / "walk.mp4", ["walk-0", "walk-1"], 24.0),
        (
            tmp_path / "videos" / "jump.mp4",
            ["jump-0", "jump-1", "jump-2"],
            50.0,
        ),
    ]
    assert [result.frame_count for result in results] == [2, 3]
    assert renderer.closed is True


def test_motion_viewer_records_an_already_planned_request() -> None:
    motions = [SimpleNamespace(root_pos=np.zeros((1, 3)))]
    request = RecordingRequest(
        targets=(
            RecordingRequestTarget(
                motion_index=0,
                source_path=Path("walk.npz"),
                output_path=Path("walk.mp4"),
            ),
        )
    )
    result = (
        SimpleNamespace(
            output_path=Path("walk.mp4"),
            frame_count=1,
            fps=30.0,
            written=True,
            reason=None,
        ),
    )

    class FakeRecorder:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def record(self, recorded_motions, recorded_request):
            self.calls.append((recorded_motions, recorded_request))
            return result

    recorder = FakeRecorder()
    viewer = MotionViewer(
        motions,
        SimpleNamespace(),
        deterministic_recorder=recorder,
    )

    assert viewer.record(request) == result
    assert recorder.calls == [(viewer.motions, request)]


def test_motion_viewer_recording_is_independent_of_interactive_playback_state(
    tmp_path: Path,
) -> None:
    motion = SimpleNamespace(
        name="walk.npz",
        fps=24.0,
        root_pos=np.zeros((3, 3)),
    )
    applied: list[int] = []

    class FakeScene:
        def apply_reference_frame(self, motion: object, frame_index: int) -> None:
            applied.append(frame_index)

    recorder = DeterministicRecorder(
        scene=FakeScene(),
        renderer_factory=lambda: SimpleNamespace(
            render_frame=lambda: "frame",
            close=lambda: None,
        ),
        attachment_factory=lambda: RecordingAttachment(
            write_video=lambda *_, **__: None
        ),
    )
    viewer = MotionViewer(
        [motion],
        SimpleNamespace(),
        deterministic_recorder=recorder,
    )
    viewer.controller.frame_position = 2.5
    viewer.controller.paused = True
    viewer.controller.increase_speed()

    viewer.record(
        RecordingRequest(
            targets=(
                RecordingRequestTarget(
                    motion_index=0,
                    source_path=tmp_path / "walk.npz",
                    output_path=tmp_path / "walk.mp4",
                ),
            )
        )
    )

    assert applied == [0, 1, 2]
    assert viewer.controller.frame_position == 2.5
    assert viewer.controller.paused is True
    assert viewer.controller.playback_speed == 1.25


@pytest.mark.parametrize(
    ("failure_stage", "message"),
    [
        ("apply", r"failed to apply frame 1/2 for motion 1/1 \(walk.npz\)"),
        ("render", r"failed to render frame 1/2 for motion 1/1 \(walk.npz\)"),
        ("write", r"failed to write recording for motion 1/1 \(walk.npz\)"),
    ],
)
def test_deterministic_recorder_reports_failure_context_and_removes_partial_output(
    tmp_path: Path,
    failure_stage: str,
    message: str,
) -> None:
    motion = SimpleNamespace(
        name="walk.npz",
        fps=30.0,
        root_pos=np.zeros((2, 3)),
    )
    output_path = tmp_path / "walk.mp4"

    class FakeScene:
        def apply_reference_frame(self, motion: object, frame_index: int) -> None:
            if failure_stage == "apply" and frame_index == 0:
                raise RuntimeError("bad frame")

    class FakeRenderer:
        def render_frame(self) -> str:
            if failure_stage == "render":
                raise RuntimeError("render unavailable")
            return "frame"

        def close(self) -> None:
            pass

    def write_video(path: Path, frames: list[str], *, fps: float) -> None:
        if failure_stage == "write":
            path.write_bytes(b"partial")
            raise RuntimeError("encoder failed")

    recorder = DeterministicRecorder(
        scene=FakeScene(),
        renderer_factory=FakeRenderer,
        attachment_factory=lambda: RecordingAttachment(write_video=write_video),
    )
    request = RecordingRequest(
        targets=(
            RecordingRequestTarget(
                motion_index=0,
                source_path=tmp_path / "walk.npz",
                output_path=output_path,
            ),
        )
    )

    with pytest.raises(RuntimeError, match=message) as error:
        recorder.record([motion], request)

    if failure_stage == "write":
        assert str(error.value) == (
            f"failed to write recording for motion 1/1 (walk.npz) to {output_path}"
        )
    assert not output_path.exists()


def test_mjlab_recorder_constructs_public_offscreen_renderer_only_when_recording(
    tmp_path: Path,
) -> None:
    calls = []

    class FakeViewerConfig:
        class OriginType:
            ASSET_ROOT = "asset-root"

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeOffscreenRenderer:
        def __init__(self, *, model, cfg, scene, sim_model=None) -> None:
            calls.append(("init", model, cfg, scene, sim_model))

        def initialize(self) -> None:
            calls.append(("initialize",))

        def update(self, data) -> None:
            calls.append(("update", data))

        def render(self) -> str:
            calls.append(("render",))
            return "rendered-frame"

        def close(self) -> None:
            calls.append(("close",))

    class FakeScene:
        mj_model = "mj-model"
        scene = "scene"
        sim = SimpleNamespace(model="sim-model", data="sim-data")

        def apply_reference_frame(self, motion: object, frame_index: int) -> None:
            calls.append(("apply", frame_index))

    scene = FakeScene()
    recorder = create_mjlab_deterministic_recorder(
        scene,
        offscreen_renderer_cls=FakeOffscreenRenderer,
        viewer_config_cls=FakeViewerConfig,
        attachment_factory=lambda: RecordingAttachment(
            write_video=lambda *_, **__: None
        ),
    )

    assert calls == []

    recorder.record(
        [SimpleNamespace(name="walk.npz", fps=30.0, root_pos=np.zeros((1, 3)))],
        RecordingRequest(
            targets=(
                RecordingRequestTarget(
                    motion_index=0,
                    source_path=tmp_path / "walk.npz",
                    output_path=tmp_path / "walk.mp4",
                ),
            )
        ),
    )

    cfg = calls[0][2]
    assert cfg.kwargs == {
        "height": 480,
        "width": 640,
        "origin_type": "asset-root",
        "entity_name": "robot",
        "distance": 2.0,
        "elevation": -5.0,
        "azimuth": 20.0,
    }
    assert calls == [
        ("init", "mj-model", cfg, "scene", "sim-model"),
        ("initialize",),
        ("apply", 0),
        ("update", "sim-data"),
        ("render",),
        ("close",),
    ]
