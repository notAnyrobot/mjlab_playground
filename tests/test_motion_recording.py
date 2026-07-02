from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mjlab_playground.motion_lib.recording import (
    RecordingAttachment,
    plan_recording_outputs,
)


def test_recording_attachment_writes_captured_frames_when_stopped(tmp_path: Path) -> None:
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
        SimpleNamespace(write_video=lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    output_path = tmp_path / "video.mp4"

    recorder = RecordingAttachment()
    recorder.start(output_path, fps=60.0)
    recorder.capture(frame)
    result = recorder.stop()

    assert result.written is True
    assert calls == [((output_path, [frame]), {"fps": 60.0})]


def test_recording_attachment_stop_with_no_frames_returns_no_output(tmp_path: Path) -> None:
    calls = []
    output_path = tmp_path / "videos" / "run" / "empty.mp4"
    recorder = RecordingAttachment(write_video=lambda *args, **kwargs: calls.append((args, kwargs)))

    recorder.start(output_path, fps=24.0)
    result = recorder.stop()

    assert result.written is False
    assert result.output_path == output_path
    assert result.frame_count == 0
    assert result.fps == 24.0
    assert result.reason == "no frames captured"
    assert calls == []
    assert not output_path.exists()


def test_plan_recording_outputs_uses_one_timestamped_run_directory(tmp_path: Path) -> None:
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


def test_plan_recording_outputs_uses_loader_selected_motion_names(tmp_path: Path) -> None:
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
        (target.source_path.name, target.output_path.name)
        for target in plan.targets
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
