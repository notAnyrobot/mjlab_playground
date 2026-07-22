from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN_MOTION_LIB_PATH = (
    ROOT / "src" / "mjlab_playground" / "motion_lib" / "scripts" / "run_motion_lib.py"
)
RUN_MOTION_LIB_MODULE = "mjlab_playground.motion_lib.scripts.run_motion_lib"


def _load_run_motion_lib_module():
    spec = importlib.util.spec_from_file_location("run_motion_lib", RUN_MOTION_LIB_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_motion_lib_cli_requires_one_output_file() -> None:
    run_motion_lib = _load_run_motion_lib_module()

    args = run_motion_lib.parse_args(
        [
            "--motion-files",
            "/tmp/pyroki-retargeted-astro",
            "--format",
            "pyroki",
            "--source-fps",
            "30",
            "--output-fps",
            "50",
            "--robot",
            "astro",
            "--device",
            "cpu",
            "--output-file",
            "/tmp/mjlab-astro.npz",
        ]
    )

    assert args.motion_files == Path("/tmp/pyroki-retargeted-astro")
    assert args.motion_format == "pyroki"
    assert args.source_fps == 30.0
    assert args.output_fps == 50.0
    assert args.robot == "astro"
    assert args.device == "cpu"
    assert args.output_file == Path("/tmp/mjlab-astro.npz")
    assert args.overwrite is False

    overwrite_args = run_motion_lib.parse_args(
        [
            "--motion-files",
            "/tmp/motion.npz",
            "--output-file",
            "/tmp/mjlab-astro.npz",
            "--overwrite",
        ]
    )
    assert overwrite_args.overwrite is True

    with pytest.raises(SystemExit) as missing_output_file:
        run_motion_lib.parse_args(["--motion-files", "/tmp/motion.npz"])
    assert missing_output_file.value.code == 2

    with pytest.raises(SystemExit) as removed_output_dir:
        run_motion_lib.parse_args(
            [
                "--motion-files",
                "/tmp/motion.npz",
                "--output-file",
                "/tmp/mjlab-astro.npz",
                "--output-dir",
                "/tmp/mjlab-astro",
            ]
        )
    assert removed_output_dir.value.code == 2


def test_run_motion_lib_rejects_unsupported_robot_and_format() -> None:
    run_motion_lib = _load_run_motion_lib_module()

    with pytest.raises(SystemExit) as robot_exit:
        run_motion_lib.parse_args(
            ["--motion-files", "/tmp/motion.npz", "--robot", "g1"]
        )
    assert robot_exit.value.code == 2

    with pytest.raises(SystemExit) as format_exit:
        run_motion_lib.parse_args(
            ["--motion-files", "/tmp/motion.npz", "--format", "gmr"]
        )
    assert format_exit.value.code == 2


def _rich_reference_motion(name: str, offset: float):
    from mjlab_playground.motion_lib import ReferenceMotion

    frame_values = torch.arange(3, dtype=torch.float32) + offset
    return ReferenceMotion(
        name=name,
        fps=50.0,
        root_pos=frame_values[:, None].repeat(1, 3),
        root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        dof_pos=frame_values[:, None].repeat(1, 2),
        root_lin_vel=(frame_values + 1.0)[:, None].repeat(1, 3),
        root_ang_vel=(frame_values + 2.0)[:, None].repeat(1, 3),
        dof_vel=(frame_values + 3.0)[:, None].repeat(1, 2),
        body_pos=(frame_values + 4.0)[:, None, None].repeat(1, 2, 3),
        body_rot=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(3, 2, 1),
        body_lin_vel=(frame_values + 5.0)[:, None, None].repeat(1, 2, 3),
        body_ang_vel=(frame_values + 6.0)[:, None, None].repeat(1, 2, 3),
        body_contacts=torch.tensor(
            [[True, False], [False, True], [True, True]], dtype=torch.bool
        ),
        dof_names=("left_hip", "right_hip"),
        body_names=("pelvis", "torso"),
    )


def test_run_motion_lib_publishes_one_ordered_multi_clip_artifact(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_file = tmp_path / "assembled.npz"
    events = []
    configs = []
    rich_motions = {
        "walk": _rich_reference_motion("walk.npz", 0.0),
        "turn": _rich_reference_motion("turn.npz", 10.0),
    }
    writer_inputs = []

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            configs.append(cfg)
            events.append(("init",))

        def load(self, motion_files: Path):
            events.append(("load", motion_files))
            return ["walk", "turn"]

        def resample(self, motion):
            events.append(("resample", motion))
            return motion

        def enrich(self, motion):
            events.append(("enrich", motion))
            return rich_motions[motion]

    class RecordingWriter:
        def write(self, motion, output_path: Path, *, overwrite: bool) -> None:
            events.append(("write", output_path, overwrite))
            writer_inputs.append(motion)
            ReferenceMotionNpzWriter().write(
                motion,
                output_path,
                overwrite=overwrite,
            )

    def fake_cfg_cls(**kwargs):
        return SimpleNamespace(**kwargs)

    result = run_motion_lib.run_pipeline(
        source_dir,
        motion_format="pyroki",
        source_fps=30.0,
        output_fps=50.0,
        robot="astro",
        device="cpu",
        output_file=output_file,
        motion_lib_cls=FakeMotionLib,
        motion_lib_cfg_cls=fake_cfg_cls,
        writer_cls=RecordingWriter,
    )

    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.source_format == "pyroki"
    assert cfg.source_fps == 30.0
    assert cfg.output_fps == 50.0
    assert cfg.robot == "astro"
    assert cfg.device == "cpu"
    assert events == [
        ("init",),
        ("load", source_dir),
        ("resample", "walk"),
        ("enrich", "walk"),
        ("resample", "turn"),
        ("enrich", "turn"),
        ("write", output_file, False),
    ]
    assert list(vars(result)) == ["assembled_motion", "written_path"]
    assert writer_inputs == [result.assembled_motion]
    assert result.written_path == output_file
    assert list(tmp_path.rglob("*.npz")) == [output_file]

    loaded_motions = MotionLoader.load(output_file, motion_format="mjlab")
    assert len(loaded_motions) == 1
    loaded = loaded_motions[0]
    assert loaded.fps == 50.0
    assert loaded.dof_names == ("left_hip", "right_hip")
    assert loaded.body_names == ("pelvis", "torso")
    assert [span.name for span in loaded.iter_clip_spans()] == [
        "walk.npz",
        "turn.npz",
    ]
    torch.testing.assert_close(loaded.clip_starts, torch.tensor([0, 3]))
    torch.testing.assert_close(loaded.clip_lengths, torch.tensor([3, 3]))
    torch.testing.assert_close(loaded.clip_fps, torch.tensor([50.0, 50.0]))
    for field_name in (
        "root_pos",
        "root_rot",
        "dof_pos",
        "root_lin_vel",
        "root_ang_vel",
        "dof_vel",
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
        "body_contacts",
    ):
        expected = torch.cat(
            [getattr(rich_motions[name], field_name) for name in ("walk", "turn")]
        )
        torch.testing.assert_close(getattr(loaded, field_name), expected)


def test_run_motion_lib_publishes_one_explicit_single_clip_artifact(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    source_file = tmp_path / "walk-source.npz"
    output_file = tmp_path / "walk-reference.npz"
    source_file.touch()
    rich_motion = _rich_reference_motion("walk.npz", 0.0)

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            del cfg

        def load(self, motion_files: Path):
            assert motion_files == source_file
            return ["walk"]

        def resample(self, motion):
            assert motion == "walk"
            return motion

        def enrich(self, motion):
            assert motion == "walk"
            return rich_motion

    result = run_motion_lib.run_pipeline(
        source_file,
        output_file=output_file,
        motion_lib_cls=FakeMotionLib,
        motion_lib_cfg_cls=lambda **kwargs: SimpleNamespace(**kwargs),
        writer_cls=ReferenceMotionNpzWriter,
    )

    loaded = MotionLoader.load(output_file, motion_format="mjlab")[0]
    assert result.written_path == output_file
    assert [span.name for span in loaded.iter_clip_spans()] == ["walk.npz"]
    torch.testing.assert_close(loaded.clip_starts, torch.tensor([0]))
    torch.testing.assert_close(loaded.clip_lengths, torch.tensor([3]))
    torch.testing.assert_close(loaded.clip_fps, torch.tensor([50.0]))
    torch.testing.assert_close(loaded.root_pos, rich_motion.root_pos)


def test_run_motion_lib_does_not_guess_pyroki_child_from_split_root(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    split_root = tmp_path / "sfu"
    source_dir = split_root / "pyroki-retargeted-astro"
    source_dir.mkdir(parents=True)
    (source_dir / "walk_retargeted.npz").touch()
    output_file = tmp_path / "published" / "assembled.npz"
    load_error = ValueError(f"No direct child .npz files found in {split_root}")

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            pass

        def load(self, motion_files: Path):
            assert motion_files == split_root
            raise load_error

        def resample(self, motion):
            raise AssertionError("resample should not run after a load failure")

        def enrich(self, motion):
            raise AssertionError("enrich should not run after a load failure")

    class FakeWriter:
        def write(self, motion, output_path: Path) -> None:
            raise AssertionError("write should not run after a load failure")

    def fake_cfg_cls(**kwargs):
        return SimpleNamespace(**kwargs)

    with pytest.raises(RuntimeError) as failure:
        run_motion_lib.run_pipeline(
            split_root,
            output_file=output_file,
            motion_format="pyroki",
            motion_lib_cls=FakeMotionLib,
            motion_lib_cfg_cls=fake_cfg_cls,
            writer_cls=FakeWriter,
        )

    assert "load" in str(failure.value)
    assert str(split_root) in str(failure.value)
    assert "No direct child .npz files found" in str(failure.value)
    assert failure.value.__cause__ is load_error
    assert not output_file.exists()
    assert not output_file.parent.exists()


def test_run_motion_lib_resample_failure_identifies_ordered_clip_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    source_file = tmp_path / "source"
    output_file = tmp_path / "published" / "assembled.npz"
    source_motions = [
        SimpleNamespace(name="walk.npz"),
        SimpleNamespace(name="turn.npz"),
    ]
    resample_error = ValueError("invalid frame timing")

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            del cfg

        def load(self, motion_files: Path):
            assert motion_files == source_file
            return source_motions

        def resample(self, motion):
            if motion is source_motions[1]:
                raise resample_error
            return motion

        def enrich(self, motion):
            assert motion is source_motions[0]
            return _rich_reference_motion("walk.npz", 0.0)

    class FakeWriter:
        def write(self, motion, output_path: Path, *, overwrite: bool) -> None:
            del motion, output_path, overwrite
            raise AssertionError("write should not run after a resample failure")

    with pytest.raises(RuntimeError) as failure:
        run_motion_lib.run_pipeline(
            source_file,
            output_file=output_file,
            motion_lib_cls=FakeMotionLib,
            motion_lib_cfg_cls=lambda **kwargs: SimpleNamespace(**kwargs),
            writer_cls=FakeWriter,
        )

    assert "resample" in str(failure.value)
    assert "clip 1" in str(failure.value)
    assert "turn.npz" in str(failure.value)
    assert failure.value.__cause__ is resample_error
    assert not output_file.exists()
    assert not output_file.parent.exists()


def test_run_motion_lib_enrich_failure_identifies_ordered_clip_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    source_file = tmp_path / "source"
    output_file = tmp_path / "published" / "assembled.npz"
    source_motions = [
        SimpleNamespace(name="walk.npz"),
        SimpleNamespace(name="turn.npz"),
    ]
    enrich_error = RuntimeError("simulator state unavailable")

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            del cfg

        def load(self, motion_files: Path):
            assert motion_files == source_file
            return source_motions

        def resample(self, motion):
            return motion

        def enrich(self, motion):
            if motion is source_motions[1]:
                raise enrich_error
            return _rich_reference_motion("walk.npz", 0.0)

    class FakeWriter:
        def write(self, motion, output_path: Path, *, overwrite: bool) -> None:
            del motion, output_path, overwrite
            raise AssertionError("write should not run after an enrich failure")

    with pytest.raises(RuntimeError) as failure:
        run_motion_lib.run_pipeline(
            source_file,
            output_file=output_file,
            motion_lib_cls=FakeMotionLib,
            motion_lib_cfg_cls=lambda **kwargs: SimpleNamespace(**kwargs),
            writer_cls=FakeWriter,
        )

    assert "enrich" in str(failure.value)
    assert "clip 1" in str(failure.value)
    assert "turn.npz" in str(failure.value)
    assert failure.value.__cause__ is enrich_error
    assert not output_file.exists()
    assert not output_file.parent.exists()


def test_run_motion_lib_assembly_failure_identifies_loader_order_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    output_file = tmp_path / "published" / "assembled.npz"
    rich_motions = [
        _rich_reference_motion("walk.npz", 0.0),
        replace(
            _rich_reference_motion("turn.npz", 10.0),
            dof_names=("right_hip", "left_hip"),
        ),
    ]

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            del cfg

        def load(self, motion_files: Path):
            del motion_files
            return rich_motions

        def resample(self, motion):
            return motion

        def enrich(self, motion):
            return motion

    class FakeWriter:
        def write(self, motion, output_path: Path, *, overwrite: bool) -> None:
            del motion, output_path, overwrite
            raise AssertionError("write should not run after an assembly failure")

    with pytest.raises(RuntimeError) as failure:
        run_motion_lib.run_pipeline(
            tmp_path / "source",
            output_file=output_file,
            motion_lib_cls=FakeMotionLib,
            motion_lib_cfg_cls=lambda **kwargs: SimpleNamespace(**kwargs),
            writer_cls=FakeWriter,
        )

    assert "assembly" in str(failure.value)
    assert "0: walk.npz" in str(failure.value)
    assert "1: turn.npz" in str(failure.value)
    assert isinstance(failure.value.__cause__, ValueError)
    assert "dof_names" in str(failure.value.__cause__)
    assert not output_file.exists()
    assert not output_file.parent.exists()


def test_run_motion_lib_protects_existing_destination_by_default(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_file = tmp_path / "published" / "assembled.npz"
    ReferenceMotionNpzWriter().write(
        _rich_reference_motion("original.npz", 0.0),
        output_file,
    )
    original_contents = output_file.read_bytes()
    replacement = _rich_reference_motion("replacement.npz", 20.0)

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            del cfg

        def load(self, motion_files: Path):
            del motion_files
            return [replacement]

        def resample(self, motion):
            return motion

        def enrich(self, motion):
            return motion

    with pytest.raises(RuntimeError) as failure:
        run_motion_lib.run_pipeline(
            tmp_path / "source",
            output_file=output_file,
            motion_lib_cls=FakeMotionLib,
            motion_lib_cfg_cls=lambda **kwargs: SimpleNamespace(**kwargs),
            writer_cls=ReferenceMotionNpzWriter,
        )

    assert "write" in str(failure.value)
    assert str(output_file) in str(failure.value)
    assert isinstance(failure.value.__cause__, FileExistsError)
    assert output_file.read_bytes() == original_contents
    assert set(output_file.parent.iterdir()) == {output_file}


def test_run_motion_lib_atomically_replaces_destination_when_overwrite_is_explicit(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_file = tmp_path / "published" / "assembled.npz"
    writer = ReferenceMotionNpzWriter()
    writer.write(_rich_reference_motion("original.npz", 0.0), output_file)
    replacement = _rich_reference_motion("replacement.npz", 20.0)

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            del cfg

        def load(self, motion_files: Path):
            del motion_files
            return [replacement]

        def resample(self, motion):
            return motion

        def enrich(self, motion):
            return motion

    result = run_motion_lib.run_pipeline(
        tmp_path / "source",
        output_file=output_file,
        overwrite=True,
        motion_lib_cls=FakeMotionLib,
        motion_lib_cfg_cls=lambda **kwargs: SimpleNamespace(**kwargs),
        writer_cls=ReferenceMotionNpzWriter,
    )

    loaded = MotionLoader.load(output_file, motion_format="mjlab")[0]
    assert result.written_path == output_file
    assert [span.name for span in loaded.iter_clip_spans()] == ["replacement.npz"]
    torch.testing.assert_close(loaded.root_pos, replacement.root_pos)
    assert set(output_file.parent.iterdir()) == {output_file}


def test_run_motion_lib_failed_write_leaves_no_artifact_or_intermediate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_file = tmp_path / "published" / "assembled.npz"
    rich_motion = _rich_reference_motion("walk.npz", 0.0)
    write_error = OSError("simulated serialization failure")

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            del cfg

        def load(self, motion_files: Path):
            del motion_files
            return [rich_motion]

        def resample(self, motion):
            return motion

        def enrich(self, motion):
            return motion

    def fail_after_partial_write(target, *args, **kwargs) -> None:
        del args, kwargs
        target.write(b"partial artifact")
        raise write_error

    monkeypatch.setattr(
        "mjlab_playground.motion_lib.reference_motion_npz_writer.np.savez",
        fail_after_partial_write,
    )

    with pytest.raises(RuntimeError) as failure:
        run_motion_lib.run_pipeline(
            tmp_path / "source",
            output_file=output_file,
            motion_lib_cls=FakeMotionLib,
            motion_lib_cfg_cls=lambda **kwargs: SimpleNamespace(**kwargs),
            writer_cls=ReferenceMotionNpzWriter,
        )

    assert "write" in str(failure.value)
    assert str(output_file) in str(failure.value)
    assert failure.value.__cause__ is write_error
    assert not output_file.exists()
    assert list(output_file.parent.iterdir()) == []


def test_run_motion_lib_failed_overwrite_preserves_valid_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_file = tmp_path / "published" / "assembled.npz"
    writer = ReferenceMotionNpzWriter()
    writer.write(_rich_reference_motion("original.npz", 0.0), output_file)
    original_contents = output_file.read_bytes()
    replacement = _rich_reference_motion("replacement.npz", 20.0)
    write_error = OSError("simulated replacement failure")

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            del cfg

        def load(self, motion_files: Path):
            del motion_files
            return [replacement]

        def resample(self, motion):
            return motion

        def enrich(self, motion):
            return motion

    def fail_after_partial_write(target, *args, **kwargs) -> None:
        del args, kwargs
        target.write(b"partial replacement")
        raise write_error

    monkeypatch.setattr(
        "mjlab_playground.motion_lib.reference_motion_npz_writer.np.savez",
        fail_after_partial_write,
    )

    with pytest.raises(RuntimeError) as failure:
        run_motion_lib.run_pipeline(
            tmp_path / "source",
            output_file=output_file,
            overwrite=True,
            motion_lib_cls=FakeMotionLib,
            motion_lib_cfg_cls=lambda **kwargs: SimpleNamespace(**kwargs),
            writer_cls=ReferenceMotionNpzWriter,
        )

    assert "write" in str(failure.value)
    assert str(output_file) in str(failure.value)
    assert failure.value.__cause__ is write_error
    assert output_file.read_bytes() == original_contents
    assert set(output_file.parent.iterdir()) == {output_file}


def test_run_motion_lib_main_returns_result_and_reports_one_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    output_file = tmp_path / "rich-motion.npz"
    written = []

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            pass

        def load(self, motion_files: Path):
            return ["source-motion"]

        def resample(self, motion):
            return "resampled-motion"

        def enrich(self, motion):
            return _rich_reference_motion("rich-motion.npz", 0.0)

    class FakeWriter:
        def write(self, motion, output_path: Path, *, overwrite: bool) -> None:
            written.append((motion, output_path, overwrite))

    def fake_cfg_cls(**kwargs):
        return SimpleNamespace(**kwargs)

    result = run_motion_lib.main(
        [
            "--motion-files",
            str(tmp_path / "motion.npz"),
            "--source-fps",
            "30",
            "--output-fps",
            "50",
            "--output-file",
            str(output_file),
            "--overwrite",
        ],
        motion_lib_cls=FakeMotionLib,
        motion_lib_cfg_cls=fake_cfg_cls,
        writer_cls=FakeWriter,
    )

    assert written == [(result.assembled_motion, output_file, True)]
    assert result.written_path == output_file
    assert capsys.readouterr().out == f"wrote {output_file}\n"


def test_run_motion_lib_help_avoids_task_registration_imports() -> None:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            RUN_MOTION_LIB_MODULE,
            "--help",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--motion-files" in result.stdout
    assert "--source-fps" in result.stdout
    assert "--output-fps" in result.stdout
    assert "--visualize" not in result.stdout
    assert "--save" not in result.stdout
    assert "--output-file" in result.stdout
    assert "--overwrite" in result.stdout
    assert "--output-dir" not in result.stdout
    assert "rsl_rl" not in result.stderr
