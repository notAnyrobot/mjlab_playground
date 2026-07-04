from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_MOTION_LIB_PATH = (
    ROOT
    / "src"
    / "mjlab_playground"
    / "motion_lib"
    / "scripts"
    / "run_motion_lib.py"
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


def test_run_motion_lib_cli_parses_pipeline_options() -> None:
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
            "--output-dir",
            "/tmp/mjlab-astro",
        ]
    )

    assert args.motion_files == Path("/tmp/pyroki-retargeted-astro")
    assert args.motion_format == "pyroki"
    assert args.source_fps == 30.0
    assert args.output_fps == 50.0
    assert args.robot == "astro"
    assert args.device == "cpu"
    assert args.output_dir == Path("/tmp/mjlab-astro")


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


def test_run_motion_lib_pipeline_calls_load_resample_enrich_in_order(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    motion_path = tmp_path / "motion.npz"
    calls = []

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            calls.append(("init", cfg))

        def load(self, motion_files: Path):
            calls.append(("load", motion_files))
            return ["source-motion"]

        def resample(self, motions):
            calls.append(("resample", motions))
            return ["resampled-motion"]

        def enrich(self, motions):
            calls.append(("enrich", motions))
            return [SimpleNamespace(name="rich-motion.npz")]

    class FakeWriter:
        def write(self, motion, output_path: Path) -> None:
            calls.append(("write", motion.name, output_path))

    def fake_cfg_cls(**kwargs):
        return SimpleNamespace(**kwargs)

    result = run_motion_lib.run_pipeline(
        motion_path,
        motion_format="pyroki",
        source_fps=30.0,
        output_fps=50.0,
        robot="astro",
        device="cpu",
        motion_lib_cls=FakeMotionLib,
        motion_lib_cfg_cls=fake_cfg_cls,
        writer_cls=FakeWriter,
    )

    cfg = calls[0][1]
    assert cfg.source_format == "pyroki"
    assert cfg.source_fps == 30.0
    assert cfg.output_fps == 50.0
    assert cfg.robot == "astro"
    assert cfg.device == "cpu"
    assert calls[1:] == [
        ("load", motion_path),
        ("resample", ["source-motion"]),
        ("enrich", ["resampled-motion"]),
        (
            "write",
            "rich-motion.npz",
            tmp_path.parent / "mjlab-astro" / "rich-motion.npz",
        ),
    ]
    assert result.source_motions == ["source-motion"]
    assert result.resampled_motions == ["resampled-motion"]
    assert result.rich_motions == [SimpleNamespace(name="rich-motion.npz")]
    assert result.written_paths == [
        tmp_path.parent / "mjlab-astro" / "rich-motion.npz"
    ]


def test_run_motion_lib_exports_rich_motions_to_explicit_output_dir(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    motion_path = tmp_path / "pyroki-retargeted-astro"
    output_dir = tmp_path / "mjlab-astro"
    calls = []
    rich_motions = [
        SimpleNamespace(name="walk_retargeted.npz", display_name="walk"),
        SimpleNamespace(name="turn_retargeted", display_name="turn"),
    ]

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            calls.append(("init", cfg))

        def load(self, motion_files: Path):
            calls.append(("load", motion_files))
            return ["source-motion"]

        def resample(self, motions):
            calls.append(("resample", motions))
            return ["resampled-motion"]

        def enrich(self, motions):
            calls.append(("enrich", motions))
            return rich_motions

    class FakeWriter:
        def write(self, motion, output_path: Path) -> None:
            calls.append(("write", motion.name, output_path))

    def fake_cfg_cls(**kwargs):
        return SimpleNamespace(**kwargs)

    result = run_motion_lib.run_pipeline(
        motion_path,
        output_dir=output_dir,
        motion_lib_cls=FakeMotionLib,
        motion_lib_cfg_cls=fake_cfg_cls,
        writer_cls=FakeWriter,
    )

    assert calls[1:] == [
        ("load", motion_path),
        ("resample", ["source-motion"]),
        ("enrich", ["resampled-motion"]),
        ("write", "walk_retargeted.npz", output_dir / "walk_retargeted.npz"),
        ("write", "turn_retargeted", output_dir / "turn_retargeted.npz"),
    ]
    assert result.written_paths == [
        output_dir / "walk_retargeted.npz",
        output_dir / "turn_retargeted.npz",
    ]


def test_run_motion_lib_defaults_output_dir_next_to_source_directory(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    motion_dir = tmp_path / "pyroki-retargeted-astro"
    motion_dir.mkdir()
    calls = []

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            pass

        def load(self, motion_files: Path):
            return ["source-motion"]

        def resample(self, motions):
            return ["resampled-motion"]

        def enrich(self, motions):
            return [SimpleNamespace(name="walk_retargeted.npz")]

    class FakeWriter:
        def write(self, motion, output_path: Path) -> None:
            calls.append(output_path)

    def fake_cfg_cls(**kwargs):
        return SimpleNamespace(**kwargs)

    result = run_motion_lib.run_pipeline(
        motion_dir,
        motion_lib_cls=FakeMotionLib,
        motion_lib_cfg_cls=fake_cfg_cls,
        writer_cls=FakeWriter,
    )

    assert calls == [tmp_path / "mjlab-astro" / "walk_retargeted.npz"]
    assert result.written_paths == calls


def test_run_motion_lib_does_not_guess_pyroki_child_from_split_root(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    split_root = tmp_path / "sfu"
    source_dir = split_root / "pyroki-retargeted-astro"
    source_dir.mkdir(parents=True)
    (source_dir / "walk_retargeted.npz").touch()

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            pass

        def load(self, motion_files: Path):
            assert motion_files == split_root
            raise ValueError(f"No direct child .npz files found in {motion_files}")

        def resample(self, motions):
            raise AssertionError("resample should not run after a load failure")

        def enrich(self, motions):
            raise AssertionError("enrich should not run after a load failure")

    class FakeWriter:
        def write(self, motion, output_path: Path) -> None:
            raise AssertionError("write should not run after a load failure")

    def fake_cfg_cls(**kwargs):
        return SimpleNamespace(**kwargs)

    with pytest.raises(ValueError, match=r"No direct child \.npz files found"):
        run_motion_lib.run_pipeline(
            split_root,
            motion_format="pyroki",
            motion_lib_cls=FakeMotionLib,
            motion_lib_cfg_cls=fake_cfg_cls,
            writer_cls=FakeWriter,
        )


def test_run_motion_lib_export_failure_names_offending_motion(
    tmp_path: Path,
) -> None:
    run_motion_lib = _load_run_motion_lib_module()
    motion_path = tmp_path / "motion.npz"

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            pass

        def load(self, motion_files: Path):
            return ["source-motion"]

        def resample(self, motions):
            return ["resampled-motion"]

        def enrich(self, motions):
            return [SimpleNamespace(name="bad_walk.npz")]

    class FailingWriter:
        def write(self, motion, output_path: Path) -> None:
            raise OSError("disk full")

    def fake_cfg_cls(**kwargs):
        return SimpleNamespace(**kwargs)

    with pytest.raises(RuntimeError, match="bad_walk\\.npz.*disk full"):
        run_motion_lib.run_pipeline(
            motion_path,
            motion_lib_cls=FakeMotionLib,
            motion_lib_cfg_cls=fake_cfg_cls,
            writer_cls=FailingWriter,
        )


def test_run_motion_lib_main_returns_pipeline_result(tmp_path: Path) -> None:
    run_motion_lib = _load_run_motion_lib_module()

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            pass

        def load(self, motion_files: Path):
            return ["source-motion"]

        def resample(self, motions):
            return ["resampled-motion"]

        def enrich(self, motions):
            return [SimpleNamespace(name="rich-motion.npz")]

    class FakeWriter:
        def write(self, motion, output_path: Path) -> None:
            pass

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
        ],
        motion_lib_cls=FakeMotionLib,
        motion_lib_cfg_cls=fake_cfg_cls,
        writer_cls=FakeWriter,
    )

    assert result.source_motions == ["source-motion"]
    assert result.resampled_motions == ["resampled-motion"]
    assert result.rich_motions == [SimpleNamespace(name="rich-motion.npz")]
    assert result.written_paths == [
        tmp_path.parent / "mjlab-astro" / "rich-motion.npz"
    ]


def test_run_motion_lib_main_reports_written_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_motion_lib = _load_run_motion_lib_module()

    class FakeMotionLib:
        def __init__(self, cfg) -> None:
            pass

        def load(self, motion_files: Path):
            return ["source-motion"]

        def resample(self, motions):
            return ["resampled-motion"]

        def enrich(self, motions):
            return [
                SimpleNamespace(name="walk_retargeted.npz"),
                SimpleNamespace(name="turn_retargeted.npz"),
            ]

    class FakeWriter:
        def write(self, motion, output_path: Path) -> None:
            pass

    def fake_cfg_cls(**kwargs):
        return SimpleNamespace(**kwargs)

    run_motion_lib.main(
        [
            "--motion-files",
            str(tmp_path / "pyroki-retargeted-astro"),
            "--output-dir",
            str(tmp_path / "mjlab-astro"),
        ],
        motion_lib_cls=FakeMotionLib,
        motion_lib_cfg_cls=fake_cfg_cls,
        writer_cls=FakeWriter,
    )

    assert capsys.readouterr().out.splitlines() == [
        f"wrote {tmp_path / 'mjlab-astro' / 'walk_retargeted.npz'}",
        f"wrote {tmp_path / 'mjlab-astro' / 'turn_retargeted.npz'}",
    ]


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
    assert "--output-dir" in result.stdout
    assert "rsl_rl" not in result.stderr
