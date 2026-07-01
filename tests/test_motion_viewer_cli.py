from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
MOTION_VIEWER_PATH = (
    ROOT
    / "src"
    / "mjlab_playground"
    / "motion_lib"
    / "tools"
    / "motion_viewer.py"
)


def _load_motion_viewer_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("motion_viewer", MOTION_VIEWER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parse_args = _load_motion_viewer_module().parse_args


def test_motion_viewer_cli_defaults_to_pyroki_format_and_astro_30fps() -> None:
    args = parse_args(["--motion-files", "/tmp/motion.npz"])

    assert args.motion_files == Path("/tmp/motion.npz")
    assert args.motion_format == "pyroki"
    assert args.fps == 30.0
    assert args.robot == "astro"


def test_motion_viewer_cli_accepts_proto_format() -> None:
    args = parse_args(
        [
            "--motion-files",
            "/tmp/motion.motion",
            "--format",
            "proto",
            "--fps",
            "50",
            "--robot",
            "g1",
        ]
    )

    assert args.motion_format == "proto"
    assert args.fps == 50.0
    assert args.robot == "g1"
