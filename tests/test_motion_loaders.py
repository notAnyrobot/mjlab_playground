from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MOTION_LOADER_PATH = (
    ROOT / "src" / "mjlab_playground" / "motion_lib" / "motion_loader.py"
)


def _load_motion_loader_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("motion_loader", MOTION_LOADER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_motion_loader = _load_motion_loader_module()
MotionLoader = _motion_loader.MotionLoader
PyrokiMotionLoader = _motion_loader.PyrokiMotionLoader


def _write_pyroki_npz(path: Path, *, frames: int = 2) -> None:
    np.savez(
        path,
        base_frame_pos=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float64,
        )[:frames],
        base_frame_wxyz=np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )[:frames],
        joint_angles=np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dtype=np.float64,
        )[:frames],
    )


def test_pyroki_motion_loader_loads_single_npz_into_reference_motion_state(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_pyroki_npz(motion_path)

    motions = PyrokiMotionLoader(motion_path, fps=50.0).load_motion()

    assert len(motions) == 1
    motion = motions[0]
    assert motion.name == "motion.npz"
    assert motion.fps == 50.0
    assert motion.root_pos.dtype == torch.float32
    assert motion.root_rot.dtype == torch.float32
    assert motion.dof_pos.dtype == torch.float32
    torch.testing.assert_close(
        motion.root_pos,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    torch.testing.assert_close(
        motion.root_rot,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        motion.dof_pos,
        torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    )


def test_motion_loader_is_abstract() -> None:
    with pytest.raises(TypeError):
        MotionLoader()


def test_pyroki_motion_loader_loads_flat_directory_in_sorted_order(
    tmp_path: Path,
) -> None:
    motion_dir = tmp_path / "pyroki"
    nested_dir = motion_dir / "nested"
    nested_dir.mkdir(parents=True)
    _write_pyroki_npz(motion_dir / "b_motion.npz")
    _write_pyroki_npz(motion_dir / "a_motion.npz")
    _write_pyroki_npz(nested_dir / "ignored.npz")

    motions = PyrokiMotionLoader(motion_dir).load_motion()

    assert len(motions) == 2
    assert [motion.root_pos[0, 0].item() for motion in motions] == [1.0, 1.0]


def test_pyroki_motion_loader_fails_fast_with_bad_clip_path(tmp_path: Path) -> None:
    motion_dir = tmp_path / "pyroki"
    motion_dir.mkdir()
    _write_pyroki_npz(motion_dir / "good.npz")
    np.savez(motion_dir / "bad.npz", base_frame_pos=np.zeros((1, 3)))

    with pytest.raises(ValueError, match=r"bad\.npz: Missing required PyRoki key"):
        PyrokiMotionLoader(motion_dir).load_motion()


def test_motion_loader_load_defaults_to_pyroki_format(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_pyroki_npz(motion_path)

    motions = MotionLoader.load(motion_path)

    assert len(motions) == 1
    assert motions[0].fps == 30.0


def test_motion_loader_selects_pyroki_adapter(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_pyroki_npz(motion_path)

    loader = MotionLoader.from_format(motion_path, motion_format="pyroki", fps=50.0)

    assert isinstance(loader, PyrokiMotionLoader)
    assert loader.load_motion()[0].fps == 50.0


def test_proto_format_is_reserved_but_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="proto motion loading is not implemented"):
        MotionLoader.load(tmp_path / "motion.motion", motion_format="proto")
