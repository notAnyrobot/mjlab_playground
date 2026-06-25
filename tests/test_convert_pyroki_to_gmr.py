"""Tests for PyRoki `.npz` to GMR `.pkl` conversion."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "src"
    / "mjlab_playground"
    / "motion_lib"
    / "scripts"
    / "convert_pyroki_to_gmr.py"
)


def _load_converter_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("convert_pyroki_to_gmr", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_converter = _load_converter_module()
convert_path = _converter.convert_path
default_output_path = _converter.default_output_path


def _write_pyroki_npz(path: Path, *, frames: int = 2, dtype: str = "float32") -> None:
    base_frame_pos = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=dtype,
    )[:frames]
    base_frame_wxyz = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=dtype,
    )[:frames]
    joint_angles = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=dtype,
    )[:frames]
    np.savez(
        path,
        base_frame_pos=base_frame_pos,
        base_frame_wxyz=base_frame_wxyz,
        joint_angles=joint_angles,
    )


def _load_pickle(path: Path) -> dict[str, object]:
    with path.open("rb") as motion_file:
        return pickle.load(motion_file)


def _array(value: object) -> np.ndarray:
    assert isinstance(value, np.ndarray)
    return value


def test_single_file_conversion_preserves_pyroki_truth_except_quat_order(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "motion.npz"
    output_path = tmp_path / "motion.pkl"
    _write_pyroki_npz(input_path, dtype="float64")

    written = convert_path(input_path, output_path, fps=30.0, force_remake=False)

    assert written == [output_path]
    gmr_motion = _load_pickle(output_path)
    root_pos = _array(gmr_motion["root_pos"])
    root_rot = _array(gmr_motion["root_rot"])
    dof_pos = _array(gmr_motion["dof_pos"])
    assert gmr_motion["fps"] == 30.0
    np.testing.assert_array_equal(
        root_pos,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64"),
    )
    np.testing.assert_array_equal(
        root_rot,
        np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]], dtype="float64"),
    )
    np.testing.assert_array_equal(
        dof_pos,
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float64"),
    )
    assert root_pos.dtype == np.dtype("float64")
    assert root_rot.dtype == np.dtype("float64")
    assert dof_pos.dtype == np.dtype("float64")
    assert gmr_motion["local_body_pos"] is None
    assert gmr_motion["link_body_list"] is None


def test_batch_conversion_uses_only_direct_child_npz_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "pyroki"
    nested_dir = input_dir / "nested"
    output_dir = tmp_path / "gmr"
    nested_dir.mkdir(parents=True)
    _write_pyroki_npz(input_dir / "b_motion.npz")
    _write_pyroki_npz(input_dir / "a_motion.npz")
    _write_pyroki_npz(nested_dir / "ignored.npz")

    written = convert_path(input_dir, output_dir, fps=50.0, force_remake=False)

    assert written == [output_dir / "a_motion.pkl", output_dir / "b_motion.pkl"]
    assert not (output_dir / "ignored.pkl").exists()
    assert _load_pickle(output_dir / "a_motion.pkl")["fps"] == 50.0


def test_batch_conversion_defaults_to_gmr_robot_sibling_dir(tmp_path: Path) -> None:
    input_dir = tmp_path / "dancedb" / "pyroki-retargeted-astro"
    input_dir.mkdir(parents=True)
    _write_pyroki_npz(input_dir / "motion.npz")

    written = convert_path(input_dir, None, fps=30.0, force_remake=False)

    output_path = tmp_path / "dancedb" / "gmr-astro" / "motion.pkl"
    assert default_output_path(input_dir) == tmp_path / "dancedb" / "gmr-astro"
    assert written == [output_path]
    assert _load_pickle(output_path)["fps"] == 30.0


def test_single_file_inside_pyroki_dir_defaults_to_gmr_robot_sibling_dir(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "dancedb" / "pyroki-retargeted-astro"
    input_dir.mkdir(parents=True)
    input_path = input_dir / "motion.npz"
    _write_pyroki_npz(input_path)

    written = convert_path(input_path, None, fps=30.0, force_remake=False)

    output_path = tmp_path / "dancedb" / "gmr-astro" / "motion.pkl"
    assert default_output_path(input_path) == output_path
    assert written == [output_path]
    assert _load_pickle(output_path)["fps"] == 30.0


def test_existing_output_requires_force_remake(tmp_path: Path) -> None:
    input_path = tmp_path / "motion.npz"
    output_path = tmp_path / "motion.pkl"
    _write_pyroki_npz(input_path)
    output_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        convert_path(input_path, output_path, fps=30.0, force_remake=False)

    convert_path(input_path, output_path, fps=30.0, force_remake=True)
    assert _load_pickle(output_path)["fps"] == 30.0


def test_batch_conversion_fails_on_bad_npz(tmp_path: Path) -> None:
    input_dir = tmp_path / "pyroki"
    output_dir = tmp_path / "gmr"
    input_dir.mkdir()
    _write_pyroki_npz(input_dir / "good.npz")
    np.savez(input_dir / "bad.npz", base_frame_pos=np.zeros((1, 3)))

    with pytest.raises(ValueError, match="Missing required PyRoki key"):
        convert_path(input_dir, output_dir, fps=30.0, force_remake=False)


def test_fps_must_be_positive_and_finite(tmp_path: Path) -> None:
    input_path = tmp_path / "motion.npz"
    output_path = tmp_path / "motion.pkl"
    _write_pyroki_npz(input_path)

    with pytest.raises(ValueError, match="--fps must be positive"):
        convert_path(input_path, output_path, fps=0.0, force_remake=False)
