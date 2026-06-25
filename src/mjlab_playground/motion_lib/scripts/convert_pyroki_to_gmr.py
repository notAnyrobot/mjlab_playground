"""Convert PyRoki retargeted motion ``.npz`` files to GMR motion pickles.

PyRoki format in this repo is a per-motion ``.npz`` containing:
- ``base_frame_pos``: root position, shape ``(T, 3)``
- ``base_frame_wxyz``: root rotation quaternion in ``wxyz`` order, shape ``(T, 4)``
- ``joint_angles``: robot joint angles, shape ``(T, D)``

GMR stores root rotation as ``xyzw`` in its pickle files.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

PYROKI_REQUIRED_KEYS = ("base_frame_pos", "base_frame_wxyz", "joint_angles")
GMR_REQUIRED_KEYS = (
    "fps",
    "root_pos",
    "root_rot",
    "dof_pos",
    "local_body_pos",
    "link_body_list",
)
PYROKI_DIR_PREFIX = "pyroki-retargeted-"
GMR_DIR_PREFIX = "gmr-"


def _validate_positive_fps(fps: float) -> float:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"--fps must be positive and finite, got {fps!r}")
    return fps


def _load_array(data: np.lib.npyio.NpzFile, key: str) -> NDArray[np.generic]:
    if key not in data.files:
        raise ValueError(f"Missing required PyRoki key {key!r}")
    array = np.asarray(data[key])
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{key!r} must be numeric, got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key!r} contains non-finite values")
    return array


def load_pyroki_motion(input_path: Path) -> dict[str, NDArray[np.generic]]:
    """Load and validate one PyRoki ``.npz`` motion."""
    if input_path.suffix != ".npz":
        raise ValueError(f"Expected a .npz PyRoki motion file, got {input_path}")

    with np.load(input_path, allow_pickle=False) as data:
        arrays = {key: _load_array(data, key) for key in PYROKI_REQUIRED_KEYS}

    root_pos = arrays["base_frame_pos"]
    root_rot_wxyz = arrays["base_frame_wxyz"]
    dof_pos = arrays["joint_angles"]

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(
            f"'base_frame_pos' must have shape (num_frames, 3), got {root_pos.shape}"
        )
    if root_rot_wxyz.ndim != 2 or root_rot_wxyz.shape[1] != 4:
        raise ValueError(
            f"'base_frame_wxyz' must have shape (num_frames, 4), got {root_rot_wxyz.shape}"
        )
    if dof_pos.ndim != 2:
        raise ValueError(f"'joint_angles' must be 2D, got shape {dof_pos.shape}")

    frame_count = root_pos.shape[0]
    if frame_count == 0:
        raise ValueError("PyRoki motion must contain at least one frame")
    if root_rot_wxyz.shape[0] != frame_count or dof_pos.shape[0] != frame_count:
        raise ValueError(
            "PyRoki arrays must have matching frame counts: "
            f"base_frame_pos={root_pos.shape[0]}, "
            f"base_frame_wxyz={root_rot_wxyz.shape[0]}, "
            f"joint_angles={dof_pos.shape[0]}"
        )

    quat_norms = np.linalg.norm(root_rot_wxyz, axis=1)
    if not np.allclose(quat_norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("'base_frame_wxyz' quaternions must be normalized")

    return arrays


def pyroki_motion_to_gmr(
    pyroki_motion: dict[str, NDArray[np.generic]],
    *,
    fps: float,
) -> dict[str, Any]:
    """Translate validated PyRoki arrays into a GMR motion dictionary."""
    _validate_positive_fps(fps)

    root_rot_wxyz = pyroki_motion["base_frame_wxyz"]
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

    return {
        "fps": fps,
        "root_pos": pyroki_motion["base_frame_pos"],
        "root_rot": root_rot_xyzw,
        "dof_pos": pyroki_motion["joint_angles"],
        "local_body_pos": None,
        "link_body_list": None,
    }


def validate_gmr_motion(gmr_motion: dict[str, Any], *, expected_fps: float) -> None:
    """Validate the GMR dict shape used by GMR and MimicKit."""
    missing = [key for key in GMR_REQUIRED_KEYS if key not in gmr_motion]
    if missing:
        raise ValueError(f"Written GMR motion is missing keys: {missing}")
    if gmr_motion["fps"] != expected_fps:
        raise ValueError(
            f"Written GMR fps mismatch: expected {expected_fps}, got {gmr_motion['fps']}"
        )

    root_pos = np.asarray(gmr_motion["root_pos"])
    root_rot = np.asarray(gmr_motion["root_rot"])
    dof_pos = np.asarray(gmr_motion["dof_pos"])

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Written root_pos must have shape (num_frames, 3), got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"Written root_rot must have shape (num_frames, 4), got {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"Written dof_pos must be 2D, got shape {dof_pos.shape}")
    if root_rot.shape[0] != root_pos.shape[0] or dof_pos.shape[0] != root_pos.shape[0]:
        raise ValueError("Written GMR arrays have mismatched frame counts")
    if not np.all(np.isfinite(root_pos)):
        raise ValueError("Written root_pos contains non-finite values")
    if not np.all(np.isfinite(root_rot)):
        raise ValueError("Written root_rot contains non-finite values")
    if not np.all(np.isfinite(dof_pos)):
        raise ValueError("Written dof_pos contains non-finite values")

    quat_norms = np.linalg.norm(root_rot, axis=1)
    if not np.allclose(quat_norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("Written root_rot quaternions must be normalized")


def write_gmr_motion(
    input_path: Path,
    output_path: Path,
    *,
    fps: float,
    force_remake: bool = False,
) -> None:
    """Convert one PyRoki ``.npz`` motion into one GMR ``.pkl`` motion."""
    if output_path.suffix != ".pkl":
        raise ValueError(f"Expected output path with .pkl suffix, got {output_path}")
    if output_path.exists() and not force_remake:
        raise FileExistsError(f"Output already exists: {output_path}")

    pyroki_motion = load_pyroki_motion(input_path)
    gmr_motion = pyroki_motion_to_gmr(pyroki_motion, fps=fps)
    validate_gmr_motion(gmr_motion, expected_fps=fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        pickle.dump(gmr_motion, output_file)

    with output_path.open("rb") as output_file:
        written_motion = pickle.load(output_file)
    validate_gmr_motion(written_motion, expected_fps=fps)


def default_output_path(input_path: Path) -> Path:
    """Return the dataset-sibling default GMR output path for an input path."""
    if input_path.is_dir():
        if input_path.name.startswith(PYROKI_DIR_PREFIX):
            robot_suffix = input_path.name[len(PYROKI_DIR_PREFIX) :]
            if robot_suffix:
                return input_path.parent / f"{GMR_DIR_PREFIX}{robot_suffix}"
        return input_path.parent / f"{GMR_DIR_PREFIX}{input_path.name}"

    if input_path.suffix != ".npz":
        raise ValueError(f"Expected a .npz PyRoki motion file, got {input_path}")

    parent = input_path.parent
    if parent.name.startswith(PYROKI_DIR_PREFIX):
        robot_suffix = parent.name[len(PYROKI_DIR_PREFIX) :]
        if robot_suffix:
            return parent.parent / f"{GMR_DIR_PREFIX}{robot_suffix}" / f"{input_path.stem}.pkl"

    return parent / f"{input_path.stem}.pkl"


def _direct_npz_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix == ".npz")


def convert_path(
    input_path: Path,
    output_path: Path | None,
    *,
    fps: float,
    force_remake: bool,
) -> list[Path]:
    """Convert one file or a flat directory of PyRoki ``.npz`` files."""
    _validate_positive_fps(fps)
    output_path = default_output_path(input_path) if output_path is None else output_path

    if input_path.is_file():
        write_gmr_motion(input_path, output_path, fps=fps, force_remake=force_remake)
        return [output_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Batch output must be a directory, got file: {output_path}")

    input_files = _direct_npz_files(input_path)
    if not input_files:
        raise ValueError(f"No direct child .npz files found in {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for npz_path in input_files:
        pkl_path = output_path / f"{npz_path.stem}.pkl"
        write_gmr_motion(npz_path, pkl_path, fps=fps, force_remake=force_remake)
        written_paths.append(pkl_path)
    return written_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PyRoki retargeted .npz motions to GMR .pkl motions."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .npz file or flat directory of .npz files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output .pkl file or output directory. Defaults to a dataset sibling "
            "directory such as gmr-astro for pyroki-retargeted-astro inputs."
        ),
    )
    parser.add_argument("--fps", required=True, type=float, help="Motion FPS to write into GMR files.")
    parser.add_argument(
        "--force-remake",
        action="store_true",
        help="Overwrite existing output .pkl files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    written_paths = convert_path(
        args.input,
        args.output,
        fps=args.fps,
        force_remake=args.force_remake,
    )
    for written_path in written_paths:
        print(f"Wrote {written_path}")


if __name__ == "__main__":
    main()
