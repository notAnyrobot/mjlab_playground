"""Tests for the all-split PyRoki to GMR Bash helper."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "batch_convert_pyroki_to_gmr.sh"
SCRIPT_ENV = {"PYTHON": sys.executable}


def _write_pyroki_npz(path: Path) -> None:
  np.savez(
    path,
    base_frame_pos=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    base_frame_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    joint_angles=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
  )


def _load_pickle(path: Path) -> dict[str, object]:
  with path.open("rb") as motion_file:
    return pickle.load(motion_file)


def test_batch_script_converts_all_direct_child_splits(tmp_path: Path) -> None:
  astro_root = tmp_path / "astro"
  for split_name in ("dancedb", "accad"):
    input_dir = astro_root / split_name / "pyroki-retargeted-astro"
    input_dir.mkdir(parents=True)
    _write_pyroki_npz(input_dir / "motion.npz")
  nested_input = astro_root / "group" / "nested" / "pyroki-retargeted-astro"
  nested_input.mkdir(parents=True)
  _write_pyroki_npz(nested_input / "ignored.npz")

  result = subprocess.run(
    [
      "bash",
      str(SCRIPT),
      "--astro-root",
      str(astro_root),
      "--fps",
      "30",
    ],
    check=True,
    text=True,
    capture_output=True,
    env=SCRIPT_ENV,
  )

  assert "Converted 2 split(s)" in result.stdout
  assert _load_pickle(astro_root / "accad" / "gmr-astro" / "motion.pkl")["fps"] == 30.0
  assert _load_pickle(astro_root / "dancedb" / "gmr-astro" / "motion.pkl")["fps"] == 30.0
  assert not (astro_root / "group" / "nested" / "gmr-astro" / "ignored.pkl").exists()


def test_batch_script_supports_explicit_split_and_strict_missing_split(
  tmp_path: Path,
) -> None:
  astro_root = tmp_path / "astro"
  for split_name in ("dancedb", "accad"):
    input_dir = astro_root / split_name / "pyroki-retargeted-astro"
    input_dir.mkdir(parents=True)
    _write_pyroki_npz(input_dir / "motion.npz")

  subprocess.run(
    [
      "bash",
      str(SCRIPT),
      "--astro-root",
      str(astro_root),
      "--fps",
      "30",
      "--split",
      "dancedb",
    ],
    check=True,
    text=True,
    capture_output=True,
    env=SCRIPT_ENV,
  )

  assert (astro_root / "dancedb" / "gmr-astro" / "motion.pkl").exists()
  assert not (astro_root / "accad" / "gmr-astro" / "motion.pkl").exists()

  result = subprocess.run(
    [
      "bash",
      str(SCRIPT),
      "--astro-root",
      str(astro_root),
      "--fps",
      "30",
      "--split",
      "missing",
    ],
    check=False,
    text=True,
    capture_output=True,
    env=SCRIPT_ENV,
  )

  assert result.returncode != 0
  assert "missing PyRoki split input directory" in result.stderr


def test_batch_script_requires_force_remake_for_existing_outputs(tmp_path: Path) -> None:
  astro_root = tmp_path / "astro"
  input_dir = astro_root / "dancedb" / "pyroki-retargeted-astro"
  output_dir = astro_root / "dancedb" / "gmr-astro"
  input_dir.mkdir(parents=True)
  output_dir.mkdir(parents=True)
  _write_pyroki_npz(input_dir / "motion.npz")
  (output_dir / "motion.pkl").write_bytes(b"existing")

  result = subprocess.run(
    [
      "bash",
      str(SCRIPT),
      "--astro-root",
      str(astro_root),
      "--fps",
      "30",
    ],
    check=False,
    text=True,
    capture_output=True,
    env=SCRIPT_ENV,
  )

  assert result.returncode != 0
  assert "Output already exists" in result.stderr

  subprocess.run(
    [
      "bash",
      str(SCRIPT),
      "--astro-root",
      str(astro_root),
      "--fps",
      "30",
      "--force-remake",
    ],
    check=True,
    text=True,
    capture_output=True,
    env=SCRIPT_ENV,
  )
  assert _load_pickle(output_dir / "motion.pkl")["fps"] == 30.0
