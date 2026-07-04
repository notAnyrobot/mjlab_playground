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
ReferenceMotionNpzLoader = _motion_loader.ReferenceMotionNpzLoader


def _write_pyroki_npz(path: Path, *, frames: int = 3) -> None:
    np.savez(
        path,
        base_frame_pos=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float64,
        )[:frames],
        base_frame_wxyz=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )[:frames],
        joint_angles=np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=np.float64,
        )[:frames],
    )


def _write_contact_labels_npz(path: Path, *, frames: int = 3) -> None:
    np.savez(
        path,
        foot_contacts=np.array(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            dtype=np.float64,
        )[:frames],
    )


def _write_train_ready_npz(path: Path, *, frames: int = 3) -> None:
    body_count = 2
    dof_count = 3
    identity_root_quat = np.tile(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        (frames, 1),
    )
    identity_body_quat = np.tile(
        np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
        (frames, body_count, 1),
    )
    np.savez(
        path,
        root_pos=np.arange(frames * 3, dtype=np.float32).reshape(frames, 3),
        root_rot=identity_root_quat,
        dof_pos=np.arange(frames * dof_count, dtype=np.float32).reshape(
            frames, dof_count
        ),
        root_lin_vel=np.ones((frames, 3), dtype=np.float32),
        root_ang_vel=np.ones((frames, 3), dtype=np.float32) * 2.0,
        dof_vel=np.ones((frames, dof_count), dtype=np.float32) * 3.0,
        body_pos=np.ones((frames, body_count, 3), dtype=np.float32) * 4.0,
        body_rot=identity_body_quat,
        body_lin_vel=np.ones((frames, body_count, 3), dtype=np.float32) * 5.0,
        body_ang_vel=np.ones((frames, body_count, 3), dtype=np.float32) * 6.0,
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
    assert motion.display_name == "motion"
    assert motion.fps == 50.0
    assert motion.root_pos.dtype == torch.float32
    assert motion.root_rot.dtype == torch.float32
    assert motion.dof_pos.dtype == torch.float32
    torch.testing.assert_close(
        motion.root_pos,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )
    torch.testing.assert_close(
        motion.root_rot,
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ),
    )
    torch.testing.assert_close(
        motion.dof_pos,
        torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
    )


def test_pyroki_motion_loader_optionally_loads_matching_contact_labels(
    tmp_path: Path,
) -> None:
    motion_dir = tmp_path / "pyroki-retargeted-astro"
    contact_dir = tmp_path / "contacts"
    motion_dir.mkdir()
    contact_dir.mkdir()
    motion_path = motion_dir / "0008_0008_Walking001_poses_keypoints_retargeted.npz"
    contact_path = contact_dir / "0008_0008_Walking001_poses_keypoints_contacts.npz"
    _write_pyroki_npz(motion_path)
    _write_contact_labels_npz(contact_path)

    motions = MotionLoader.load(motion_path, contact_labels=contact_dir)

    assert len(motions) == 1
    assert motions[0].foot_contacts is not None
    torch.testing.assert_close(
        motions[0].foot_contacts,
        torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
    )


def test_pyroki_motion_loader_validates_real_sfu_contact_label_artifact() -> None:
    motion_path = Path(
        "/media/android/data/motion_datasets/protomotions/astro/sfu/"
        "pyroki-retargeted-astro/"
        "0008_0008_Walking001_poses_keypoints_retargeted.npz"
    )
    contact_dir = Path(
        "/media/android/data/motion_datasets/protomotions/smpl/sfu/contacts"
    )
    if not motion_path.exists() or not contact_dir.exists():
        pytest.skip("local SFU PyRoki/contact artifacts are not available")

    motion = MotionLoader.load(motion_path, contact_labels=contact_dir)[0]

    assert motion.foot_contacts is not None
    assert motion.foot_contacts.shape == (motion.root_pos.shape[0], 2)
    assert motion.foot_contacts.dtype == torch.float32
    assert motion.root_pos.shape == (807, 3)
    assert torch.all((motion.foot_contacts >= 0.0) & (motion.foot_contacts <= 1.0))


def test_pyroki_motion_loader_fails_fast_when_contact_labels_are_missing(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "motion_retargeted.npz"
    contact_dir = tmp_path / "contacts"
    contact_dir.mkdir()
    _write_pyroki_npz(motion_path)

    with pytest.raises(FileNotFoundError, match="motion_contacts\\.npz"):
        MotionLoader.load(motion_path, contact_labels=contact_dir)


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


def test_reference_motion_npz_loader_loads_train_ready_export(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "walk_retargeted.npz"
    _write_train_ready_npz(motion_path)

    motions = ReferenceMotionNpzLoader(motion_path, fps=50.0).load_motion()

    assert len(motions) == 1
    motion = motions[0]
    assert motion.name == "walk_retargeted.npz"
    assert motion.display_name == "walk_retargeted"
    assert motion.fps == 50.0
    torch.testing.assert_close(
        motion.root_pos,
        torch.arange(9, dtype=torch.float32).reshape(3, 3),
    )
    assert motion.root_lin_vel is not None
    assert motion.root_ang_vel is not None
    assert motion.dof_vel is not None
    assert motion.body_pos is not None
    assert motion.body_rot is not None
    assert motion.body_lin_vel is not None
    assert motion.body_ang_vel is not None


def test_reference_motion_npz_loader_optionally_loads_body_contacts(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "walk.npz"
    _write_train_ready_npz(motion_path)
    with np.load(motion_path, allow_pickle=False) as original:
        payload = {key: original[key] for key in original.files}
    payload["body_contacts"] = np.array(
        [[True, False], [False, True], [True, True]],
        dtype=np.bool_,
    )
    np.savez(motion_path, **payload)

    motion = ReferenceMotionNpzLoader(motion_path).load_motion()[0]

    assert motion.body_contacts is not None
    assert motion.body_contacts.dtype == torch.bool
    torch.testing.assert_close(
        motion.body_contacts,
        torch.tensor([[True, False], [False, True], [True, True]]),
    )


def test_reference_motion_npz_loader_loads_flat_directory_in_sorted_order(
    tmp_path: Path,
) -> None:
    motion_dir = tmp_path / "mjlab-astro"
    motion_dir.mkdir()
    _write_train_ready_npz(motion_dir / "b_motion.npz")
    _write_train_ready_npz(motion_dir / "a_motion.npz")
    (motion_dir / "README.txt").write_text("sidecar")

    motions = ReferenceMotionNpzLoader(motion_dir).load_motion()

    assert [motion.name for motion in motions] == ["a_motion.npz", "b_motion.npz"]


def test_motion_loader_selects_reference_motion_npz_adapter(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_train_ready_npz(motion_path)

    loader = MotionLoader.from_format(motion_path, motion_format="mjlab", fps=50.0)

    assert isinstance(loader, ReferenceMotionNpzLoader)
    assert loader.load_motion()[0].fps == 50.0


def test_reference_motion_npz_loader_fails_fast_with_bad_clip_path(
    tmp_path: Path,
) -> None:
    motion_dir = tmp_path / "mjlab-astro"
    motion_dir.mkdir()
    _write_train_ready_npz(motion_dir / "good.npz")
    np.savez(motion_dir / "bad.npz", root_pos=np.zeros((3, 3), dtype=np.float32))

    with pytest.raises(ValueError, match=r"bad\.npz: Missing required MotionLib key"):
        ReferenceMotionNpzLoader(motion_dir).load_motion()


def test_proto_format_is_reserved_but_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="proto motion loading is not implemented"):
        MotionLoader.load(tmp_path / "motion.motion", motion_format="proto")
