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


def _rewrite_npz(path: Path, **overrides: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as original:
        payload = {key: original[key] for key in original.files}
    payload.update(overrides)
    np.savez(path, **payload)


def _write_versioned_npz(path: Path, *, frames: int = 3) -> None:
    _write_train_ready_npz(path, frames=frames)
    clip_name = b"walk.npz"
    _rewrite_npz(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        fps=np.asarray(50.0, dtype=np.float64),
        clip_starts=np.asarray([0], dtype=np.int64),
        clip_lengths=np.asarray([frames], dtype=np.int64),
        clip_fps=np.asarray([50.0], dtype=np.float64),
        clip_name_bytes=np.frombuffer(clip_name, dtype=np.uint8),
        clip_name_offsets=np.asarray([0, len(clip_name)], dtype=np.int64),
        dof_names=np.asarray(["left_hip", "right_hip", "waist"]),
        body_names=np.asarray(["pelvis", "torso"]),
    )


def test_pyroki_motion_loader_loads_single_npz_into_reference_motion(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_pyroki_npz(motion_path)

    motions = PyrokiMotionLoader(motion_path, fps=50.0).load_motion()

    assert len(motions) == 1
    motion = motions[0]
    assert motion.name == "motion.npz"
    assert not hasattr(motion, "display_name")
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


def test_motion_loader_normalizes_pyroki_float_tensors_and_quaternions(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "scaled_quaternions.npz"
    _write_pyroki_npz(motion_path)
    with np.load(motion_path, allow_pickle=False) as original:
        payload = {key: original[key] for key in original.files}
    payload["base_frame_wxyz"] *= 2.0
    np.savez(motion_path, **payload)

    motion = MotionLoader.load(
        motion_path,
        motion_format="pyroki",
        device=torch.device("cpu"),
    )[0]

    for tensor in (motion.root_pos, motion.root_rot, motion.dof_pos):
        assert tensor.dtype == torch.float32
        assert tensor.device == torch.device("cpu")
    torch.testing.assert_close(
        torch.linalg.vector_norm(motion.root_rot, dim=-1),
        torch.ones(3),
    )
    torch.testing.assert_close(
        motion.root_rot,
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ),
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

    motions = MotionLoader.load(motion_path, motion_format="mjlab", fps=50.0)

    assert len(motions) == 1
    motion = motions[0]
    assert motion.name == "walk_retargeted.npz"
    assert not hasattr(motion, "display_name")
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
    assert motion.clip_starts is None
    assert motion.clip_lengths is None
    assert motion.clip_fps is None
    assert motion.clip_name_bytes is None
    assert motion.clip_name_offsets is None
    assert motion.dof_names is None
    assert motion.body_names is None


@pytest.mark.parametrize("versioned", [False, True], ids=["legacy", "versioned"])
def test_motion_loader_normalizes_mjlab_float_tensors_and_quaternions(
    tmp_path: Path,
    versioned: bool,
) -> None:
    motion_path = tmp_path / "motion.npz"
    if versioned:
        _write_versioned_npz(motion_path)
    else:
        _write_train_ready_npz(motion_path)
    with np.load(motion_path, allow_pickle=False) as original:
        payload = {key: original[key] for key in original.files}
    for key, value in payload.items():
        if np.issubdtype(value.dtype, np.floating):
            payload[key] = value.astype(np.float64)
    payload["root_rot"] *= 2.0
    payload["body_rot"] *= 3.0
    np.savez(motion_path, **payload)

    motion = MotionLoader.load(
        motion_path,
        motion_format="mjlab",
        device=torch.device("cpu"),
    )[0]

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
    ):
        tensor = getattr(motion, field_name)
        assert tensor.dtype == torch.float32
        assert tensor.device == torch.device("cpu")
    torch.testing.assert_close(
        torch.linalg.vector_norm(motion.root_rot, dim=-1),
        torch.ones(3),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(motion.body_rot, dim=-1),
        torch.ones(3, 2),
    )
    if versioned:
        assert motion.clip_fps.dtype == torch.float32
        assert motion.clip_fps.device == torch.device("cpu")


@pytest.mark.parametrize(
    ("motion_format", "versioned", "quaternion_key"),
    [
        pytest.param("pyroki", False, "base_frame_wxyz", id="pyroki"),
        pytest.param("mjlab", False, "root_rot", id="legacy"),
        pytest.param("mjlab", True, "root_rot", id="versioned"),
    ],
)
def test_motion_loader_rejects_nonfinite_source_quaternions_with_file_context(
    tmp_path: Path,
    motion_format: str,
    versioned: bool,
    quaternion_key: str,
) -> None:
    motion_path = tmp_path / f"bad_{motion_format}.npz"
    if motion_format == "pyroki":
        _write_pyroki_npz(motion_path)
    elif versioned:
        _write_versioned_npz(motion_path)
    else:
        _write_train_ready_npz(motion_path)
    with np.load(motion_path, allow_pickle=False) as original:
        payload = {key: original[key] for key in original.files}
    payload[quaternion_key] = payload[quaternion_key].copy()
    payload[quaternion_key].reshape(-1, 4)[1, 0] = np.nan
    np.savez(motion_path, **payload)

    with pytest.raises(
        ValueError,
        match=rf"{motion_path.name}.*{quaternion_key}.*non-finite quaternions",
    ):
        MotionLoader.load(motion_path, motion_format=motion_format)


@pytest.mark.parametrize(
    ("motion_format", "versioned", "quaternion_key"),
    [
        pytest.param("pyroki", False, "base_frame_wxyz", id="pyroki-root"),
        pytest.param("mjlab", False, "body_rot", id="legacy-body"),
        pytest.param("mjlab", True, "root_rot", id="versioned-root"),
    ],
)
def test_motion_loader_rejects_zero_source_quaternions_with_file_context(
    tmp_path: Path,
    motion_format: str,
    versioned: bool,
    quaternion_key: str,
) -> None:
    motion_path = tmp_path / f"zero_{motion_format}.npz"
    if motion_format == "pyroki":
        _write_pyroki_npz(motion_path)
    elif versioned:
        _write_versioned_npz(motion_path)
    else:
        _write_train_ready_npz(motion_path)
    with np.load(motion_path, allow_pickle=False) as original:
        payload = {key: original[key] for key in original.files}
    payload[quaternion_key] = payload[quaternion_key].copy()
    payload[quaternion_key].reshape(-1, 4)[1] = 0.0
    np.savez(motion_path, **payload)

    with pytest.raises(
        ValueError,
        match=rf"{motion_path.name}.*{quaternion_key}.*zero quaternions",
    ):
        MotionLoader.load(motion_path, motion_format=motion_format)


@pytest.mark.parametrize(
    ("motion_format", "versioned", "quaternion_key", "malformed_value"),
    [
        pytest.param(
            "pyroki",
            False,
            "base_frame_wxyz",
            np.ones((3, 3), dtype=np.float32),
            id="pyroki-root",
        ),
        pytest.param(
            "mjlab",
            False,
            "body_rot",
            np.ones((3, 2, 3), dtype=np.float32),
            id="legacy-body",
        ),
        pytest.param(
            "mjlab",
            True,
            "root_rot",
            np.ones((3, 1, 4), dtype=np.float32),
            id="versioned-root-rank",
        ),
    ],
)
def test_motion_loader_rejects_malformed_quaternion_axes_with_file_context(
    tmp_path: Path,
    motion_format: str,
    versioned: bool,
    quaternion_key: str,
    malformed_value: np.ndarray,
) -> None:
    motion_path = tmp_path / f"malformed_{motion_format}.npz"
    if motion_format == "pyroki":
        _write_pyroki_npz(motion_path)
    elif versioned:
        _write_versioned_npz(motion_path)
    else:
        _write_train_ready_npz(motion_path)
    _rewrite_npz(motion_path, **{quaternion_key: malformed_value})

    with pytest.raises(
        ValueError,
        match=rf"{motion_path.name}.*{quaternion_key}.*canonical wxyz shape",
    ):
        MotionLoader.load(motion_path, motion_format=motion_format)


def test_motion_loader_rejects_misaligned_pyroki_frame_counts(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "misaligned_pyroki.npz"
    _write_pyroki_npz(motion_path)
    _rewrite_npz(
        motion_path,
        joint_angles=np.ones((2, 3), dtype=np.float64),
    )

    with pytest.raises(
        ValueError,
        match=rf"{motion_path.name}.*matching frame counts.*joint_angles=2",
    ):
        MotionLoader.load(motion_path, motion_format="pyroki")


@pytest.mark.parametrize("versioned", [False, True], ids=["legacy", "versioned"])
@pytest.mark.parametrize(
    ("field_name", "malformed_value", "expected_error"),
    [
        pytest.param(
            "root_lin_vel",
            np.ones((2, 3), dtype=np.float32),
            r"frame counts.*root_pos=3.*root_lin_vel=2",
            id="frame-count",
        ),
        pytest.param(
            "dof_vel",
            np.ones((3, 2), dtype=np.float32),
            r"DOF axes.*dof_pos=3.*dof_vel=2",
            id="dof-axis",
        ),
        pytest.param(
            "body_ang_vel",
            np.ones((3, 1, 3), dtype=np.float32),
            r"body axes.*body_pos=2.*body_ang_vel=1",
            id="body-axis",
        ),
    ],
)
def test_motion_loader_rejects_misaligned_mjlab_tensor_relations(
    tmp_path: Path,
    versioned: bool,
    field_name: str,
    malformed_value: np.ndarray,
    expected_error: str,
) -> None:
    motion_path = tmp_path / "misaligned.npz"
    if versioned:
        _write_versioned_npz(motion_path)
    else:
        _write_train_ready_npz(motion_path)
    _rewrite_npz(motion_path, **{field_name: malformed_value})

    with pytest.raises(
        ValueError,
        match=rf"{motion_path.name}.*{expected_error}",
    ):
        MotionLoader.load(motion_path, motion_format="mjlab")


@pytest.mark.parametrize(
    ("metadata_key", "malformed_value", "expected_error"),
    [
        pytest.param(
            "clip_starts",
            np.asarray([1], dtype=np.int64),
            r"clip spans must start at frame 0",
            id="first-clip-start",
        ),
        pytest.param(
            "clip_lengths",
            np.asarray([4], dtype=np.int64),
            r"clip spans must cover packed frame count 3, got 4",
            id="packed-frame-coverage",
        ),
        pytest.param(
            "clip_fps",
            np.asarray([25.0], dtype=np.float32),
            r"clip_fps values must match fps 50\.0",
            id="clip-fps",
        ),
        pytest.param(
            "clip_fps",
            np.asarray([[50.0]], dtype=np.float32),
            r"clip_fps must be a 1D floating array",
            id="clip-fps-rank",
        ),
        pytest.param(
            "clip_name_offsets",
            np.asarray([0, 7], dtype=np.int64),
            r"clip_name_offsets must span clip_name_bytes length 8, got end 7",
            id="clip-name-span",
        ),
        pytest.param(
            "dof_names",
            np.asarray(["left_hip"]),
            r"dof_names count must match dof_pos axis 3, got 1",
            id="dof-name-count",
        ),
        pytest.param(
            "body_names",
            np.asarray(["pelvis"]),
            r"body_names count must match body_pos axis 2, got 1",
            id="body-name-count",
        ),
    ],
)
def test_motion_loader_rejects_malformed_versioned_operational_metadata(
    tmp_path: Path,
    metadata_key: str,
    malformed_value: np.ndarray,
    expected_error: str,
) -> None:
    motion_path = tmp_path / "malformed_metadata.npz"
    _write_versioned_npz(motion_path)
    _rewrite_npz(motion_path, **{metadata_key: malformed_value})

    with pytest.raises(
        ValueError,
        match=rf"{motion_path.name}.*{expected_error}",
    ):
        MotionLoader.load(motion_path, motion_format="mjlab")


def test_motion_loader_preserves_versioned_operational_metadata(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "versioned.npz"
    _write_versioned_npz(motion_path)

    motion = MotionLoader.load(motion_path, motion_format="mjlab")[0]

    torch.testing.assert_close(motion.clip_starts, torch.tensor([0]))
    torch.testing.assert_close(motion.clip_lengths, torch.tensor([3]))
    torch.testing.assert_close(motion.clip_fps, torch.tensor([50.0]))
    assert motion.clip_name(0) == "walk.npz"
    assert motion.dof_names == ("left_hip", "right_hip", "waist")
    assert motion.body_names == ("pelvis", "torso")


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
