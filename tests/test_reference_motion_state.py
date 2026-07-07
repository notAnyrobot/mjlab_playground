from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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
ReferenceMotionState = _motion_loader.ReferenceMotionState


def _identity_root_rot(num_frames: int) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_frames, 1)


def test_reference_motion_state_defaults_to_30_fps_and_preserves_required_tensors() -> None:
    root_pos = torch.zeros(3, 3)
    root_rot = _identity_root_rot(3)
    dof_pos = torch.zeros(3, 29)

    state = ReferenceMotionState(root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos)

    assert state.fps == 30.0
    assert state.root_pos is root_pos
    assert state.root_rot is root_rot
    assert state.dof_pos is dof_pos
    assert state.dof_vel is None
    assert state.body_pos is None
    assert state.body_contacts is None
    assert state.clip_starts is None
    assert state.clip_lengths is None
    assert state.clip_fps is None


def test_reference_motion_state_accepts_full_reference_motion_fields() -> None:
    state = ReferenceMotionState(
        fps=50.0,
        root_pos=torch.zeros(4, 3),
        root_rot=_identity_root_rot(4),
        dof_pos=torch.zeros(4, 29),
        root_lin_vel=torch.zeros(4, 3),
        root_ang_vel=torch.zeros(4, 3),
        dof_vel=torch.zeros(4, 29),
        body_pos=torch.zeros(4, 31, 3),
        body_rot=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(4, 31, 1),
        body_lin_vel=torch.zeros(4, 31, 3),
        body_ang_vel=torch.zeros(4, 31, 3),
        body_contacts=torch.zeros(4, 31, dtype=torch.bool),
    )

    assert state.fps == 50.0
    assert state.body_pos is not None
    assert state.body_pos.shape == (4, 31, 3)
    assert state.body_contacts is not None
    assert state.body_contacts.dtype == torch.bool


def test_reference_motion_state_accepts_valid_multi_clip_package_metadata() -> None:
    state = ReferenceMotionState(
        fps=50.0,
        root_pos=torch.zeros(9, 3),
        root_rot=_identity_root_rot(9),
        dof_pos=torch.zeros(9, 29),
        clip_starts=torch.tensor([0, 3, 6]),
        clip_lengths=torch.tensor([3, 3, 3]),
        clip_fps=torch.tensor([50.0, 50.0, 50.0]),
    )

    torch.testing.assert_close(state.clip_starts, torch.tensor([0, 3, 6]))
    torch.testing.assert_close(state.clip_lengths, torch.tensor([3, 3, 3]))
    torch.testing.assert_close(state.clip_fps, torch.tensor([50.0, 50.0, 50.0]))


def test_reference_motion_state_accepts_source_foot_contacts() -> None:
    foot_contacts = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        dtype=torch.float32,
    )

    state = ReferenceMotionState(
        root_pos=torch.zeros(3, 3),
        root_rot=_identity_root_rot(3),
        dof_pos=torch.zeros(3, 29),
        foot_contacts=foot_contacts,
    )

    assert state.foot_contacts is foot_contacts


def test_reference_motion_state_rejects_bad_source_foot_contacts() -> None:
    with pytest.raises(ValueError, match=r"foot_contacts must have shape \(T, 2\)"):
        ReferenceMotionState(
            root_pos=torch.zeros(3, 3),
            root_rot=_identity_root_rot(3),
            dof_pos=torch.zeros(3, 29),
            foot_contacts=torch.zeros(3, 3),
        )

    with pytest.raises(ValueError, match="foot_contacts must share frame count"):
        ReferenceMotionState(
            root_pos=torch.zeros(3, 3),
            root_rot=_identity_root_rot(3),
            dof_pos=torch.zeros(3, 29),
            foot_contacts=torch.zeros(4, 2),
        )

    with pytest.raises(ValueError, match="foot_contacts contains non-finite values"):
        ReferenceMotionState(
            root_pos=torch.zeros(3, 3),
            root_rot=_identity_root_rot(3),
            dof_pos=torch.zeros(3, 29),
            foot_contacts=torch.tensor(
                [[1.0, 0.0], [float("nan"), 0.5], [0.0, 1.0]]
            ),
        )


@pytest.mark.parametrize("fps", [0.0, -30.0, float("inf"), float("nan")])
def test_reference_motion_state_rejects_non_positive_or_non_finite_fps(
    fps: float,
) -> None:
    with pytest.raises(ValueError, match="fps must be positive and finite"):
        ReferenceMotionState(
            fps=fps,
            root_pos=torch.zeros(3, 3),
            root_rot=_identity_root_rot(3),
            dof_pos=torch.zeros(3, 29),
        )


def test_reference_motion_state_rejects_non_integer_fps() -> None:
    with pytest.raises(ValueError, match="fps must be integer-valued"):
        ReferenceMotionState(
            fps=29.97,
            root_pos=torch.zeros(3, 3),
            root_rot=_identity_root_rot(3),
            dof_pos=torch.zeros(3, 29),
        )


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (
            {
                "clip_starts": torch.tensor([0, 3]),
                "clip_lengths": torch.tensor([3, 2]),
                "clip_fps": torch.tensor([30.0, 30.0]),
            },
            "clip spans must cover",
        ),
        (
            {
                "clip_starts": torch.tensor([0, 2]),
                "clip_lengths": torch.tensor([3, 3]),
                "clip_fps": torch.tensor([30.0, 30.0]),
            },
            "clip spans must be contiguous",
        ),
        (
            {
                "clip_starts": torch.tensor([0, -3]),
                "clip_lengths": torch.tensor([3, 3]),
                "clip_fps": torch.tensor([30.0, 30.0]),
            },
            "clip_starts must be non-negative",
        ),
        (
            {
                "clip_starts": torch.tensor([0, 3]),
                "clip_lengths": torch.tensor([3, 0]),
                "clip_fps": torch.tensor([30.0, 30.0]),
            },
            "clip_lengths must be positive",
        ),
        (
            {
                "clip_starts": torch.tensor([0, 3]),
                "clip_lengths": torch.tensor([3, 3]),
                "clip_fps": torch.tensor([30.0, 0.0]),
            },
            "clip_fps must be positive and finite",
        ),
        (
            {
                "clip_starts": torch.tensor([0, 3]),
                "clip_lengths": torch.tensor([3, 3]),
                "clip_fps": torch.tensor([30.0, float("inf")]),
            },
            "clip_fps must be positive and finite",
        ),
        (
            {
                "clip_starts": torch.tensor([0, 3]),
                "clip_lengths": torch.tensor([3, 3]),
                "clip_fps": torch.tensor([30.0, 29.97]),
            },
            "clip_fps must be integer-valued",
        ),
        (
            {
                "clip_starts": torch.tensor([0, 3]),
                "clip_lengths": torch.tensor([3, 3]),
                "clip_fps": torch.tensor([30.0, 60.0]),
            },
            "clip_fps must match fps",
        ),
        (
            {
                "clip_starts": torch.tensor([False, True]),
                "clip_lengths": torch.tensor([3, 3]),
                "clip_fps": torch.tensor([30.0, 30.0]),
            },
            "clip_starts must be an integer tensor",
        ),
        (
            {
                "clip_starts": torch.tensor([0, 3]),
                "clip_lengths": torch.tensor([True, True]),
                "clip_fps": torch.tensor([30.0, 30.0]),
            },
            "clip_lengths must be an integer tensor",
        ),
    ],
)
def test_reference_motion_state_rejects_invalid_package_metadata(
    metadata: dict[str, torch.Tensor],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        ReferenceMotionState(
            fps=30.0,
            root_pos=torch.zeros(6, 3),
            root_rot=_identity_root_rot(6),
            dof_pos=torch.zeros(6, 29),
            **metadata,
        )


def test_reference_motion_state_rejects_incomplete_package_metadata() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        ReferenceMotionState(
            fps=30.0,
            root_pos=torch.zeros(6, 3),
            root_rot=_identity_root_rot(6),
            dof_pos=torch.zeros(6, 29),
            clip_starts=torch.tensor([0, 3]),
        )


@pytest.mark.parametrize("num_frames", [1, 2])
def test_reference_motion_state_accepts_small_sampled_batches_without_package_metadata(
    num_frames: int,
) -> None:
    state = ReferenceMotionState(
        root_pos=torch.zeros(num_frames, 3),
        root_rot=_identity_root_rot(num_frames),
        dof_pos=torch.zeros(num_frames, 29),
    )

    assert state.root_pos.shape[0] == num_frames
    assert state.clip_starts is None
    assert state.clip_lengths is None
    assert state.clip_fps is None


def test_reference_motion_state_rejects_shape_dtype_device_and_finite_mismatches() -> None:
    with pytest.raises(ValueError, match="root_pos must have shape"):
        ReferenceMotionState(
            root_pos=torch.zeros(3, 4),
            root_rot=_identity_root_rot(3),
            dof_pos=torch.zeros(3, 29),
        )

    with pytest.raises(ValueError, match="dof_pos must share dtype"):
        ReferenceMotionState(
            root_pos=torch.zeros(3, 3, dtype=torch.float32),
            root_rot=_identity_root_rot(3).to(torch.float32),
            dof_pos=torch.zeros(3, 29, dtype=torch.float64),
        )

    root_pos = torch.zeros(3, 3)
    root_pos[1, 0] = float("nan")
    with pytest.raises(ValueError, match="root_pos contains non-finite values"):
        ReferenceMotionState(
            root_pos=root_pos,
            root_rot=_identity_root_rot(3),
            dof_pos=torch.zeros(3, 29),
        )


def test_reference_motion_state_rejects_unnormalized_root_rotation() -> None:
    with pytest.raises(ValueError, match="root_rot quaternions must be normalized"):
        ReferenceMotionState(
            root_pos=torch.zeros(3, 3),
            root_rot=torch.ones(3, 4),
            dof_pos=torch.zeros(3, 29),
        )


def test_motion_state_alias_is_not_exported_from_module_file() -> None:
    assert not hasattr(_motion_loader, "MotionState")
