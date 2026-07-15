from __future__ import annotations

import dataclasses
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
ReferenceFrame = _motion_loader.ReferenceFrame
ReferenceMotion = _motion_loader.ReferenceMotion
ReferenceMotionState = _motion_loader.ReferenceMotionState


def _identity_root_rot(num_frames: int) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_frames, 1)


def _rich_one_clip(
    *,
    name: str,
    frames: int = 3,
    value_offset: float = 0.0,
    **overrides,
):
    frame_values = torch.arange(frames, dtype=torch.float32) + value_offset
    fields = {
        "name": name,
        "fps": 50.0,
        "root_pos": frame_values[:, None].repeat(1, 3),
        "root_rot": _identity_root_rot(frames),
        "dof_pos": frame_values[:, None].repeat(1, 2),
        "root_lin_vel": frame_values[:, None].repeat(1, 3),
        "root_ang_vel": frame_values[:, None].repeat(1, 3),
        "dof_vel": frame_values[:, None].repeat(1, 2),
        "body_pos": frame_values[:, None, None].repeat(1, 2, 3),
        "body_rot": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(
            frames, 2, 1
        ),
        "body_lin_vel": frame_values[:, None, None].repeat(1, 2, 3),
        "body_ang_vel": frame_values[:, None, None].repeat(1, 2, 3),
        "body_contacts": torch.arange(frames)[:, None].repeat(1, 2) % 2 == 0,
        "dof_names": ("left_hip", "right_hip"),
        "body_names": ("pelvis", "torso"),
    }
    fields.update(overrides)
    return ReferenceMotion(**fields)


def test_reference_motion_state_is_compatibility_alias_for_reference_motion() -> None:
    from mjlab_playground.motion_lib import ReferenceMotion as ExportedReferenceMotion
    from mjlab_playground.motion_lib import (
        ReferenceMotionState as ExportedReferenceMotionState,
    )

    assert ReferenceMotionState is ReferenceMotion
    assert ExportedReferenceMotionState is ExportedReferenceMotion


def test_reference_frame_accepts_one_timestep_fields() -> None:
    frame = ReferenceFrame(
        root_pos=torch.zeros(3),
        root_rot=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        dof_pos=torch.zeros(29),
        root_lin_vel=torch.ones(3),
        root_ang_vel=torch.full((3,), 2.0),
        dof_vel=torch.full((29,), 3.0),
        body_pos=torch.zeros(31, 3),
        body_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(31, 1),
        body_lin_vel=torch.ones(31, 3),
        body_ang_vel=torch.full((31, 3), 2.0),
        body_contacts=torch.zeros(31, dtype=torch.bool),
        foot_contacts=torch.tensor([1.0, 0.0]),
    )

    assert frame.root_pos.shape == (3,)
    assert frame.dof_pos.shape == (29,)
    assert frame.body_contacts is not None
    assert frame.body_contacts.shape == (31,)


def test_reference_frame_rejects_invalid_timestep_fields() -> None:
    with pytest.raises(ValueError, match="root_pos must have shape"):
        ReferenceFrame(
            root_pos=torch.zeros(1, 3),
            root_rot=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            dof_pos=torch.zeros(29),
        )

    with pytest.raises(ValueError, match="root_rot quaternions must be normalized"):
        ReferenceFrame(
            root_pos=torch.zeros(3),
            root_rot=torch.ones(4),
            dof_pos=torch.zeros(29),
        )


def test_reference_motion_state_defaults_to_30_fps_and_preserves_required_tensors() -> (
    None
):
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


def test_reference_motion_frame_extracts_single_reference_frame() -> None:
    motion = ReferenceMotion(
        fps=50.0,
        root_pos=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        root_rot=_identity_root_rot(4),
        dof_pos=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        root_lin_vel=torch.ones(4, 3),
        root_ang_vel=torch.full((4, 3), 2.0),
        dof_vel=torch.full((4, 3), 3.0),
        body_pos=torch.arange(24, dtype=torch.float32).reshape(4, 2, 3),
        body_rot=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(4, 2, 1),
        body_lin_vel=torch.full((4, 2, 3), 4.0),
        body_ang_vel=torch.full((4, 2, 3), 5.0),
        body_contacts=torch.tensor(
            [[True, False], [False, True], [True, True], [False, False]]
        ),
        foot_contacts=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 1.0]]),
        clip_starts=torch.tensor([0]),
        clip_lengths=torch.tensor([4]),
        clip_fps=torch.tensor([50.0]),
    )

    frame = motion.frame(2)

    assert isinstance(frame, ReferenceFrame)
    torch.testing.assert_close(frame.root_pos, motion.root_pos[2])
    torch.testing.assert_close(frame.root_rot, motion.root_rot[2])
    torch.testing.assert_close(frame.dof_pos, motion.dof_pos[2])
    torch.testing.assert_close(frame.root_lin_vel, motion.root_lin_vel[2])
    torch.testing.assert_close(frame.root_ang_vel, motion.root_ang_vel[2])
    torch.testing.assert_close(frame.dof_vel, motion.dof_vel[2])
    torch.testing.assert_close(frame.body_pos, motion.body_pos[2])
    torch.testing.assert_close(frame.body_rot, motion.body_rot[2])
    torch.testing.assert_close(frame.body_lin_vel, motion.body_lin_vel[2])
    torch.testing.assert_close(frame.body_ang_vel, motion.body_ang_vel[2])
    torch.testing.assert_close(frame.body_contacts, motion.body_contacts[2])
    torch.testing.assert_close(frame.foot_contacts, motion.foot_contacts[2])
    assert not hasattr(frame, "clip_starts")


@pytest.mark.parametrize("frame_index", [-1, 4])
def test_reference_motion_frame_rejects_out_of_range_indices(frame_index: int) -> None:
    motion = ReferenceMotion(
        root_pos=torch.zeros(4, 3),
        root_rot=_identity_root_rot(4),
        dof_pos=torch.zeros(4, 3),
    )

    with pytest.raises(IndexError, match=f"frame index {frame_index} out of range"):
        motion.frame(frame_index)


def test_reference_motion_from_frames_reconstructs_tensor_backed_motion() -> None:
    frames = [
        ReferenceFrame(
            root_pos=torch.tensor([1.0, 2.0, 3.0]),
            root_rot=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            dof_pos=torch.tensor([0.1, 0.2]),
            root_lin_vel=torch.tensor([1.0, 0.0, 0.0]),
            root_ang_vel=torch.tensor([0.0, 1.0, 0.0]),
            dof_vel=torch.tensor([0.3, 0.4]),
            body_pos=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            body_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            body_lin_vel=torch.zeros(2, 3),
            body_ang_vel=torch.ones(2, 3),
            body_contacts=torch.tensor([True, False]),
            foot_contacts=torch.tensor([1.0, 0.0]),
        ),
        ReferenceFrame(
            root_pos=torch.tensor([4.0, 5.0, 6.0]),
            root_rot=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            dof_pos=torch.tensor([0.5, 0.6]),
            root_lin_vel=torch.tensor([0.0, 1.0, 0.0]),
            root_ang_vel=torch.tensor([0.0, 0.0, 1.0]),
            dof_vel=torch.tensor([0.7, 0.8]),
            body_pos=torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            body_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            body_lin_vel=torch.ones(2, 3),
            body_ang_vel=torch.full((2, 3), 2.0),
            body_contacts=torch.tensor([False, True]),
            foot_contacts=torch.tensor([0.0, 1.0]),
        ),
    ]

    motion = ReferenceMotion.from_frames(
        frames,
        name="walk.npz",
        display_name="walk",
        fps=60.0,
        clip_starts=torch.tensor([0]),
        clip_lengths=torch.tensor([2]),
        clip_fps=torch.tensor([60.0]),
    )

    assert motion.name == "walk.npz"
    assert motion.display_name == "walk"
    assert motion.fps == 60.0
    torch.testing.assert_close(
        motion.root_pos, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    torch.testing.assert_close(motion.dof_pos, torch.tensor([[0.1, 0.2], [0.5, 0.6]]))
    assert motion.body_pos is not None
    assert motion.body_pos.shape == (2, 2, 3)
    assert motion.body_contacts is not None
    torch.testing.assert_close(
        motion.body_contacts, torch.tensor([[True, False], [False, True]])
    )
    torch.testing.assert_close(motion.clip_starts, torch.tensor([0]))
    torch.testing.assert_close(motion.clip_lengths, torch.tensor([2]))
    torch.testing.assert_close(motion.clip_fps, torch.tensor([60.0]))


def test_reference_motion_from_frames_requires_compatible_optional_fields() -> None:
    frames = [
        ReferenceFrame(
            root_pos=torch.zeros(3),
            root_rot=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            dof_pos=torch.zeros(2),
            body_pos=torch.zeros(1, 3),
        ),
        ReferenceFrame(
            root_pos=torch.ones(3),
            root_rot=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            dof_pos=torch.ones(2),
        ),
    ]

    with pytest.raises(
        ValueError, match="body_pos must be present on every frame or none"
    ):
        ReferenceMotion.from_frames(frames)


def test_reference_motion_from_frames_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="requires at least one frame"):
        ReferenceMotion.from_frames([])


def test_reference_motion_from_clips_assembles_implicit_and_explicit_one_clip_values() -> (
    None
):
    first = _rich_one_clip(name="walk.npz", frames=3)
    second_name = "turn-测试.npz"
    encoded_second_name = second_name.encode("utf-8")
    second = _rich_one_clip(
        name="package-file.npz",
        frames=4,
        value_offset=10.0,
        clip_starts=torch.tensor([0]),
        clip_lengths=torch.tensor([4]),
        clip_fps=torch.tensor([50.0]),
        clip_name_bytes=torch.tensor(list(encoded_second_name), dtype=torch.uint8),
        clip_name_offsets=torch.tensor([0, len(encoded_second_name)]),
    )

    assembled = ReferenceMotion.from_clips([first, second])

    assert assembled.fps == 50.0
    torch.testing.assert_close(assembled.clip_starts, torch.tensor([0, 3]))
    torch.testing.assert_close(assembled.clip_lengths, torch.tensor([3, 4]))
    torch.testing.assert_close(assembled.clip_fps, torch.tensor([50.0, 50.0]))
    assert assembled.clip_name(0) == "walk.npz"
    assert assembled.clip_name(1) == second_name
    assert assembled.dof_names == first.dof_names
    assert assembled.body_names == first.body_names
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
        expected = torch.cat([getattr(first, field_name), getattr(second, field_name)])
        torch.testing.assert_close(getattr(assembled, field_name), expected)


def test_reference_motion_from_clips_accepts_one_rich_clip() -> None:
    clip = _rich_one_clip(name="walk.npz")

    assembled = ReferenceMotion.from_clips([clip])

    torch.testing.assert_close(assembled.clip_starts, torch.tensor([0]))
    torch.testing.assert_close(assembled.clip_lengths, torch.tensor([3]))
    assert assembled.clip_name(0) == "walk.npz"
    torch.testing.assert_close(assembled.root_pos, clip.root_pos)


def test_reference_motion_from_clips_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="requires at least one clip"):
        ReferenceMotion.from_clips([])


def test_reference_motion_from_clips_rejects_already_multi_clip_input() -> None:
    package = ReferenceMotion.from_clips(
        [
            _rich_one_clip(name="walk.npz"),
            _rich_one_clip(name="turn.npz", value_offset=10.0),
        ]
    )

    with pytest.raises(ValueError, match="clip 0.*already multi-clip"):
        ReferenceMotion.from_clips([package])


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("fps", "fps"),
        ("dtype", "dtype"),
        ("dof_count", "DOF count"),
        ("body_count", "body count"),
        ("dof_names", "dof_names"),
        ("body_names", "body_names"),
        ("body_contacts", "body_contacts presence"),
        ("body_contacts_dtype", "body_contacts dtype"),
        ("root_rot", "root_rot quaternions"),
        ("body_rot", "body_rot quaternions"),
    ],
)
def test_reference_motion_from_clips_identifies_incompatible_clip(
    mismatch: str,
    message: str,
) -> None:
    first = _rich_one_clip(name="walk.npz")
    second = _rich_one_clip(name="turn.npz", value_offset=10.0)
    if mismatch == "fps":
        second = dataclasses.replace(second, fps=60.0)
    elif mismatch == "dtype":
        second = dataclasses.replace(
            second,
            **{
                field_name: getattr(second, field_name).to(torch.float64)
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
                )
            },
        )
    elif mismatch == "dof_count":
        second = dataclasses.replace(
            second,
            dof_pos=torch.zeros(3, 3),
            dof_vel=torch.zeros(3, 3),
            dof_names=("left_hip", "right_hip", "waist"),
        )
    elif mismatch == "body_count":
        second = dataclasses.replace(
            second,
            body_pos=torch.zeros(3, 3, 3),
            body_rot=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(3, 3, 1),
            body_lin_vel=torch.zeros(3, 3, 3),
            body_ang_vel=torch.zeros(3, 3, 3),
            body_contacts=torch.zeros(3, 3, dtype=torch.bool),
            body_names=("pelvis", "torso", "foot"),
        )
    elif mismatch == "dof_names":
        second = dataclasses.replace(
            second, dof_names=("right_hip", "left_hip")
        )
    elif mismatch == "body_names":
        second = dataclasses.replace(second, body_names=("torso", "pelvis"))
    elif mismatch == "body_contacts":
        second = dataclasses.replace(second, body_contacts=None)
    elif mismatch == "body_contacts_dtype":
        second = dataclasses.replace(
            second, body_contacts=second.body_contacts.to(torch.float32)
        )
    elif mismatch == "root_rot":
        object.__setattr__(second, "root_rot", second.root_rot * 2.0)
    elif mismatch == "body_rot":
        object.__setattr__(second, "body_rot", second.body_rot * 2.0)
    else:  # pragma: no cover - keeps the test table exhaustive
        raise AssertionError(f"unknown mismatch {mismatch}")

    with pytest.raises(ValueError, match=rf"clip 1.*{message}"):
        ReferenceMotion.from_clips([first, second])


@pytest.mark.parametrize(
    "field_name",
    [
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
    ],
)
def test_reference_motion_from_clips_identifies_missing_rich_field(
    field_name: str,
) -> None:
    first = _rich_one_clip(name="walk.npz")
    second = _rich_one_clip(name="turn.npz", value_offset=10.0)
    object.__setattr__(second, field_name, None)

    with pytest.raises(ValueError, match=rf"clip 1.*{field_name}"):
        ReferenceMotion.from_clips([first, second])


@pytest.mark.parametrize(
    "field_name",
    [
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
    ],
)
def test_reference_motion_from_clips_identifies_malformed_rich_field_shape(
    field_name: str,
) -> None:
    first = _rich_one_clip(name="walk.npz")
    second = _rich_one_clip(name="turn.npz", value_offset=10.0)
    malformed = (
        torch.zeros(3, 4)
        if field_name == "root_pos"
        else getattr(second, field_name)[:-1]
    )
    object.__setattr__(second, field_name, malformed)
    mismatch = "shape" if field_name == "root_pos" else "frame count"

    with pytest.raises(ValueError, match=rf"clip 1.*{field_name}.*{mismatch}"):
        ReferenceMotion.from_clips([first, second])


def test_reference_motion_from_clips_rejects_source_contacts() -> None:
    first = _rich_one_clip(name="walk.npz")
    second = dataclasses.replace(
        _rich_one_clip(name="turn.npz", value_offset=10.0),
        foot_contacts=torch.zeros(3, 2),
    )

    with pytest.raises(ValueError, match="clip 1.*foot_contacts"):
        ReferenceMotion.from_clips([first, second])


def test_reference_motion_from_clips_identifies_short_rich_clip() -> None:
    first = _rich_one_clip(name="walk.npz")
    second = _rich_one_clip(name="turn.npz", frames=2, value_offset=10.0)

    with pytest.raises(ValueError, match="clip 1.*at least 3 frames"):
        ReferenceMotion.from_clips([first, second])


def test_reference_motion_from_clips_rejects_selected_device_mismatch() -> None:
    clip = _rich_one_clip(name="walk.npz")

    with pytest.raises(ValueError, match="clip 0.*device"):
        ReferenceMotion.from_clips([clip], device="cuda")


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


def test_reference_motion_exposes_axis_names_and_lazy_utf8_clip_identity() -> None:
    clip_name = "walking-测试.npz"
    encoded_name = clip_name.encode("utf-8")
    motion = ReferenceMotion(
        fps=50.0,
        root_pos=torch.zeros(3, 3),
        root_rot=_identity_root_rot(3),
        dof_pos=torch.zeros(3, 2),
        body_pos=torch.zeros(3, 2, 3),
        dof_names=("left_hip", "right_hip"),
        body_names=("pelvis", "torso"),
        clip_starts=torch.tensor([0]),
        clip_lengths=torch.tensor([3]),
        clip_fps=torch.tensor([50.0]),
        clip_name_bytes=torch.tensor(list(encoded_name), dtype=torch.uint8),
        clip_name_offsets=torch.tensor([0, len(encoded_name)]),
    )

    assert motion.dof_names == ("left_hip", "right_hip")
    assert motion.body_names == ("pelvis", "torso")
    assert motion.clip_name_bytes.device.type == "cpu"
    assert motion.clip_name_offsets.device.type == "cpu"
    assert motion.clip_name(0) == clip_name


def test_reference_motion_rejects_axis_names_that_are_not_valid_utf8() -> None:
    with pytest.raises(ValueError, match="dof_names must contain valid UTF-8"):
        ReferenceMotion(
            root_pos=torch.zeros(3, 3),
            root_rot=_identity_root_rot(3),
            dof_pos=torch.zeros(3, 2),
            dof_names=(chr(0xD800), "right_hip"),
        )


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
            foot_contacts=torch.tensor([[1.0, 0.0], [float("nan"), 0.5], [0.0, 1.0]]),
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


def test_reference_motion_state_rejects_shape_dtype_device_and_finite_mismatches() -> (
    None
):
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
