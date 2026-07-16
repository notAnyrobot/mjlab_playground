"""ReferenceMotion public contract tests."""

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
        "body_rot": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(frames, 2, 1),
        "body_lin_vel": frame_values[:, None, None].repeat(1, 2, 3),
        "body_ang_vel": frame_values[:, None, None].repeat(1, 2, 3),
        "body_contacts": torch.arange(frames)[:, None].repeat(1, 2) % 2 == 0,
        "dof_names": ("left_hip", "right_hip"),
        "body_names": ("pelvis", "torso"),
    }
    fields.update(overrides)
    return ReferenceMotion(**fields)


def test_reference_motion_state_is_deprecated_compatibility_alias() -> None:
    DeprecatedReferenceMotionState = _motion_loader.ReferenceMotionState
    from mjlab_playground.motion_lib import ReferenceMotion as ExportedReferenceMotion
    from mjlab_playground.motion_lib import (
        ReferenceMotionState as ExportedReferenceMotionState,
    )

    assert DeprecatedReferenceMotionState is ReferenceMotion
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


def test_reference_frame_direct_construction_preserves_fields_without_validation() -> (
    None
):
    root_pos = torch.tensor([[float("nan")]], dtype=torch.float64)
    root_rot = torch.ones(2, dtype=torch.int64)
    dof_pos = torch.zeros(3, dtype=torch.float32)
    body_contacts = "unvalidated contacts"

    frame = ReferenceFrame(
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        body_contacts=body_contacts,
    )

    assert frame.root_pos is root_pos
    assert frame.root_rot is root_rot
    assert frame.dof_pos is dof_pos
    assert frame.body_contacts is body_contacts


def test_reference_motion_defaults_to_30_fps_and_preserves_required_tensors() -> None:
    root_pos = torch.zeros(3, 3)
    root_rot = _identity_root_rot(3)
    dof_pos = torch.zeros(3, 29)

    state = ReferenceMotion(root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos)

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


def test_reference_motion_accepts_full_reference_motion_fields() -> None:
    state = ReferenceMotion(
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


def test_reference_motion_get_frame_extracts_single_reference_frame() -> None:
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

    frame = motion.get_frame(2)

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
def test_reference_motion_get_frame_rejects_out_of_range_indices(
    frame_index: int,
) -> None:
    motion = ReferenceMotion(
        root_pos=torch.zeros(4, 3),
        root_rot=_identity_root_rot(4),
        dof_pos=torch.zeros(4, 3),
    )

    with pytest.raises(IndexError, match=f"frame index {frame_index} out of range"):
        motion.get_frame(frame_index)


def test_reference_motion_does_not_expose_old_frame_method() -> None:
    assert not hasattr(ReferenceMotion, "frame")


def test_reference_motion_does_not_accept_or_expose_display_name() -> None:
    fields = {
        "root_pos": torch.zeros(3, 3),
        "root_rot": _identity_root_rot(3),
        "dof_pos": torch.zeros(3, 2),
    }

    motion = ReferenceMotion(**fields)

    assert not hasattr(motion, "display_name")
    with pytest.raises(TypeError, match="unexpected keyword argument 'display_name'"):
        ReferenceMotion(**fields, display_name="walk")


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
        fps=60.0,
        clip_starts=torch.tensor([0]),
        clip_lengths=torch.tensor([2]),
        clip_fps=torch.tensor([60.0]),
    )

    assert motion.name == "walk.npz"
    assert not hasattr(motion, "display_name")
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


def test_reference_motion_from_clips_delegates_dtype_handling_to_torch_cat() -> None:
    first = _rich_one_clip(name="walk.npz")
    second = _rich_one_clip(name="turn.npz", value_offset=10.0)
    float_fields = (
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
    second = dataclasses.replace(
        second,
        **{
            field_name: getattr(second, field_name).to(torch.float64)
            for field_name in float_fields
        },
    )

    assembled = ReferenceMotion.from_clips([first, second])

    assert assembled.root_pos.dtype == torch.float64
    torch.testing.assert_close(
        assembled.root_pos,
        torch.cat([first.root_pos, second.root_pos]),
    )


def test_reference_motion_from_clips_does_not_revalidate_quaternion_values() -> None:
    first = _rich_one_clip(name="walk.npz")
    second = dataclasses.replace(
        _rich_one_clip(name="turn.npz", value_offset=10.0),
        root_rot=torch.full((3, 4), 2.0),
        body_rot=torch.full((3, 2, 4), 3.0),
    )

    assembled = ReferenceMotion.from_clips([first, second])

    torch.testing.assert_close(
        assembled.root_rot,
        torch.cat([first.root_rot, second.root_rot]),
    )
    torch.testing.assert_close(
        assembled.body_rot,
        torch.cat([first.body_rot, second.body_rot]),
    )


def test_reference_motion_from_clips_does_not_scan_tensor_finiteness() -> None:
    first = _rich_one_clip(name="walk.npz")
    second_root_lin_vel = torch.zeros(3, 3)
    second_root_lin_vel[1, 0] = float("nan")
    second = dataclasses.replace(
        _rich_one_clip(name="turn.npz", value_offset=10.0),
        root_lin_vel=second_root_lin_vel,
    )

    assembled = ReferenceMotion.from_clips([first, second])

    assert torch.isnan(assembled.root_lin_vel[4, 0])


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
    ("metadata", "message"),
    [
        pytest.param(
            {"clip_starts": torch.tensor([0])},
            "operational clip metadata must be provided together",
            id="partial",
        ),
        pytest.param(
            {
                "clip_starts": torch.tensor([1]),
                "clip_lengths": torch.tensor([3]),
                "clip_fps": torch.tensor([50.0]),
            },
            "clip_starts must describe one clip starting at frame 0",
            id="start",
        ),
        pytest.param(
            {
                "clip_starts": torch.tensor([0]),
                "clip_lengths": torch.tensor([2]),
                "clip_fps": torch.tensor([50.0]),
            },
            "clip_lengths must match packed frame count 3",
            id="length",
        ),
        pytest.param(
            {
                "clip_starts": torch.tensor([0]),
                "clip_lengths": torch.tensor([3]),
                "clip_fps": torch.tensor([60.0]),
            },
            "clip_fps must match fps 50.0",
            id="fps",
        ),
        pytest.param(
            {
                "clip_starts": torch.tensor([0]),
                "clip_lengths": torch.tensor([3]),
                "clip_fps": torch.tensor([50.0]),
                "clip_name_offsets": torch.tensor([0, 3]),
            },
            "clip_name_offsets must span clip_name_bytes",
            id="name-span",
        ),
    ],
)
def test_reference_motion_from_clips_rejects_incoherent_operational_metadata(
    metadata: dict[str, torch.Tensor],
    message: str,
) -> None:
    encoded_name = torch.tensor(list(b"walk.npz"), dtype=torch.uint8)
    fields = {
        "clip_name_bytes": encoded_name,
        "clip_name_offsets": torch.tensor([0, encoded_name.numel()]),
    }
    fields.update(metadata)
    clip = dataclasses.replace(_rich_one_clip(name="package.npz"), **fields)

    with pytest.raises(ValueError, match=rf"clip 0.*{message}"):
        ReferenceMotion.from_clips([clip])


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("fps", "fps"),
        ("dof_names", "dof_names"),
        ("body_names", "body_names"),
        ("body_contacts", "body_contacts presence"),
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
    elif mismatch == "dof_names":
        second = dataclasses.replace(second, dof_names=("right_hip", "left_hip"))
    elif mismatch == "body_names":
        second = dataclasses.replace(second, body_names=("torso", "pelvis"))
    elif mismatch == "body_contacts":
        second = dataclasses.replace(second, body_contacts=None)
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


def test_reference_motion_from_clips_delegates_shape_incompatibility_to_torch_cat() -> (
    None
):
    first = _rich_one_clip(name="walk.npz")
    second = dataclasses.replace(
        _rich_one_clip(name="turn.npz", value_offset=10.0),
        root_pos=torch.zeros(3, 4),
    )

    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
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


def test_reference_motion_accepts_valid_multi_clip_package_metadata() -> None:
    state = ReferenceMotion(
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


def test_reference_motion_accepts_source_foot_contacts() -> None:
    foot_contacts = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        dtype=torch.float32,
    )

    state = ReferenceMotion(
        root_pos=torch.zeros(3, 3),
        root_rot=_identity_root_rot(3),
        dof_pos=torch.zeros(3, 29),
        foot_contacts=foot_contacts,
    )

    assert state.foot_contacts is foot_contacts


@pytest.mark.parametrize("num_frames", [1, 2])
def test_reference_motion_accepts_small_sampled_batches_without_package_metadata(
    num_frames: int,
) -> None:
    state = ReferenceMotion(
        root_pos=torch.zeros(num_frames, 3),
        root_rot=_identity_root_rot(num_frames),
        dof_pos=torch.zeros(num_frames, 29),
    )

    assert state.root_pos.shape[0] == num_frames
    assert state.clip_starts is None
    assert state.clip_lengths is None
    assert state.clip_fps is None


def test_reference_motion_direct_construction_preserves_fields_without_validation() -> (
    None
):
    root_pos = torch.tensor([float("nan")], dtype=torch.float64)
    root_rot = torch.ones(2, dtype=torch.int64)
    dof_pos = torch.zeros(3, 7, dtype=torch.float32)
    clip_starts = torch.tensor([99.5])
    dof_names = (chr(0xD800),)

    motion = ReferenceMotion(
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        fps=-29.97,
        clip_starts=clip_starts,
        dof_names=dof_names,
    )

    assert motion.root_pos is root_pos
    assert motion.root_rot is root_rot
    assert motion.dof_pos is dof_pos
    assert motion.fps == -29.97
    assert motion.clip_starts is clip_starts
    assert motion.dof_names is dof_names


@pytest.mark.parametrize("value_class", [ReferenceFrame, ReferenceMotion])
def test_reference_values_remain_frozen_and_keyword_only(value_class: type) -> None:
    values = (
        torch.zeros(3),
        torch.ones(4),
        torch.zeros(2),
    )

    with pytest.raises(TypeError):
        value_class(*values)

    value = value_class(root_pos=values[0], root_rot=values[1], dof_pos=values[2])
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.root_pos = torch.ones(3)


def test_motion_state_alias_is_not_exported_from_module_file() -> None:
    assert not hasattr(_motion_loader, "MotionState")
