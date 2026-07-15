from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from mjlab_playground.motion_lib._reference_motion_npz_schema import (
    REFERENCE_MOTION_NPZ_V1,
)

MotionFormat = Literal["pyroki", "proto", "mjlab"]


def _validate_integer_fps(fps: float, *, field_name: str = "fps") -> float:
    value = float(fps)
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{field_name} must be positive and finite")
    if not value.is_integer():
        raise ValueError(f"{field_name} must be integer-valued")
    return value


def _validate_float_tensor(
    tensor: torch.Tensor,
    *,
    field_name: str,
    shape_suffix: tuple[int | None, ...],
    frame_count: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor")
    expected_ndim = len(shape_suffix) + 1
    if tensor.ndim != expected_ndim:
        shape_text = _format_tensor_shape(shape_suffix)
        raise ValueError(
            f"{field_name} must have shape ({shape_text}), got {tuple(tensor.shape)}"
        )
    if frame_count is not None and tensor.shape[0] != frame_count:
        raise ValueError(
            f"{field_name} must share frame count {frame_count}, got {tensor.shape[0]}"
        )
    for axis, expected_dim in enumerate(shape_suffix, start=1):
        if expected_dim is not None and tensor.shape[axis] != expected_dim:
            shape_text = _format_tensor_shape(shape_suffix)
            raise ValueError(
                f"{field_name} must have shape ({shape_text}), got {tuple(tensor.shape)}"
            )
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{field_name} must be a floating point tensor")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{field_name} must share dtype {dtype}, got {tensor.dtype}")
    if device is not None and tensor.device != device:
        raise ValueError(
            f"{field_name} must share device {device}, got {tensor.device}"
        )
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{field_name} contains non-finite values")


def _format_tensor_shape(shape_suffix: tuple[int | None, ...]) -> str:
    return ", ".join("D" if dim is None else str(dim) for dim in ("T", *shape_suffix))


def _validate_quaternion_tensor(tensor: torch.Tensor, *, field_name: str) -> None:
    quat_norms = torch.linalg.norm(tensor, dim=-1)
    expected = torch.ones_like(quat_norms)
    if not torch.allclose(quat_norms, expected, rtol=1e-4, atol=1e-4):
        raise ValueError(f"{field_name} quaternions must be normalized")


@dataclass(frozen=True, kw_only=True)
class ReferenceFrame:
    """One reference timestep in robot coordinates."""

    root_pos: torch.Tensor
    root_rot: torch.Tensor
    dof_pos: torch.Tensor
    root_lin_vel: torch.Tensor | None = None
    root_ang_vel: torch.Tensor | None = None
    dof_vel: torch.Tensor | None = None
    body_pos: torch.Tensor | None = None
    body_rot: torch.Tensor | None = None
    body_lin_vel: torch.Tensor | None = None
    body_ang_vel: torch.Tensor | None = None
    body_contacts: torch.Tensor | None = None
    foot_contacts: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _validate_frame_float_tensor(
            self.root_pos, field_name="root_pos", shape_suffix=(3,)
        )
        dtype = self.root_pos.dtype
        device = self.root_pos.device
        _validate_frame_float_tensor(
            self.root_rot,
            field_name="root_rot",
            shape_suffix=(4,),
            dtype=dtype,
            device=device,
        )
        _validate_quaternion_tensor(self.root_rot, field_name="root_rot")
        _validate_frame_float_tensor(
            self.dof_pos,
            field_name="dof_pos",
            shape_suffix=(None,),
            dtype=dtype,
            device=device,
        )
        self._validate_optional_float_field("root_lin_vel", self.root_lin_vel, (3,))
        self._validate_optional_float_field("root_ang_vel", self.root_ang_vel, (3,))
        self._validate_optional_float_field(
            "dof_vel", self.dof_vel, (self.dof_pos.shape[0],)
        )
        self._validate_optional_float_field("body_pos", self.body_pos, (None, 3))
        self._validate_optional_float_field("body_rot", self.body_rot, (None, 4))
        if self.body_rot is not None:
            _validate_quaternion_tensor(self.body_rot, field_name="body_rot")
        self._validate_optional_float_field(
            "body_lin_vel", self.body_lin_vel, (None, 3)
        )
        self._validate_optional_float_field(
            "body_ang_vel", self.body_ang_vel, (None, 3)
        )
        self._validate_optional_contacts()
        self._validate_optional_foot_contacts()

    def _validate_optional_float_field(
        self,
        field_name: str,
        value: torch.Tensor | None,
        shape_suffix: tuple[int | None, ...],
    ) -> None:
        if value is None:
            return
        _validate_frame_float_tensor(
            value,
            field_name=field_name,
            shape_suffix=shape_suffix,
            dtype=self.root_pos.dtype,
            device=self.root_pos.device,
        )

    def _validate_optional_contacts(self) -> None:
        if self.body_contacts is None:
            return
        if not isinstance(self.body_contacts, torch.Tensor):
            raise TypeError("body_contacts must be a torch.Tensor")
        if self.body_contacts.ndim != 1:
            raise ValueError(
                f"body_contacts must have shape (B,), got {tuple(self.body_contacts.shape)}"
            )
        if self.body_contacts.device != self.root_pos.device:
            raise ValueError(
                f"body_contacts must share device {self.root_pos.device}, "
                f"got {self.body_contacts.device}"
            )
        if (
            torch.is_floating_point(self.body_contacts)
            and not torch.isfinite(self.body_contacts).all().item()
        ):
            raise ValueError("body_contacts contains non-finite values")

    def _validate_optional_foot_contacts(self) -> None:
        if self.foot_contacts is None:
            return
        if not isinstance(self.foot_contacts, torch.Tensor):
            raise TypeError("foot_contacts must be a torch.Tensor")
        if self.foot_contacts.ndim != 1 or self.foot_contacts.shape[0] != 2:
            raise ValueError(
                f"foot_contacts must have shape (2,), got {tuple(self.foot_contacts.shape)}"
            )
        if self.foot_contacts.device != self.root_pos.device:
            raise ValueError(
                f"foot_contacts must share device {self.root_pos.device}, "
                f"got {self.foot_contacts.device}"
            )
        if torch.is_floating_point(self.foot_contacts):
            if not torch.isfinite(self.foot_contacts).all().item():
                raise ValueError("foot_contacts contains non-finite values")
            return
        if self.foot_contacts.dtype != torch.bool:
            raise TypeError("foot_contacts must be a floating point or bool tensor")


def _validate_frame_float_tensor(
    tensor: torch.Tensor,
    *,
    field_name: str,
    shape_suffix: tuple[int | None, ...],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor")
    if tensor.ndim != len(shape_suffix):
        shape_text = _format_frame_tensor_shape(shape_suffix)
        raise ValueError(
            f"{field_name} must have shape ({shape_text}), got {tuple(tensor.shape)}"
        )
    for axis, expected_dim in enumerate(shape_suffix):
        if expected_dim is not None and tensor.shape[axis] != expected_dim:
            shape_text = _format_frame_tensor_shape(shape_suffix)
            raise ValueError(
                f"{field_name} must have shape ({shape_text}), got {tuple(tensor.shape)}"
            )
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{field_name} must be a floating point tensor")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{field_name} must share dtype {dtype}, got {tensor.dtype}")
    if device is not None and tensor.device != device:
        raise ValueError(
            f"{field_name} must share device {device}, got {tensor.device}"
        )
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{field_name} contains non-finite values")


def _format_frame_tensor_shape(shape_suffix: tuple[int | None, ...]) -> str:
    return ", ".join("D" if dim is None else str(dim) for dim in shape_suffix)


@dataclass(frozen=True, kw_only=True)
class ReferenceMotion:
    """Tensor-backed reference motion over packed frame tensors."""

    root_pos: torch.Tensor
    root_rot: torch.Tensor
    dof_pos: torch.Tensor
    name: str | None = None
    display_name: str | None = None
    fps: float = 30.0
    root_lin_vel: torch.Tensor | None = None
    root_ang_vel: torch.Tensor | None = None
    dof_vel: torch.Tensor | None = None
    body_pos: torch.Tensor | None = None
    body_rot: torch.Tensor | None = None
    body_lin_vel: torch.Tensor | None = None
    body_ang_vel: torch.Tensor | None = None
    body_contacts: torch.Tensor | None = None
    foot_contacts: torch.Tensor | None = None
    clip_starts: torch.Tensor | None = None
    clip_lengths: torch.Tensor | None = None
    clip_fps: torch.Tensor | None = None
    clip_name_bytes: torch.Tensor | None = None
    clip_name_offsets: torch.Tensor | None = None
    dof_names: tuple[str, ...] | None = None
    body_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fps", _validate_integer_fps(self.fps))

        _validate_float_tensor(self.root_pos, field_name="root_pos", shape_suffix=(3,))
        frame_count = self.root_pos.shape[0]
        if frame_count < 1:
            raise ValueError("ReferenceMotion requires at least one frame")
        dtype = self.root_pos.dtype
        device = self.root_pos.device

        _validate_float_tensor(
            self.root_rot,
            field_name="root_rot",
            shape_suffix=(4,),
            frame_count=frame_count,
            dtype=dtype,
            device=device,
        )
        _validate_quaternion_tensor(self.root_rot, field_name="root_rot")
        _validate_float_tensor(
            self.dof_pos,
            field_name="dof_pos",
            shape_suffix=(None,),
            frame_count=frame_count,
            dtype=dtype,
            device=device,
        )

        self._validate_optional_float_field("root_lin_vel", self.root_lin_vel, (3,))
        self._validate_optional_float_field("root_ang_vel", self.root_ang_vel, (3,))
        self._validate_optional_float_field(
            "dof_vel",
            self.dof_vel,
            (self.dof_pos.shape[1],),
        )
        self._validate_optional_float_field("body_pos", self.body_pos, (None, 3))
        self._validate_optional_float_field("body_rot", self.body_rot, (None, 4))
        if self.body_rot is not None:
            _validate_quaternion_tensor(self.body_rot, field_name="body_rot")
        self._validate_optional_float_field(
            "body_lin_vel", self.body_lin_vel, (None, 3)
        )
        self._validate_optional_float_field(
            "body_ang_vel", self.body_ang_vel, (None, 3)
        )
        self._validate_optional_contacts()
        self._validate_optional_foot_contacts()
        self._validate_package_metadata()
        self._validate_axis_names()
        self._validate_clip_names()

    @classmethod
    def from_frames(
        cls,
        frames: list[ReferenceFrame],
        *,
        name: str | None = None,
        display_name: str | None = None,
        fps: float = 30.0,
        clip_starts: torch.Tensor | None = None,
        clip_lengths: torch.Tensor | None = None,
        clip_fps: torch.Tensor | None = None,
        clip_name_bytes: torch.Tensor | None = None,
        clip_name_offsets: torch.Tensor | None = None,
        dof_names: tuple[str, ...] | None = None,
        body_names: tuple[str, ...] | None = None,
    ) -> "ReferenceMotion":
        """Construct a tensor-backed reference motion from frame values."""
        if not frames:
            raise ValueError("ReferenceMotion.from_frames requires at least one frame")
        for frame in frames:
            if not isinstance(frame, ReferenceFrame):
                raise TypeError("frames must contain ReferenceFrame objects")
        return cls(
            name=name,
            display_name=display_name,
            fps=fps,
            root_pos=torch.stack([frame.root_pos for frame in frames]),
            root_rot=torch.stack([frame.root_rot for frame in frames]),
            dof_pos=torch.stack([frame.dof_pos for frame in frames]),
            root_lin_vel=cls._stack_optional_frame_field(frames, "root_lin_vel"),
            root_ang_vel=cls._stack_optional_frame_field(frames, "root_ang_vel"),
            dof_vel=cls._stack_optional_frame_field(frames, "dof_vel"),
            body_pos=cls._stack_optional_frame_field(frames, "body_pos"),
            body_rot=cls._stack_optional_frame_field(frames, "body_rot"),
            body_lin_vel=cls._stack_optional_frame_field(frames, "body_lin_vel"),
            body_ang_vel=cls._stack_optional_frame_field(frames, "body_ang_vel"),
            body_contacts=cls._stack_optional_frame_field(frames, "body_contacts"),
            foot_contacts=cls._stack_optional_frame_field(frames, "foot_contacts"),
            clip_starts=clip_starts,
            clip_lengths=clip_lengths,
            clip_fps=clip_fps,
            clip_name_bytes=clip_name_bytes,
            clip_name_offsets=clip_name_offsets,
            dof_names=dof_names,
            body_names=body_names,
        )

    @classmethod
    def from_clips(
        cls,
        clips: Iterable["ReferenceMotion"],
        *,
        device: str | torch.device = "cpu",
    ) -> "ReferenceMotion":
        """Perform reference motion assembly over rich reference clips."""
        clip_list = list(clips)
        if not clip_list:
            raise ValueError("ReferenceMotion.from_clips requires at least one clip")

        selected_device = torch.device(device)
        clip_name_parts: list[torch.Tensor] = []
        clip_name_offsets = [0]
        clip_lengths: list[int] = []
        first = clip_list[0]

        def validate_tensor(
            clip: "ReferenceMotion",
            clip_id: int,
            field_name: str,
            shape_suffix: tuple[int | None, ...],
            *,
            frame_count: int | None = None,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor:
            value = getattr(clip, field_name)
            if value is None:
                raise ValueError(
                    f"clip {clip_id} requires rich reference field {field_name}"
                )
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"clip {clip_id} field {field_name} must be a torch.Tensor"
                )
            try:
                _validate_float_tensor(
                    value,
                    field_name=field_name,
                    shape_suffix=shape_suffix,
                    frame_count=frame_count,
                    dtype=dtype,
                    device=selected_device,
                )
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"clip {clip_id} {exc}") from exc
            return value

        for clip_id, clip in enumerate(clip_list):
            if not isinstance(clip, cls):
                raise TypeError(
                    f"clip {clip_id} must be a ReferenceMotion, got {type(clip).__name__}"
                )

            root_pos = validate_tensor(clip, clip_id, "root_pos", (3,))
            frame_count = root_pos.shape[0]
            dtype = root_pos.dtype
            root_rot = validate_tensor(
                clip,
                clip_id,
                "root_rot",
                (4,),
                frame_count=frame_count,
                dtype=dtype,
            )
            dof_pos = validate_tensor(
                clip,
                clip_id,
                "dof_pos",
                (None,),
                frame_count=frame_count,
                dtype=dtype,
            )
            validate_tensor(
                clip,
                clip_id,
                "root_lin_vel",
                (3,),
                frame_count=frame_count,
                dtype=dtype,
            )
            validate_tensor(
                clip,
                clip_id,
                "root_ang_vel",
                (3,),
                frame_count=frame_count,
                dtype=dtype,
            )
            validate_tensor(
                clip,
                clip_id,
                "dof_vel",
                (dof_pos.shape[1],),
                frame_count=frame_count,
                dtype=dtype,
            )
            body_pos = validate_tensor(
                clip,
                clip_id,
                "body_pos",
                (None, 3),
                frame_count=frame_count,
                dtype=dtype,
            )
            body_count = body_pos.shape[1]
            body_rot = validate_tensor(
                clip,
                clip_id,
                "body_rot",
                (body_count, 4),
                frame_count=frame_count,
                dtype=dtype,
            )
            validate_tensor(
                clip,
                clip_id,
                "body_lin_vel",
                (body_count, 3),
                frame_count=frame_count,
                dtype=dtype,
            )
            validate_tensor(
                clip,
                clip_id,
                "body_ang_vel",
                (body_count, 3),
                frame_count=frame_count,
                dtype=dtype,
            )
            if clip.foot_contacts is not None:
                raise ValueError(
                    f"clip {clip_id} contains source foot_contacts; "
                    "assembly accepts body_contacts only"
                )
            if clip.root_pos.shape[0] < 3:
                raise ValueError(
                    f"clip {clip_id} rich reference clip must contain at least 3 frames"
                )
            if clip.dof_names is None:
                raise ValueError(f"clip {clip_id} requires dof_names")
            if clip.body_names is None:
                raise ValueError(f"clip {clip_id} requires body_names")
            if clip.body_contacts is not None:
                if clip.body_contacts.device != selected_device:
                    raise ValueError(
                        f"clip {clip_id} body_contacts device must match selected "
                        f"device {selected_device}, got {clip.body_contacts.device}"
                    )
            try:
                _validate_quaternion_tensor(root_rot, field_name="root_rot")
                _validate_quaternion_tensor(body_rot, field_name="body_rot")
            except ValueError as exc:
                raise ValueError(f"clip {clip_id} {exc}") from exc

            if clip_id > 0:
                if clip.fps != first.fps:
                    raise ValueError(
                        f"clip {clip_id} fps must match clip 0 fps "
                        f"{first.fps}, got {clip.fps}"
                    )
                if clip.root_pos.dtype != first.root_pos.dtype:
                    raise ValueError(
                        f"clip {clip_id} dtype must match clip 0 dtype "
                        f"{first.root_pos.dtype}, got {clip.root_pos.dtype}"
                    )
                if clip.dof_pos.shape[1] != first.dof_pos.shape[1]:
                    raise ValueError(
                        f"clip {clip_id} DOF count must match clip 0 DOF count "
                        f"{first.dof_pos.shape[1]}, got {clip.dof_pos.shape[1]}"
                    )
                assert clip.body_pos is not None
                assert first.body_pos is not None
                if clip.body_pos.shape[1] != first.body_pos.shape[1]:
                    raise ValueError(
                        f"clip {clip_id} body count must match clip 0 body count "
                        f"{first.body_pos.shape[1]}, got {clip.body_pos.shape[1]}"
                    )
                if clip.dof_names != first.dof_names:
                    raise ValueError(
                        f"clip {clip_id} dof_names must match clip 0 in order"
                    )
                if clip.body_names != first.body_names:
                    raise ValueError(
                        f"clip {clip_id} body_names must match clip 0 in order"
                    )
                if (clip.body_contacts is None) != (first.body_contacts is None):
                    raise ValueError(
                        f"clip {clip_id} body_contacts presence must match clip 0"
                    )
                if (
                    clip.body_contacts is not None
                    and first.body_contacts is not None
                    and clip.body_contacts.dtype != first.body_contacts.dtype
                ):
                    raise ValueError(
                        f"clip {clip_id} body_contacts dtype must match clip 0 dtype "
                        f"{first.body_contacts.dtype}, got {clip.body_contacts.dtype}"
                    )
            if clip.clip_starts is not None:
                if clip.clip_starts.numel() != 1:
                    raise ValueError(
                        f"clip {clip_id} is already multi-clip; "
                        "ReferenceMotion.from_clips accepts one-clip inputs"
                    )
                if clip.clip_name_bytes is None or clip.clip_name_offsets is None:
                    raise ValueError(
                        f"clip {clip_id} explicit metadata requires a clip identity"
                    )
                encoded_name = clip.clip_name_bytes
            else:
                if not clip.name:
                    raise ValueError(f"clip {clip_id} requires a clip identity")
                try:
                    encoded_name = torch.tensor(
                        list(clip.name.encode("utf-8")), dtype=torch.uint8
                    )
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        f"clip {clip_id} name must be valid UTF-8"
                    ) from exc
            clip_name_parts.append(encoded_name)
            clip_name_offsets.append(clip_name_offsets[-1] + encoded_name.numel())
            clip_lengths.append(clip.root_pos.shape[0])

        contacts_present = [clip.body_contacts is not None for clip in clip_list]
        lengths = torch.tensor(clip_lengths, dtype=torch.int64, device=selected_device)
        starts = torch.empty_like(lengths)
        starts[0] = 0
        if len(clip_list) > 1:
            starts[1:] = torch.cumsum(lengths[:-1], dim=0)

        def concatenate(field_name: str) -> torch.Tensor:
            values = [getattr(clip, field_name) for clip in clip_list]
            assert all(isinstance(value, torch.Tensor) for value in values)
            return torch.cat(values, dim=0)

        packed_body_contacts: torch.Tensor | None = None
        if all(contacts_present):
            contact_values = [clip.body_contacts for clip in clip_list]
            assert all(value is not None for value in contact_values)
            packed_body_contacts = torch.cat(
                [value for value in contact_values if value is not None], dim=0
            )
        return cls(
            fps=first.fps,
            root_pos=concatenate("root_pos"),
            root_rot=concatenate("root_rot"),
            dof_pos=concatenate("dof_pos"),
            root_lin_vel=concatenate("root_lin_vel"),
            root_ang_vel=concatenate("root_ang_vel"),
            dof_vel=concatenate("dof_vel"),
            body_pos=concatenate("body_pos"),
            body_rot=concatenate("body_rot"),
            body_lin_vel=concatenate("body_lin_vel"),
            body_ang_vel=concatenate("body_ang_vel"),
            body_contacts=packed_body_contacts,
            clip_starts=starts,
            clip_lengths=lengths,
            clip_fps=torch.tensor(
                [clip.fps for clip in clip_list],
                dtype=torch.float32,
                device=selected_device,
            ),
            clip_name_bytes=torch.cat(clip_name_parts),
            clip_name_offsets=torch.tensor(clip_name_offsets, dtype=torch.int64),
            dof_names=first.dof_names,
            body_names=first.body_names,
        )

    @staticmethod
    def _stack_optional_frame_field(
        frames: list[ReferenceFrame],
        field_name: str,
    ) -> torch.Tensor | None:
        values = [getattr(frame, field_name) for frame in frames]
        present = [value is not None for value in values]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"{field_name} must be present on every frame or none")
        return torch.stack([value for value in values if value is not None])

    def frame(self, index: int) -> ReferenceFrame:
        """Return one packed reference frame by index."""
        if not isinstance(index, int):
            raise TypeError("frame index must be an int")
        frame_count = self.root_pos.shape[0]
        if index < 0 or index >= frame_count:
            raise IndexError(
                f"frame index {index} out of range for {frame_count} packed frames"
            )
        return ReferenceFrame(
            root_pos=self.root_pos[index],
            root_rot=self.root_rot[index],
            dof_pos=self.dof_pos[index],
            root_lin_vel=self.root_lin_vel[index]
            if self.root_lin_vel is not None
            else None,
            root_ang_vel=self.root_ang_vel[index]
            if self.root_ang_vel is not None
            else None,
            dof_vel=self.dof_vel[index] if self.dof_vel is not None else None,
            body_pos=self.body_pos[index] if self.body_pos is not None else None,
            body_rot=self.body_rot[index] if self.body_rot is not None else None,
            body_lin_vel=self.body_lin_vel[index]
            if self.body_lin_vel is not None
            else None,
            body_ang_vel=self.body_ang_vel[index]
            if self.body_ang_vel is not None
            else None,
            body_contacts=self.body_contacts[index]
            if self.body_contacts is not None
            else None,
            foot_contacts=self.foot_contacts[index]
            if self.foot_contacts is not None
            else None,
        )

    def _validate_optional_float_field(
        self,
        field_name: str,
        value: torch.Tensor | None,
        shape_suffix: tuple[int | None, ...],
    ) -> None:
        if value is None:
            return
        _validate_float_tensor(
            value,
            field_name=field_name,
            shape_suffix=shape_suffix,
            frame_count=self.root_pos.shape[0],
            dtype=self.root_pos.dtype,
            device=self.root_pos.device,
        )

    def _validate_optional_contacts(self) -> None:
        if self.body_contacts is None:
            return
        if not isinstance(self.body_contacts, torch.Tensor):
            raise TypeError("body_contacts must be a torch.Tensor")
        if self.body_contacts.ndim != 2:
            raise ValueError(
                f"body_contacts must have shape (T, B), got {tuple(self.body_contacts.shape)}"
            )
        if self.body_contacts.shape[0] != self.root_pos.shape[0]:
            raise ValueError(
                "body_contacts must share frame count "
                f"{self.root_pos.shape[0]}, got {self.body_contacts.shape[0]}"
            )
        if self.body_contacts.device != self.root_pos.device:
            raise ValueError(
                f"body_contacts must share device {self.root_pos.device}, "
                f"got {self.body_contacts.device}"
            )
        if (
            torch.is_floating_point(self.body_contacts)
            and not torch.isfinite(self.body_contacts).all().item()
        ):
            raise ValueError("body_contacts contains non-finite values")

    def _validate_optional_foot_contacts(self) -> None:
        if self.foot_contacts is None:
            return
        if not isinstance(self.foot_contacts, torch.Tensor):
            raise TypeError("foot_contacts must be a torch.Tensor")
        if self.foot_contacts.ndim != 2 or self.foot_contacts.shape[1] != 2:
            raise ValueError(
                "foot_contacts must have shape "
                f"(T, 2), got {tuple(self.foot_contacts.shape)}"
            )
        if self.foot_contacts.shape[0] != self.root_pos.shape[0]:
            raise ValueError(
                "foot_contacts must share frame count "
                f"{self.root_pos.shape[0]}, got {self.foot_contacts.shape[0]}"
            )
        if self.foot_contacts.device != self.root_pos.device:
            raise ValueError(
                f"foot_contacts must share device {self.root_pos.device}, "
                f"got {self.foot_contacts.device}"
            )
        if torch.is_floating_point(self.foot_contacts):
            if not torch.isfinite(self.foot_contacts).all().item():
                raise ValueError("foot_contacts contains non-finite values")
            return
        if self.foot_contacts.dtype != torch.bool:
            raise TypeError("foot_contacts must be a floating point or bool tensor")

    def _validate_package_metadata(self) -> None:
        metadata = (self.clip_starts, self.clip_lengths, self.clip_fps)
        if all(value is None for value in metadata):
            return

        if any(value is None for value in metadata):
            raise ValueError(
                "clip_starts, clip_lengths, and clip_fps must be provided together"
            )
        assert self.clip_starts is not None
        assert self.clip_lengths is not None
        assert self.clip_fps is not None

        if self.clip_starts.ndim != 1:
            raise ValueError("clip_starts must be 1D")
        if self.clip_lengths.ndim != 1:
            raise ValueError("clip_lengths must be 1D")
        if self.clip_fps.ndim != 1:
            raise ValueError("clip_fps must be 1D")
        if not (
            self.clip_starts.shape == self.clip_lengths.shape == self.clip_fps.shape
        ):
            raise ValueError("clip metadata tensors must have matching lengths")
        if self.clip_starts.numel() == 0:
            raise ValueError("clip metadata must contain at least one clip")
        if self.clip_starts.device != self.root_pos.device:
            raise ValueError(
                f"clip_starts must share device {self.root_pos.device}, "
                f"got {self.clip_starts.device}"
            )
        if self.clip_lengths.device != self.root_pos.device:
            raise ValueError(
                f"clip_lengths must share device {self.root_pos.device}, "
                f"got {self.clip_lengths.device}"
            )
        if self.clip_fps.device != self.root_pos.device:
            raise ValueError(
                f"clip_fps must share device {self.root_pos.device}, got {self.clip_fps.device}"
            )
        integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
        if self.clip_starts.dtype not in integer_dtypes:
            raise TypeError("clip_starts must be an integer tensor")
        if self.clip_lengths.dtype not in integer_dtypes:
            raise TypeError("clip_lengths must be an integer tensor")
        if not torch.is_floating_point(self.clip_fps):
            raise TypeError("clip_fps must be a floating point tensor")
        if (self.clip_starts < 0).any().item():
            raise ValueError("clip_starts must be non-negative")
        if (self.clip_lengths <= 0).any().item():
            raise ValueError("clip_lengths must be positive")
        if (
            not torch.isfinite(self.clip_fps).all().item()
            or (self.clip_fps <= 0.0).any().item()
        ):
            raise ValueError("clip_fps must be positive and finite")
        if not torch.equal(self.clip_fps, torch.round(self.clip_fps)):
            raise ValueError("clip_fps must be integer-valued")
        expected_fps = torch.full_like(self.clip_fps, self.fps)
        if not torch.equal(self.clip_fps, expected_fps):
            raise ValueError("clip_fps must match fps")

        expected_starts = torch.empty_like(self.clip_starts)
        expected_starts[0] = 0
        if self.clip_lengths.numel() > 1:
            expected_starts[1:] = torch.cumsum(self.clip_lengths[:-1], dim=0)
        if not torch.equal(self.clip_starts, expected_starts):
            raise ValueError("clip spans must be contiguous")
        total_frames = int(self.clip_lengths.sum().item())
        if total_frames != self.root_pos.shape[0]:
            raise ValueError("clip spans must cover the packed frame tensors")

    def _validate_axis_names(self) -> None:
        if self.dof_names is not None:
            self._validate_name_sequence(
                self.dof_names,
                field_name="dof_names",
                expected_count=self.dof_pos.shape[1],
            )
        if self.body_names is None:
            return
        self._validate_name_sequence(
            self.body_names,
            field_name="body_names",
            expected_count=len(self.body_names),
        )
        body_count = len(self.body_names)
        for field_name in (
            "body_pos",
            "body_rot",
            "body_lin_vel",
            "body_ang_vel",
            "body_contacts",
        ):
            value = getattr(self, field_name)
            if value is not None and value.shape[1] != body_count:
                raise ValueError(
                    f"{field_name} body axis must match body_names count "
                    f"{body_count}, got {value.shape[1]}"
                )

    @staticmethod
    def _validate_name_sequence(
        names: tuple[str, ...],
        *,
        field_name: str,
        expected_count: int,
    ) -> None:
        if not isinstance(names, tuple):
            raise TypeError(f"{field_name} must be a tuple of strings")
        if len(names) != expected_count:
            raise ValueError(
                f"{field_name} count must match its tensor axis "
                f"{expected_count}, got {len(names)}"
            )
        for name in names:
            if not isinstance(name, str) or not name:
                raise ValueError(f"{field_name} must contain non-empty strings")
            try:
                name.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    f"{field_name} must contain valid UTF-8 strings"
                ) from exc

    def _validate_clip_names(self) -> None:
        metadata = (self.clip_name_bytes, self.clip_name_offsets)
        if all(value is None for value in metadata):
            return
        if any(value is None for value in metadata):
            raise ValueError(
                "clip_name_bytes and clip_name_offsets must be provided together"
            )
        if self.clip_starts is None:
            raise ValueError("clip names require explicit clip metadata")
        assert self.clip_name_bytes is not None
        assert self.clip_name_offsets is not None
        if self.clip_name_bytes.device.type != "cpu":
            raise ValueError("clip_name_bytes must remain on CPU")
        if self.clip_name_offsets.device.type != "cpu":
            raise ValueError("clip_name_offsets must remain on CPU")
        if self.clip_name_bytes.ndim != 1 or self.clip_name_bytes.dtype != torch.uint8:
            raise TypeError("clip_name_bytes must be a 1D uint8 tensor")
        if self.clip_name_offsets.ndim != 1 or self.clip_name_offsets.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("clip_name_offsets must be a 1D integer tensor")
        expected_offset_count = self.clip_starts.numel() + 1
        if self.clip_name_offsets.numel() != expected_offset_count:
            raise ValueError(
                "clip_name_offsets must contain one more entry than the clip count"
            )
        offsets = self.clip_name_offsets.tolist()
        if offsets[0] != 0 or offsets[-1] != self.clip_name_bytes.numel():
            raise ValueError("clip_name_offsets must span clip_name_bytes exactly")
        adjacent_offsets = zip(offsets[:-1], offsets[1:], strict=True)
        if any(start >= end for start, end in adjacent_offsets):
            raise ValueError("clip names must be non-empty contiguous byte spans")
        encoded_names = self.clip_name_bytes.numpy()
        adjacent_offsets = zip(offsets[:-1], offsets[1:], strict=True)
        for clip_id, (start, end) in enumerate(adjacent_offsets):
            if not self._is_valid_utf8(encoded_names[start:end]):
                raise ValueError(f"clip {clip_id} name is not valid UTF-8")

    @staticmethod
    def _is_valid_utf8(encoded: np.ndarray) -> bool:
        index = 0
        while index < len(encoded):
            first = int(encoded[index])
            if first <= 0x7F:
                index += 1
                continue
            if 0xC2 <= first <= 0xDF:
                continuation_ranges = ((0x80, 0xBF),)
            elif first == 0xE0:
                continuation_ranges = ((0xA0, 0xBF), (0x80, 0xBF))
            elif 0xE1 <= first <= 0xEC or 0xEE <= first <= 0xEF:
                continuation_ranges = ((0x80, 0xBF), (0x80, 0xBF))
            elif first == 0xED:
                continuation_ranges = ((0x80, 0x9F), (0x80, 0xBF))
            elif first == 0xF0:
                continuation_ranges = (
                    (0x90, 0xBF),
                    (0x80, 0xBF),
                    (0x80, 0xBF),
                )
            elif 0xF1 <= first <= 0xF3:
                continuation_ranges = (
                    (0x80, 0xBF),
                    (0x80, 0xBF),
                    (0x80, 0xBF),
                )
            elif first == 0xF4:
                continuation_ranges = (
                    (0x80, 0x8F),
                    (0x80, 0xBF),
                    (0x80, 0xBF),
                )
            else:
                return False
            if index + len(continuation_ranges) >= len(encoded):
                return False
            for offset, (lower, upper) in enumerate(continuation_ranges, start=1):
                if not lower <= int(encoded[index + offset]) <= upper:
                    return False
            index += len(continuation_ranges) + 1
        return True

    def clip_name(self, clip_id: int) -> str:
        """Decode one clip's UTF-8 identity on demand."""
        if not isinstance(clip_id, int):
            raise TypeError("clip_id must be an int")
        if self.clip_name_bytes is None or self.clip_name_offsets is None:
            raise ValueError("reference motion does not contain clip-name metadata")
        clip_count = self.clip_name_offsets.numel() - 1
        if clip_id < 0 or clip_id >= clip_count:
            raise IndexError(f"clip_id {clip_id} out of range for {clip_count} clips")
        start = int(self.clip_name_offsets[clip_id].item())
        end = int(self.clip_name_offsets[clip_id + 1].item())
        try:
            return bytes(self.clip_name_bytes[start:end].tolist()).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"clip {clip_id} name is not valid UTF-8") from exc


ReferenceMotionState = ReferenceMotion


class MotionLoader(ABC):
    """Base class for source-format adapters that produce reference motions."""

    @classmethod
    def load(
        cls,
        motion_files: str | Path,
        *,
        motion_format: MotionFormat = "pyroki",
        fps: float = 30.0,
        device: str | torch.device = "cpu",
        contact_labels: str | Path | None = None,
    ) -> list[ReferenceMotionState]:
        """Load reference motions from a known source format."""
        return cls.from_format(
            motion_files,
            motion_format=motion_format,
            fps=fps,
            device=device,
            contact_labels=contact_labels,
        ).load_motion()

    @classmethod
    def from_format(
        cls,
        motion_files: str | Path,
        *,
        motion_format: MotionFormat = "pyroki",
        fps: float = 30.0,
        device: str | torch.device = "cpu",
        contact_labels: str | Path | None = None,
    ) -> "MotionLoader":
        """Build the loader adapter for a known source format."""
        if motion_format == "pyroki":
            return PyrokiMotionLoader(
                motion_files,
                fps=fps,
                device=device,
                contact_labels=contact_labels,
            )
        if motion_format == "mjlab":
            if contact_labels is not None:
                raise ValueError("mjlab motion loading does not accept contact_labels")
            return ReferenceMotionNpzLoader(motion_files, fps=fps, device=device)
        if motion_format == "proto":
            raise NotImplementedError("proto motion loading is not implemented yet")
        raise ValueError(f"Unsupported motion format {motion_format!r}")

    @abstractmethod
    def load_motion(self) -> list[ReferenceMotionState]:
        """Load source motion data into reference motion states."""


class PyrokiMotionLoader(MotionLoader):
    """Load PyRoki retargeted ``.npz`` motions into reference motion states."""

    _REQUIRED_KEYS = ("base_frame_pos", "base_frame_wxyz", "joint_angles")

    def __init__(
        self,
        motion_files: str | Path,
        *,
        fps: float = 30.0,
        device: str | torch.device = "cpu",
        contact_labels: str | Path | None = None,
    ) -> None:
        self.motion_files = Path(motion_files)
        self.fps = _validate_integer_fps(fps)
        self.device = torch.device(device)
        self.contact_labels = (
            Path(contact_labels) if contact_labels is not None else None
        )

    def load_motion(self) -> list[ReferenceMotionState]:
        motions = []
        motion_paths = self._motion_paths()
        if (
            self.contact_labels is not None
            and self.contact_labels.is_file()
            and len(motion_paths) != 1
        ):
            raise ValueError(
                "A single contact label file can only be used with one motion file"
            )
        for motion_path in motion_paths:
            try:
                motions.append(self._load_file(motion_path))
            except Exception as exc:
                msg = f"{motion_path}: {exc}"
                raise type(exc)(msg) from exc
        return motions

    def _motion_paths(self) -> list[Path]:
        if self.motion_files.is_file():
            if self.motion_files.suffix != ".npz":
                raise ValueError(
                    f"Expected a .npz PyRoki motion file, got {self.motion_files}"
                )
            return [self.motion_files]

        if self.motion_files.is_dir():
            motion_paths = sorted(
                path
                for path in self.motion_files.iterdir()
                if path.is_file() and path.suffix == ".npz"
            )
            if not motion_paths:
                raise ValueError(
                    f"No direct child .npz files found in {self.motion_files}"
                )
            return motion_paths

        raise FileNotFoundError(f"Motion path does not exist: {self.motion_files}")

    def _load_file(self, motion_path: Path) -> ReferenceMotionState:
        with np.load(motion_path, allow_pickle=False) as data:
            arrays = {key: self._load_array(data, key) for key in self._REQUIRED_KEYS}

        root_pos = arrays["base_frame_pos"]
        root_rot = arrays["base_frame_wxyz"]
        dof_pos = arrays["joint_angles"]
        self._validate_shapes(root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos)

        return ReferenceMotionState(
            name=motion_path.name,
            display_name=self._display_name(motion_path.name),
            fps=self.fps,
            root_pos=self._to_float32_tensor(root_pos),
            root_rot=self._to_float32_tensor(root_rot),
            dof_pos=self._to_float32_tensor(dof_pos),
            foot_contacts=self._load_contact_labels(
                motion_path, frame_count=root_pos.shape[0]
            ),
        )

    @staticmethod
    def _display_name(name: str) -> str:
        suffixes = (
            "_poses_keypoints_retargeted.npz",
            ".npz",
        )
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def _load_contact_labels(
        self,
        motion_path: Path,
        *,
        frame_count: int,
    ) -> torch.Tensor | None:
        if self.contact_labels is None:
            return None

        contact_path = self._contact_label_path(motion_path)
        with np.load(contact_path, allow_pickle=False) as data:
            foot_contacts = self._load_array(data, "foot_contacts")

        if foot_contacts.ndim != 2 or foot_contacts.shape[1] != 2:
            raise ValueError(
                "'foot_contacts' must have shape (num_frames, 2), "
                f"got {foot_contacts.shape}"
            )
        if foot_contacts.shape[0] != frame_count:
            raise ValueError(
                "'foot_contacts' must share frame count "
                f"{frame_count}, got {foot_contacts.shape[0]}"
            )
        return self._to_float32_tensor(foot_contacts)

    def _contact_label_path(self, motion_path: Path) -> Path:
        assert self.contact_labels is not None
        if self.contact_labels.is_file():
            return self.contact_labels
        if not self.contact_labels.is_dir():
            raise FileNotFoundError(
                f"Contact labels path does not exist: {self.contact_labels}"
            )

        base_name = motion_path.stem
        if base_name.endswith("_retargeted"):
            base_name = base_name[: -len("_retargeted")]
        contact_path = self.contact_labels / f"{base_name}_contacts.npz"
        if not contact_path.is_file():
            raise FileNotFoundError(f"Contact labels file not found: {contact_path}")
        return contact_path

    @staticmethod
    def _load_array(
        data: np.lib.npyio.NpzFile,
        key: str,
    ) -> np.ndarray:
        if key not in data.files:
            raise ValueError(f"Missing required PyRoki key {key!r}")
        array = np.asarray(data[key])
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{key!r} must be numeric, got dtype {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{key!r} contains non-finite values")
        return array

    @staticmethod
    def _validate_shapes(
        *,
        root_pos: np.ndarray,
        root_rot: np.ndarray,
        dof_pos: np.ndarray,
    ) -> None:
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(
                f"'base_frame_pos' must have shape (num_frames, 3), got {root_pos.shape}"
            )
        if root_rot.ndim != 2 or root_rot.shape[1] != 4:
            raise ValueError(
                f"'base_frame_wxyz' must have shape (num_frames, 4), got {root_rot.shape}"
            )
        if dof_pos.ndim != 2:
            raise ValueError(f"'joint_angles' must be 2D, got shape {dof_pos.shape}")

        frame_count = root_pos.shape[0]
        if frame_count <= 2:
            raise ValueError("PyRoki motion must contain at least 3 frames")
        if root_rot.shape[0] != frame_count or dof_pos.shape[0] != frame_count:
            raise ValueError(
                "PyRoki arrays must have matching frame counts: "
                f"base_frame_pos={root_pos.shape[0]}, "
                f"base_frame_wxyz={root_rot.shape[0]}, "
                f"joint_angles={dof_pos.shape[0]}"
            )

        quat_norms = np.linalg.norm(root_rot, axis=1)
        if not np.allclose(quat_norms, 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("'base_frame_wxyz' quaternions must be normalized")

    def _to_float32_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)


class ReferenceMotionNpzLoader(MotionLoader):
    """Load legacy or versioned reference motion ``.npz`` artifacts."""

    def __init__(
        self,
        motion_files: str | Path,
        *,
        fps: float = 30.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.motion_files = Path(motion_files)
        self.fps = _validate_integer_fps(fps)
        self.device = torch.device(device)

    def load_motion(self) -> list[ReferenceMotionState]:
        motions = []
        for motion_path in self._motion_paths():
            try:
                motions.append(self._load_file(motion_path))
            except Exception as exc:
                msg = f"{motion_path}: {exc}"
                raise type(exc)(msg) from exc
        return motions

    def _motion_paths(self) -> list[Path]:
        if self.motion_files.is_file():
            if self.motion_files.suffix != ".npz":
                raise ValueError(
                    "Expected a .npz MotionLib reference motion file, "
                    f"got {self.motion_files}"
                )
            return [self.motion_files]

        if self.motion_files.is_dir():
            motion_paths = sorted(
                path
                for path in self.motion_files.iterdir()
                if path.is_file() and path.suffix == ".npz"
            )
            if not motion_paths:
                raise ValueError(
                    f"No direct child .npz files found in {self.motion_files}"
                )
            return motion_paths

        raise FileNotFoundError(f"Motion path does not exist: {self.motion_files}")

    def _load_file(self, motion_path: Path) -> ReferenceMotionState:
        with np.load(motion_path, allow_pickle=False) as data:
            if REFERENCE_MOTION_NPZ_V1.version_key in data.files:
                return self._load_versioned_file(data, motion_path)
            arrays = {
                key: self._load_float_array(data, key)
                for key in REFERENCE_MOTION_NPZ_V1.required_tensor_keys
            }
            body_contacts = self._load_optional_contacts(data)

        return ReferenceMotionState(
            name=motion_path.name,
            display_name=motion_path.stem,
            fps=self.fps,
            root_pos=self._to_tensor(arrays["root_pos"]),
            root_rot=self._to_tensor(arrays["root_rot"]),
            dof_pos=self._to_tensor(arrays["dof_pos"]),
            root_lin_vel=self._to_tensor(arrays["root_lin_vel"]),
            root_ang_vel=self._to_tensor(arrays["root_ang_vel"]),
            dof_vel=self._to_tensor(arrays["dof_vel"]),
            body_pos=self._to_tensor(arrays["body_pos"]),
            body_rot=self._to_tensor(arrays["body_rot"]),
            body_lin_vel=self._to_tensor(arrays["body_lin_vel"]),
            body_ang_vel=self._to_tensor(arrays["body_ang_vel"]),
            body_contacts=body_contacts,
        )

    def _load_versioned_file(
        self,
        data: np.lib.npyio.NpzFile,
        motion_path: Path,
    ) -> ReferenceMotionState:
        schema_version = self._load_schema_version(data)
        if schema_version != REFERENCE_MOTION_NPZ_V1.version:
            raise ValueError(
                "Unsupported reference motion schema version "
                f"{schema_version}; supported version is "
                f"{REFERENCE_MOTION_NPZ_V1.version}"
            )
        for key in REFERENCE_MOTION_NPZ_V1.required_metadata_keys:
            if key not in data.files:
                raise ValueError(
                    f"Missing required versioned reference motion key {key!r}"
                )

        arrays = {
            key: self._load_float_array(data, key)
            for key in REFERENCE_MOTION_NPZ_V1.required_tensor_keys
        }
        fps = self._load_scalar_fps(data)
        clip_starts = self._load_integer_metadata(
            data, REFERENCE_MOTION_NPZ_V1.clip_starts_key
        )
        clip_lengths = self._load_integer_metadata(
            data, REFERENCE_MOTION_NPZ_V1.clip_lengths_key
        )
        clip_fps = self._load_float_array(data, REFERENCE_MOTION_NPZ_V1.clip_fps_key)
        clip_name_bytes = self._load_clip_name_bytes(data)
        clip_name_offsets = self._load_integer_metadata(
            data,
            REFERENCE_MOTION_NPZ_V1.clip_name_offsets_key,
            device=torch.device("cpu"),
        )
        dof_names = self._load_names(data, REFERENCE_MOTION_NPZ_V1.dof_names_key)
        body_names = self._load_names(data, REFERENCE_MOTION_NPZ_V1.body_names_key)
        short_clip_ids = torch.nonzero(clip_lengths < 3).flatten()
        if short_clip_ids.numel() > 0:
            clip_id = int(short_clip_ids[0].item())
            raise ValueError(
                f"Schema version 1 rich clip {clip_id} must contain at least 3 frames"
            )

        return ReferenceMotionState(
            name=motion_path.name,
            display_name=motion_path.stem,
            fps=fps,
            root_pos=self._to_tensor(arrays["root_pos"]),
            root_rot=self._to_tensor(arrays["root_rot"]),
            dof_pos=self._to_tensor(arrays["dof_pos"]),
            root_lin_vel=self._to_tensor(arrays["root_lin_vel"]),
            root_ang_vel=self._to_tensor(arrays["root_ang_vel"]),
            dof_vel=self._to_tensor(arrays["dof_vel"]),
            body_pos=self._to_tensor(arrays["body_pos"]),
            body_rot=self._to_tensor(arrays["body_rot"]),
            body_lin_vel=self._to_tensor(arrays["body_lin_vel"]),
            body_ang_vel=self._to_tensor(arrays["body_ang_vel"]),
            body_contacts=self._load_optional_contacts(data),
            clip_starts=clip_starts,
            clip_lengths=clip_lengths,
            clip_fps=torch.as_tensor(clip_fps, device=self.device),
            clip_name_bytes=clip_name_bytes,
            clip_name_offsets=clip_name_offsets,
            dof_names=dof_names,
            body_names=body_names,
        )

    @staticmethod
    def _load_schema_version(data: np.lib.npyio.NpzFile) -> int:
        array = np.asarray(data[REFERENCE_MOTION_NPZ_V1.version_key])
        if array.ndim != 0 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("'schema_version' must be an integer scalar")
        return int(array.item())

    @staticmethod
    def _load_scalar_fps(data: np.lib.npyio.NpzFile) -> float:
        key = REFERENCE_MOTION_NPZ_V1.fps_key
        array = np.asarray(data[key])
        is_real_number = np.issubdtype(array.dtype, np.integer) or np.issubdtype(
            array.dtype, np.floating
        )
        if array.ndim != 0 or not is_real_number:
            raise ValueError(f"{key!r} must be a numeric scalar")
        return _validate_integer_fps(float(array.item()), field_name=key)

    def _load_integer_metadata(
        self,
        data: np.lib.npyio.NpzFile,
        key: str,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        array = np.asarray(data[key])
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{key!r} must be a 1D integer array")
        return torch.as_tensor(array, device=device or self.device)

    @staticmethod
    def _load_clip_name_bytes(data: np.lib.npyio.NpzFile) -> torch.Tensor:
        key = REFERENCE_MOTION_NPZ_V1.clip_name_bytes_key
        array = np.asarray(data[key])
        if array.ndim != 1 or array.dtype != np.uint8:
            raise ValueError(f"{key!r} must be a 1D uint8 array")
        return torch.as_tensor(array.copy(), dtype=torch.uint8, device="cpu")

    @staticmethod
    def _load_names(data: np.lib.npyio.NpzFile, key: str) -> tuple[str, ...]:
        array = np.asarray(data[key])
        if array.ndim != 1 or array.dtype.kind not in ("U", "S"):
            raise ValueError(f"{key!r} must be a 1D string array")
        try:
            if array.dtype.kind == "S":
                return tuple(bytes(value).decode("utf-8") for value in array)
            return tuple(str(value) for value in array)
        except UnicodeDecodeError as exc:
            raise ValueError(f"{key!r} contains invalid UTF-8") from exc

    @staticmethod
    def _load_float_array(
        data: np.lib.npyio.NpzFile,
        key: str,
    ) -> np.ndarray:
        if key not in data.files:
            raise ValueError(f"Missing required MotionLib key {key!r}")
        array = np.asarray(data[key])
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{key!r} must be floating point, got dtype {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{key!r} contains non-finite values")
        return array

    def _load_optional_contacts(
        self,
        data: np.lib.npyio.NpzFile,
    ) -> torch.Tensor | None:
        if "body_contacts" not in data.files:
            return None
        array = np.asarray(data["body_contacts"])
        if not (
            np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.number)
        ):
            raise ValueError(
                f"'body_contacts' must be bool or numeric, got dtype {array.dtype}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ValueError("'body_contacts' contains non-finite values")
        return torch.as_tensor(array, device=self.device)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device)
