from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

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
    """Load MotionLib train-ready ``.npz`` clips into reference motion states."""

    _REQUIRED_KEYS = (
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
            arrays = {
                key: self._load_float_array(data, key) for key in self._REQUIRED_KEYS
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
