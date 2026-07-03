from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

MotionFormat = Literal["pyroki", "proto"]


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
        raise ValueError(f"{field_name} must share device {device}, got {tensor.device}")
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
class ReferenceMotionState:
    """Reference motion state for one frame or a batch of frames."""

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "fps", _validate_integer_fps(self.fps))

        _validate_float_tensor(self.root_pos, field_name="root_pos", shape_suffix=(3,))
        frame_count = self.root_pos.shape[0]
        if frame_count <= 2:
            raise ValueError("ReferenceMotionState requires at least 3 frames")
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
        self._validate_optional_float_field("body_lin_vel", self.body_lin_vel, (None, 3))
        self._validate_optional_float_field("body_ang_vel", self.body_ang_vel, (None, 3))
        self._validate_optional_contacts()

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
        if torch.is_floating_point(self.body_contacts) and not torch.isfinite(
            self.body_contacts
        ).all().item():
            raise ValueError("body_contacts contains non-finite values")


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
    ) -> list[ReferenceMotionState]:
        """Load reference motions from a known source format."""
        return cls.from_format(
            motion_files,
            motion_format=motion_format,
            fps=fps,
            device=device,
        ).load_motion()

    @classmethod
    def from_format(
        cls,
        motion_files: str | Path,
        *,
        motion_format: MotionFormat = "pyroki",
        fps: float = 30.0,
        device: str | torch.device = "cpu",
    ) -> "MotionLoader":
        """Build the loader adapter for a known source format."""
        if motion_format == "pyroki":
            return PyrokiMotionLoader(motion_files, fps=fps, device=device)
        if motion_format == "proto":
            raise NotImplementedError("proto motion loading is not implemented yet")

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
                raise ValueError(f"Expected a .npz PyRoki motion file, got {self.motion_files}")
            return [self.motion_files]

        if self.motion_files.is_dir():
            motion_paths = sorted(
                path
                for path in self.motion_files.iterdir()
                if path.is_file() and path.suffix == ".npz"
            )
            if not motion_paths:
                raise ValueError(f"No direct child .npz files found in {self.motion_files}")
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
