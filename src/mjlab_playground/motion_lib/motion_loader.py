from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

MotionFormat = Literal["pyroki", "proto"]


@dataclass(frozen=True, kw_only=True)
class ReferenceMotionState:
    """Reference motion state for one frame or a batch of frames."""

    root_pos: torch.Tensor
    root_rot: torch.Tensor
    dof_pos: torch.Tensor
    name: str | None = None
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
        if self.fps <= 0.0 or not math.isfinite(self.fps):
            raise ValueError("fps must be positive and finite")


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
        if fps <= 0.0 or not math.isfinite(fps):
            raise ValueError("fps must be positive and finite")
        self.motion_files = Path(motion_files)
        self.fps = float(fps)
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
            fps=self.fps,
            root_pos=self._to_float32_tensor(root_pos),
            root_rot=self._to_float32_tensor(root_rot),
            dof_pos=self._to_float32_tensor(dof_pos),
        )

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
        if frame_count == 0:
            raise ValueError("PyRoki motion must contain at least one frame")
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
