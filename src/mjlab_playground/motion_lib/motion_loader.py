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
    validate_axis_names,
    validate_common_clip_fps,
    validate_compact_clip_names,
    validate_contiguous_clip_spans,
    validate_minimum_clip_lengths,
)

MotionFormat = Literal["pyroki", "proto", "mjlab"]


def _validate_integer_fps(fps: float, *, field_name: str = "fps") -> float:
    value = float(fps)
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{field_name} must be positive and finite")
    if not value.is_integer():
        raise ValueError(f"{field_name} must be integer-valued")
    return value


def _normalize_quaternion_array(
    array: np.ndarray,
    *,
    field_name: str,
    expected_ndim: int,
) -> np.ndarray:
    if array.ndim != expected_ndim or array.shape[-1] != 4:
        raise ValueError(
            f"{field_name!r} quaternions must have canonical wxyz shape; "
            f"got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name!r} contains non-finite quaternions")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"{field_name!r} contains zero quaternions")
    return array / norms


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


@dataclass(frozen=True, kw_only=True)
class ReferenceMotion:
    """Tensor-backed reference motion over packed frame tensors."""

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
    foot_contacts: torch.Tensor | None = None
    clip_starts: torch.Tensor | None = None
    clip_lengths: torch.Tensor | None = None
    clip_fps: torch.Tensor | None = None
    clip_name_bytes: torch.Tensor | None = None
    clip_name_offsets: torch.Tensor | None = None
    dof_names: tuple[str, ...] | None = None
    body_names: tuple[str, ...] | None = None

    @classmethod
    def from_frames(
        cls,
        frames: list[ReferenceFrame],
        *,
        name: str | None = None,
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

        root_pos = torch.stack([frame.root_pos for frame in frames])
        root_rot = torch.stack([frame.root_rot for frame in frames])
        dof_pos = torch.stack([frame.dof_pos for frame in frames])
        root_lin_vel = cls._stack_optional_frame_field(frames, "root_lin_vel")
        root_ang_vel = cls._stack_optional_frame_field(frames, "root_ang_vel")
        dof_vel = cls._stack_optional_frame_field(frames, "dof_vel")
        body_pos = cls._stack_optional_frame_field(frames, "body_pos")
        body_rot = cls._stack_optional_frame_field(frames, "body_rot")
        body_lin_vel = cls._stack_optional_frame_field(frames, "body_lin_vel")
        body_ang_vel = cls._stack_optional_frame_field(frames, "body_ang_vel")
        body_contacts = cls._stack_optional_frame_field(frames, "body_contacts")
        foot_contacts = cls._stack_optional_frame_field(frames, "foot_contacts")

        return cls(
            name=name,
            fps=fps,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            body_pos=body_pos,
            body_rot=body_rot,
            body_lin_vel=body_lin_vel,
            body_ang_vel=body_ang_vel,
            body_contacts=body_contacts,
            foot_contacts=foot_contacts,
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
        rich_field_names = (
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
        clip_name_parts: list[torch.Tensor] = []
        clip_name_offset_values = [0]
        clip_length_values: list[int] = []
        first = clip_list[0]

        for clip_id, clip in enumerate(clip_list):
            if not isinstance(clip, cls):
                raise TypeError(
                    f"clip {clip_id} must be a ReferenceMotion, got {type(clip).__name__}"
                )
            for field_name in rich_field_names:
                if getattr(clip, field_name) is None:
                    raise ValueError(
                        f"clip {clip_id} requires rich reference field {field_name}"
                    )
            if clip.foot_contacts is not None:
                raise ValueError(
                    f"clip {clip_id} contains source foot_contacts; "
                    "assembly accepts body_contacts only"
                )
            frame_count = clip.root_pos.shape[0]
            if frame_count < 3:
                raise ValueError(
                    f"clip {clip_id} rich reference clip must contain at least 3 frames"
                )
            if clip.dof_names is None:
                raise ValueError(f"clip {clip_id} requires dof_names")
            if clip.body_names is None:
                raise ValueError(f"clip {clip_id} requires body_names")
            if clip_id > 0:
                if clip.fps != first.fps:
                    raise ValueError(
                        f"clip {clip_id} fps must match clip 0 fps "
                        f"{first.fps}, got {clip.fps}"
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

            operational_metadata = (
                clip.clip_starts,
                clip.clip_lengths,
                clip.clip_fps,
                clip.clip_name_bytes,
                clip.clip_name_offsets,
            )
            if all(value is None for value in operational_metadata):
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
            else:
                if any(value is None for value in operational_metadata):
                    raise ValueError(
                        f"clip {clip_id} operational clip metadata must be provided together"
                    )
                assert clip.clip_starts is not None
                assert clip.clip_lengths is not None
                assert clip.clip_fps is not None
                assert clip.clip_name_bytes is not None
                assert clip.clip_name_offsets is not None
                if clip.clip_starts.shape != (1,):
                    raise ValueError(
                        f"clip {clip_id} is already multi-clip; "
                        "ReferenceMotion.from_clips accepts one-clip inputs"
                    )
                if int(clip.clip_starts[0].item()) != 0:
                    raise ValueError(
                        f"clip {clip_id} clip_starts must describe one clip starting at frame 0"
                    )
                if (
                    clip.clip_lengths.shape != (1,)
                    or int(clip.clip_lengths[0].item()) != frame_count
                ):
                    raise ValueError(
                        f"clip {clip_id} clip_lengths must match packed frame count {frame_count}"
                    )
                if clip.clip_fps.shape != (1,) or float(
                    clip.clip_fps[0].item()
                ) != float(clip.fps):
                    raise ValueError(
                        f"clip {clip_id} clip_fps must match fps {clip.fps}"
                    )
                if clip.clip_name_offsets.shape != (2,):
                    raise ValueError(
                        f"clip {clip_id} clip_name_offsets must describe one clip name"
                    )
                name_start = int(clip.clip_name_offsets[0].item())
                name_end = int(clip.clip_name_offsets[1].item())
                if (
                    name_start != 0
                    or name_end != clip.clip_name_bytes.numel()
                    or name_end <= name_start
                ):
                    raise ValueError(
                        f"clip {clip_id} clip_name_offsets must span clip_name_bytes"
                    )
                try:
                    bytes(clip.clip_name_bytes.tolist()).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"clip {clip_id} clip name must be valid UTF-8"
                    ) from exc
                encoded_name = clip.clip_name_bytes

            clip_name_parts.append(encoded_name)
            clip_name_offset_values.append(
                clip_name_offset_values[-1] + encoded_name.numel()
            )
            clip_length_values.append(frame_count)

        contacts_present = [clip.body_contacts is not None for clip in clip_list]
        clip_lengths = torch.tensor(
            clip_length_values, dtype=torch.int64, device=selected_device
        )
        clip_starts = torch.empty_like(clip_lengths)
        clip_starts[0] = 0
        if len(clip_list) > 1:
            clip_starts[1:] = torch.cumsum(clip_lengths[:-1], dim=0)
        clip_fps = torch.tensor(
            [clip.fps for clip in clip_list],
            dtype=torch.float32,
            device=selected_device,
        )
        clip_name_bytes = torch.cat(clip_name_parts)
        clip_name_offsets = torch.tensor(clip_name_offset_values, dtype=torch.int64)

        def concatenate(field_name: str) -> torch.Tensor:
            values = [getattr(clip, field_name) for clip in clip_list]
            return torch.cat(values, dim=0)

        root_pos = concatenate("root_pos")
        root_rot = concatenate("root_rot")
        dof_pos = concatenate("dof_pos")
        root_lin_vel = concatenate("root_lin_vel")
        root_ang_vel = concatenate("root_ang_vel")
        dof_vel = concatenate("dof_vel")
        body_pos = concatenate("body_pos")
        body_rot = concatenate("body_rot")
        body_lin_vel = concatenate("body_lin_vel")
        body_ang_vel = concatenate("body_ang_vel")
        body_contacts: torch.Tensor | None = None
        if all(contacts_present):
            contact_values = [clip.body_contacts for clip in clip_list]
            body_contacts = torch.cat(
                [value for value in contact_values if value is not None], dim=0
            )

        return cls(
            fps=first.fps,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            body_pos=body_pos,
            body_rot=body_rot,
            body_lin_vel=body_lin_vel,
            body_ang_vel=body_ang_vel,
            body_contacts=body_contacts,
            clip_starts=clip_starts,
            clip_lengths=clip_lengths,
            clip_fps=clip_fps,
            clip_name_bytes=clip_name_bytes,
            clip_name_offsets=clip_name_offsets,
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

    def get_frame(self, index: int) -> ReferenceFrame:
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


# Deprecated public compatibility alias; use ReferenceMotion in new code.
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
    ) -> list[ReferenceMotion]:
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
    def load_motion(self) -> list[ReferenceMotion]:
        """Load source motion data into reference motions."""


class PyrokiMotionLoader(MotionLoader):
    """Load PyRoki retargeted ``.npz`` motions into reference motions."""

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

    def load_motion(self) -> list[ReferenceMotion]:
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

    def _load_file(self, motion_path: Path) -> ReferenceMotion:
        with np.load(motion_path, allow_pickle=False) as data:
            arrays = {key: self._load_array(data, key) for key in self._REQUIRED_KEYS}

        root_pos = arrays["base_frame_pos"]
        root_rot = _normalize_quaternion_array(
            arrays["base_frame_wxyz"],
            field_name="base_frame_wxyz",
            expected_ndim=2,
        )
        dof_pos = arrays["joint_angles"]
        self._validate_shapes(root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos)

        return ReferenceMotion(
            name=motion_path.name,
            fps=self.fps,
            root_pos=self._to_float32_tensor(root_pos),
            root_rot=self._to_float32_tensor(root_rot),
            dof_pos=self._to_float32_tensor(dof_pos),
            foot_contacts=self._load_contact_labels(
                motion_path, frame_count=root_pos.shape[0]
            ),
        )

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

    def load_motion(self) -> list[ReferenceMotion]:
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

    def _load_file(self, motion_path: Path) -> ReferenceMotion:
        with np.load(motion_path, allow_pickle=False) as data:
            if REFERENCE_MOTION_NPZ_V1.version_key in data.files:
                return self._load_versioned_file(data, motion_path)
            arrays = {
                key: self._load_float_array(data, key)
                for key in REFERENCE_MOTION_NPZ_V1.required_tensor_keys
            }
            self._normalize_quaternion_arrays(arrays)
            body_contacts = self._load_optional_contacts(data)
            self._validate_array_relations(arrays, body_contacts=body_contacts)

        return ReferenceMotion(
            name=motion_path.name,
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
    ) -> ReferenceMotion:
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
        self._normalize_quaternion_arrays(arrays)
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
        body_contacts = self._load_optional_contacts(data)
        self._validate_array_relations(arrays, body_contacts=body_contacts)
        self._validate_versioned_operational_metadata(
            arrays=arrays,
            fps=fps,
            clip_starts=clip_starts,
            clip_lengths=clip_lengths,
            clip_fps=clip_fps,
            clip_name_bytes=clip_name_bytes,
            clip_name_offsets=clip_name_offsets,
            dof_names=dof_names,
            body_names=body_names,
        )
        validate_minimum_clip_lengths(
            clip_lengths=clip_lengths.tolist(),
            error_prefix="Schema version 1 ",
        )

        return ReferenceMotion(
            name=motion_path.name,
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
            body_contacts=body_contacts,
            clip_starts=clip_starts,
            clip_lengths=clip_lengths,
            clip_fps=self._to_tensor(clip_fps),
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
        return array

    @staticmethod
    def _normalize_quaternion_arrays(arrays: dict[str, np.ndarray]) -> None:
        arrays["root_rot"] = _normalize_quaternion_array(
            arrays["root_rot"],
            field_name="root_rot",
            expected_ndim=2,
        )
        arrays["body_rot"] = _normalize_quaternion_array(
            arrays["body_rot"],
            field_name="body_rot",
            expected_ndim=3,
        )

    @staticmethod
    def _validate_array_relations(
        arrays: dict[str, np.ndarray],
        *,
        body_contacts: torch.Tensor | None,
    ) -> None:
        for field_name, array in arrays.items():
            if array.ndim == 0:
                raise ValueError(f"{field_name!r} must expose a frame axis")

        frame_counts = {
            field_name: array.shape[0] for field_name, array in arrays.items()
        }
        if body_contacts is not None:
            if body_contacts.ndim == 0:
                raise ValueError("'body_contacts' must expose a frame axis")
            frame_counts["body_contacts"] = body_contacts.shape[0]
        if len(set(frame_counts.values())) != 1:
            counts = ", ".join(
                f"{field_name}={frame_count}"
                for field_name, frame_count in frame_counts.items()
            )
            raise ValueError(
                f"MotionLib arrays must have matching frame counts: {counts}"
            )

        dof_fields = ("dof_pos", "dof_vel")
        for field_name in dof_fields:
            if arrays[field_name].ndim < 2:
                raise ValueError(f"{field_name!r} must expose a DOF axis")
        dof_counts = {
            field_name: arrays[field_name].shape[1] for field_name in dof_fields
        }
        if len(set(dof_counts.values())) != 1:
            counts = ", ".join(
                f"{field_name}={dof_count}"
                for field_name, dof_count in dof_counts.items()
            )
            raise ValueError(f"MotionLib DOF axes must align: {counts}")

        body_fields = (
            "body_pos",
            "body_rot",
            "body_lin_vel",
            "body_ang_vel",
        )
        for field_name in body_fields:
            if arrays[field_name].ndim < 2:
                raise ValueError(f"{field_name!r} must expose a body axis")
        body_counts = {
            field_name: arrays[field_name].shape[1] for field_name in body_fields
        }
        if body_contacts is not None:
            if body_contacts.ndim < 2:
                raise ValueError("'body_contacts' must expose a body axis")
            body_counts["body_contacts"] = body_contacts.shape[1]
        if len(set(body_counts.values())) != 1:
            counts = ", ".join(
                f"{field_name}={body_count}"
                for field_name, body_count in body_counts.items()
            )
            raise ValueError(f"MotionLib body axes must align: {counts}")

    @staticmethod
    def _validate_versioned_operational_metadata(
        *,
        arrays: dict[str, np.ndarray],
        fps: float,
        clip_starts: torch.Tensor,
        clip_lengths: torch.Tensor,
        clip_fps: np.ndarray,
        clip_name_bytes: torch.Tensor,
        clip_name_offsets: torch.Tensor,
        dof_names: tuple[str, ...],
        body_names: tuple[str, ...],
    ) -> None:
        clip_count = clip_starts.numel()
        if clip_count == 0:
            raise ValueError("clip metadata must contain at least one clip")
        if clip_fps.ndim != 1:
            raise ValueError("clip_fps must be a 1D floating array")
        if clip_lengths.numel() != clip_count or clip_fps.size != clip_count:
            raise ValueError("clip metadata tensors must have matching lengths")
        if (clip_lengths <= 0).any().item():
            raise ValueError("clip_lengths must be positive")
        frame_count = arrays["root_pos"].shape[0]
        validate_contiguous_clip_spans(
            clip_starts=clip_starts.tolist(),
            clip_lengths=clip_lengths.tolist(),
            packed_frame_count=frame_count,
        )
        validate_common_clip_fps(
            clip_fps=clip_fps.tolist(),
            fps=fps,
            mismatch_error=f"clip_fps values must match fps {fps}",
        )

        offsets = clip_name_offsets.tolist()
        encoded_names = bytes(clip_name_bytes.tolist())
        validate_compact_clip_names(
            encoded_names=encoded_names,
            offsets=offsets,
            clip_count=clip_count,
        )

        for field_name, names, tensor_field_name in (
            ("dof_names", dof_names, "dof_pos"),
            ("body_names", body_names, "body_pos"),
        ):
            validate_axis_names(
                field_name=field_name,
                names=names,
                expected_count=arrays[tensor_field_name].shape[1],
                tensor_field_name=tensor_field_name,
            )

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
        return torch.as_tensor(array, device=self.device)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)
