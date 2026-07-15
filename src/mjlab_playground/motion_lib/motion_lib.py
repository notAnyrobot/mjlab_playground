from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .motion_loader import (
    MotionFormat,
    MotionLoader,
    ReferenceFrame,
    ReferenceMotion,
    ReferenceMotionState,
    _validate_integer_fps,
)
from .motion_resampler import MotionResamplingCfg, ReferenceMotionResampler
from .mujoco_scene_adapter import MujocoSceneAdapter, MujocoSceneAdapterCfg

MotionLibRobot = Literal["astro"]


def _load_astro_constants_module() -> Any:
    module_name = "_mjlab_playground_motion_lib_astro_constants"
    if module_name in sys.modules:
        return sys.modules[module_name]

    src_path = Path(__file__).resolve().parents[1]
    constants_path = src_path / "asset_zoo" / "robots" / "astro" / "astro_constants.py"
    spec = importlib.util.spec_from_file_location(module_name, constants_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Astro constants from {constants_path}")

    existing_package = sys.modules.get("mjlab_playground")
    installed_stub = existing_package is None
    if installed_stub:
        package_stub = types.ModuleType("mjlab_playground")
        package_stub.__dict__["MJLAB_PLAYGROUND_SRC_PATH"] = src_path
        package_stub.__path__ = [str(src_path)]  # type: ignore[attr-defined]
        sys.modules["mjlab_playground"] = package_stub

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if installed_stub:
            sys.modules.pop("mjlab_playground", None)

    return module


@dataclass(frozen=True, kw_only=True)
class MotionLibCfg:
    """Stable public configuration for the MotionLib stage pipeline."""

    source_format: MotionFormat = "pyroki"
    source_fps: float = 30.0
    output_fps: float = 30.0
    robot: MotionLibRobot = "astro"
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if self.robot != "astro":
            raise ValueError("Only robot='astro' is supported by MotionLib v1")
        if self.source_format == "proto":
            raise NotImplementedError(
                "proto source format is reserved but not implemented"
            )
        if self.source_format != "pyroki":
            raise ValueError("Only source_format='pyroki' is supported by MotionLib v1")
        object.__setattr__(
            self,
            "source_fps",
            _validate_integer_fps(self.source_fps, field_name="source_fps"),
        )
        object.__setattr__(
            self,
            "output_fps",
            _validate_integer_fps(self.output_fps, field_name="output_fps"),
        )
        object.__setattr__(self, "device", torch.device(self.device))


class MotionLib:
    """Public source-to-rich reference motion stage interface."""

    _ENRICH_INPUT_RICH_FIELDS = (
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
        "body_contacts",
    )
    _ENRICH_OUTPUT_BODY_FIELDS = (
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
    )

    def __init__(self, cfg: MotionLibCfg) -> None:
        self.cfg = cfg
        self._resampler = ReferenceMotionResampler(
            MotionResamplingCfg(output_fps=cfg.output_fps)
        )
        self._enrichment_adapter: Any | None = None

    def load(self, motion_files: str | Path) -> list[ReferenceMotionState]:
        """Load source generalized-coordinate clips with the configured source format."""
        return MotionLoader.load(
            motion_files,
            motion_format=self.cfg.source_format,
            fps=self.cfg.source_fps,
            device=self.cfg.device,
        )

    def resample(
        self,
        motion: ReferenceMotionState,
    ) -> ReferenceMotionState:
        """Resample one source clip without simulator enrichment."""
        if not isinstance(motion, ReferenceMotionState):
            raise TypeError("MotionLib.resample expects one ReferenceMotionState")
        try:
            return self._resampler.resample(motion)
        except Exception as exc:
            identifier = self._motion_identifier(motion)
            raise type(exc)(f"{identifier}: {exc}") from exc

    def query(
        self,
        motion: ReferenceMotionState,
        *,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> ReferenceMotionState:
        """Query package-shaped reference motion tensors by clip ID and seconds."""
        if not isinstance(motion_ids, torch.Tensor):
            raise TypeError("motion_ids must be a torch.Tensor")
        if not isinstance(motion_times, torch.Tensor):
            raise TypeError("motion_times must be a torch.Tensor")
        if motion_ids.ndim != 1:
            raise ValueError("motion_ids must be 1D")
        if motion_times.ndim != 1:
            raise ValueError("motion_times must be 1D")
        if motion_ids.shape != motion_times.shape:
            raise ValueError("motion_ids and motion_times must have matching shapes")
        integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
        if motion_ids.dtype not in integer_dtypes:
            raise TypeError("motion_ids must be an integer tensor")
        if not torch.is_floating_point(motion_times):
            raise TypeError("motion_times must be a floating point tensor")
        if motion_ids.device != motion.root_pos.device:
            raise ValueError(
                f"motion_ids must share device {motion.root_pos.device}, got {motion_ids.device}"
            )
        if motion_times.device != motion.root_pos.device:
            raise ValueError(
                "motion_times must share device "
                f"{motion.root_pos.device}, got {motion_times.device}"
            )
        if not torch.isfinite(motion_times).all().item():
            raise ValueError("motion_times contains non-finite values")

        clip_starts, clip_lengths, clip_fps = self._package_clip_metadata(motion)
        num_clips = int(clip_starts.numel())
        if ((motion_ids < 0) | (motion_ids >= num_clips)).any().item():
            raise ValueError("motion_ids contains out-of-range clip IDs")

        selected_clip_fps = clip_fps[motion_ids]
        selected_clip_lengths = clip_lengths[motion_ids]
        clip_durations = (
            selected_clip_lengths.to(motion_times.dtype) - 1.0
        ) / selected_clip_fps.to(motion_times.dtype)
        if ((motion_times < 0.0) | (motion_times > clip_durations)).any().item():
            raise ValueError("motion_times contains out-of-range clip-local times")

        frame_positions = motion_times * selected_clip_fps.to(motion_times.dtype)
        local_lower = torch.floor(frame_positions).to(dtype=torch.long)
        local_upper = torch.minimum(local_lower + 1, selected_clip_lengths - 1)
        blend = frame_positions - local_lower.to(dtype=frame_positions.dtype)
        lower_indices = clip_starts[motion_ids] + local_lower
        upper_indices = clip_starts[motion_ids] + local_upper

        return ReferenceMotionState(
            name=motion.name,
            display_name=motion.display_name,
            fps=motion.fps,
            root_pos=self._lerp_indexed_values(
                motion.root_pos, lower_indices, upper_indices, blend
            ),
            root_rot=self._slerp_indexed_quaternions_wxyz(
                motion.root_rot, lower_indices, upper_indices, blend
            ),
            dof_pos=self._lerp_indexed_values(
                motion.dof_pos, lower_indices, upper_indices, blend
            ),
            root_lin_vel=self._maybe_lerp_indexed_values(
                motion.root_lin_vel, lower_indices, upper_indices, blend
            ),
            root_ang_vel=self._maybe_lerp_indexed_values(
                motion.root_ang_vel, lower_indices, upper_indices, blend
            ),
            dof_vel=self._maybe_lerp_indexed_values(
                motion.dof_vel, lower_indices, upper_indices, blend
            ),
            body_pos=self._maybe_lerp_indexed_values(
                motion.body_pos, lower_indices, upper_indices, blend
            ),
            body_rot=self._maybe_slerp_indexed_quaternions_wxyz(
                motion.body_rot, lower_indices, upper_indices, blend
            ),
            body_lin_vel=self._maybe_lerp_indexed_values(
                motion.body_lin_vel, lower_indices, upper_indices, blend
            ),
            body_ang_vel=self._maybe_lerp_indexed_values(
                motion.body_ang_vel, lower_indices, upper_indices, blend
            ),
            body_contacts=self._maybe_query_contact_values(
                motion.body_contacts, lower_indices, upper_indices, blend
            ),
            foot_contacts=self._maybe_query_contact_values(
                motion.foot_contacts, lower_indices, upper_indices, blend
            ),
        )

    @staticmethod
    def _package_clip_metadata(
        motion: ReferenceMotionState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            motion.clip_starts is None
            and motion.clip_lengths is None
            and motion.clip_fps is None
        ):
            return (
                torch.tensor([0], dtype=torch.long, device=motion.root_pos.device),
                torch.tensor(
                    [motion.root_pos.shape[0]],
                    dtype=torch.long,
                    device=motion.root_pos.device,
                ),
                torch.tensor(
                    [motion.fps],
                    dtype=motion.root_pos.dtype,
                    device=motion.root_pos.device,
                ),
            )
        if (
            motion.clip_starts is None
            or motion.clip_lengths is None
            or motion.clip_fps is None
        ):
            raise ValueError(
                "clip_starts, clip_lengths, and clip_fps must be provided together"
            )
        return motion.clip_starts, motion.clip_lengths, motion.clip_fps

    @staticmethod
    def _maybe_query_contact_values(
        values: torch.Tensor | None,
        lower_indices: torch.Tensor,
        upper_indices: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        if torch.is_floating_point(values):
            return MotionLib._lerp_indexed_values(
                values, lower_indices, upper_indices, blend
            )
        return values[lower_indices]

    @staticmethod
    def _maybe_lerp_indexed_values(
        values: torch.Tensor | None,
        lower_indices: torch.Tensor,
        upper_indices: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        return MotionLib._lerp_indexed_values(
            values, lower_indices, upper_indices, blend
        )

    @staticmethod
    def _lerp_indexed_values(
        values: torch.Tensor,
        lower_indices: torch.Tensor,
        upper_indices: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor:
        view_shape = (-1,) + (1,) * (values.ndim - 1)
        blend_view = blend.reshape(view_shape)
        return (
            values[lower_indices] * (1.0 - blend_view)
            + values[upper_indices] * blend_view
        )

    @staticmethod
    def _maybe_slerp_indexed_quaternions_wxyz(
        quaternions: torch.Tensor | None,
        lower_indices: torch.Tensor,
        upper_indices: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor | None:
        if quaternions is None:
            return None
        return MotionLib._slerp_indexed_quaternions_wxyz(
            quaternions, lower_indices, upper_indices, blend
        )

    @staticmethod
    def _slerp_indexed_quaternions_wxyz(
        quaternions: torch.Tensor,
        lower_indices: torch.Tensor,
        upper_indices: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor:
        q0 = quaternions[lower_indices]
        q1 = quaternions[upper_indices]
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        t = blend.reshape((-1,) + (1,) * (q0.ndim - 1))
        lerp_result = q0 * (1.0 - t) + q1 * t

        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        slerp_result = (
            torch.sin((1.0 - t) * theta) / sin_theta * q0
            + torch.sin(t * theta) / sin_theta * q1
        )
        near_parallel = dot.abs() > 0.9995
        result = torch.where(near_parallel, lerp_result, slerp_result)
        return torch.nn.functional.normalize(result, dim=-1)

    @staticmethod
    def _motion_identifier(motion: ReferenceMotionState) -> str:
        return motion.name or motion.display_name or "<unnamed motion>"

    def enrich(
        self,
        motion: ReferenceMotionState,
    ) -> ReferenceMotionState:
        """Enrich one already-resampled clip with simulator-derived body fields."""
        if not isinstance(motion, ReferenceMotionState):
            raise TypeError("MotionLib.enrich expects one ReferenceMotionState")
        self._validate_enrich_input(motion)
        adapter = self._get_enrichment_adapter()
        try:
            rich_motion = self._enrich_motion(adapter, motion)
            self._validate_enrich_output(motion, rich_motion)
        except Exception as exc:
            identifier = self._motion_identifier(motion)
            raise type(exc)(f"{identifier}: {exc}") from exc
        return rich_motion

    def _enrich_motion(
        self,
        adapter: Any,
        motion: ReferenceMotionState,
    ) -> ReferenceMotionState:
        frames: list[ReferenceFrame] = []
        frame_count = int(motion.root_pos.shape[0])
        for frame_index in range(frame_count):
            source_frame = motion.frame(frame_index)
            adapter.apply_frame(source_frame)
            rich_frame = adapter.read_robot_state()
            frames.append(
                ReferenceFrame(
                    root_pos=rich_frame.root_pos,
                    root_rot=rich_frame.root_rot,
                    dof_pos=rich_frame.dof_pos,
                    root_lin_vel=rich_frame.root_lin_vel,
                    root_ang_vel=rich_frame.root_ang_vel,
                    dof_vel=rich_frame.dof_vel,
                    body_pos=rich_frame.body_pos,
                    body_rot=rich_frame.body_rot,
                    body_lin_vel=rich_frame.body_lin_vel,
                    body_ang_vel=rich_frame.body_ang_vel,
                    body_contacts=source_frame.body_contacts,
                    foot_contacts=source_frame.foot_contacts,
                )
            )
        return ReferenceMotion.from_frames(
            frames,
            name=motion.name,
            display_name=motion.display_name,
            fps=motion.fps,
            clip_starts=motion.clip_starts,
            clip_lengths=motion.clip_lengths,
            clip_fps=motion.clip_fps,
            clip_name_bytes=motion.clip_name_bytes,
            clip_name_offsets=motion.clip_name_offsets,
            dof_names=tuple(adapter.joint_names),
            body_names=tuple(adapter.body_names),
        )

    def _validate_enrich_input(self, motion: ReferenceMotionState) -> None:
        if motion.fps != self.cfg.output_fps:
            identifier = self._motion_identifier(motion)
            raise ValueError(
                f"{identifier}: MotionLib.enrich expects resampled clips at "
                f"cfg.output_fps={self.cfg.output_fps}; got motion.fps={motion.fps}"
            )

        for field_name in self._ENRICH_INPUT_RICH_FIELDS:
            if getattr(motion, field_name) is not None:
                identifier = self._motion_identifier(motion)
                raise ValueError(
                    f"{identifier}: MotionLib.enrich accepts resampled "
                    "generalized-coordinate clips only; got existing rich field "
                    f"{field_name}"
                )

        if motion.foot_contacts is not None:
            identifier = self._motion_identifier(motion)
            raise ValueError(
                f"{identifier}: MotionLib.enrich v1 does not support foot_contacts"
            )

    def _validate_enrich_output(
        self,
        source_motion: ReferenceMotionState,
        rich_motion: object,
    ) -> None:
        if not isinstance(rich_motion, ReferenceMotionState):
            raise TypeError(
                "MotionLib.enrich must return ReferenceMotionState objects"
            )
        for field_name in self._ENRICH_OUTPUT_BODY_FIELDS:
            if getattr(rich_motion, field_name) is None:
                identifier = self._motion_identifier(source_motion)
                raise ValueError(
                    f"{identifier}: MuJoCo scene adapter did not fill {field_name}"
                )

    def _get_enrichment_adapter(self) -> Any:
        if self._enrichment_adapter is None:
            mujoco_scene_cfg = MujocoSceneAdapterCfg(
                robot_cfg=self._robot_cfg(),
                output_fps=self.cfg.output_fps,
                device=self.cfg.device,
            )
            mujoco_scene_adapter = MujocoSceneAdapter(mujoco_scene_cfg)
            self._enrichment_adapter = mujoco_scene_adapter
        return self._enrichment_adapter

    def _robot_cfg(self) -> Any:
        if self.cfg.robot == "astro":
            return _load_astro_constants_module().get_astro_robot_cfg()
        raise ValueError(
            f"Unsupported robot {self.cfg.robot!r}. Supported robots: astro"
        )
