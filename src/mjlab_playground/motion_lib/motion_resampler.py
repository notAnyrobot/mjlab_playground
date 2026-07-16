from __future__ import annotations

from dataclasses import dataclass

import torch

from .math_util import (
    finite_difference_velocity,
    lerp_frame_values,
    quaternion_angular_velocity_wxyz,
    slerp_frame_quaternions_wxyz,
)
from .motion_loader import (
    ReferenceMotion,
    _validate_integer_fps,
)


@dataclass(frozen=True, kw_only=True)
class MotionResamplingCfg:
    """Configuration for source-agnostic reference motion resampling."""

    output_fps: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_fps",
            _validate_integer_fps(self.output_fps, field_name="output_fps"),
        )


class ReferenceMotionResampler:
    """Resample generalized-coordinate reference motion clips."""

    _RICH_FIELDS = (
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
        "body_contacts",
        "foot_contacts",
    )

    def __init__(self, cfg: MotionResamplingCfg) -> None:
        self.cfg = cfg

    def resample(self, motion: ReferenceMotion) -> ReferenceMotion:
        self._validate_resampling_request(motion)

        frame_positions = self._target_frame_positions(motion)
        root_pos = lerp_frame_values(motion.root_pos, frame_positions)
        root_rot = slerp_frame_quaternions_wxyz(motion.root_rot, frame_positions)
        dof_pos = lerp_frame_values(motion.dof_pos, frame_positions)

        return ReferenceMotion(
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            name=motion.name,
            fps=self.cfg.output_fps,
            root_lin_vel=finite_difference_velocity(root_pos, self.cfg.output_fps),
            root_ang_vel=quaternion_angular_velocity_wxyz(root_rot, self.cfg.output_fps),
            dof_vel=finite_difference_velocity(dof_pos, self.cfg.output_fps),
        )

    def _validate_resampling_request(self, motion: ReferenceMotion) -> None:
        if self.cfg.output_fps < motion.fps:
            raise ValueError(
                "ReferenceMotionResampler v1 does not support downsampling: "
                f"output_fps={self.cfg.output_fps}, motion.fps={motion.fps}"
            )
        for field_name in self._RICH_FIELDS:
            if getattr(motion, field_name) is not None:
                raise ValueError(
                    "ReferenceMotionResampler v1 accepts generalized-coordinate "
                    f"motions only; got rich field {field_name}"
                )

    def _target_frame_positions(self, motion: ReferenceMotion) -> torch.Tensor:
        duration = (motion.root_pos.shape[0] - 1) / motion.fps
        target_frames = int(round(duration * self.cfg.output_fps)) + 1
        return torch.linspace(
            0.0,
            float(motion.root_pos.shape[0] - 1),
            target_frames,
            dtype=motion.root_pos.dtype,
            device=motion.root_pos.device,
        )
