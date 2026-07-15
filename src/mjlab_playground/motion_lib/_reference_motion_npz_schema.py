from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReferenceMotionNpzSchema:
    """Private descriptor for one reference-motion NPZ schema version."""

    version_key: str
    version: int
    required_tensor_keys: tuple[str, ...]
    fps_key: str
    clip_starts_key: str
    clip_lengths_key: str
    clip_fps_key: str
    clip_name_bytes_key: str
    clip_name_offsets_key: str
    dof_names_key: str
    body_names_key: str

    @property
    def required_metadata_keys(self) -> tuple[str, ...]:
        return (
            self.fps_key,
            self.clip_starts_key,
            self.clip_lengths_key,
            self.clip_fps_key,
            self.clip_name_bytes_key,
            self.clip_name_offsets_key,
            self.dof_names_key,
            self.body_names_key,
        )


REFERENCE_MOTION_NPZ_V1 = ReferenceMotionNpzSchema(
    version_key="schema_version",
    version=1,
    required_tensor_keys=(
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
    ),
    fps_key="fps",
    clip_starts_key="clip_starts",
    clip_lengths_key="clip_lengths",
    clip_fps_key="clip_fps",
    clip_name_bytes_key="clip_name_bytes",
    clip_name_offsets_key="clip_name_offsets",
    dof_names_key="dof_names",
    body_names_key="body_names",
)
