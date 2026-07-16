from __future__ import annotations

from collections.abc import Sequence
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


def validate_contiguous_clip_spans(
    *,
    clip_starts: Sequence[int],
    clip_lengths: Sequence[int],
    packed_frame_count: int,
    error_prefix: str = "",
) -> None:
    """Validate representation-agnostic packed clip span semantics."""
    if clip_starts[0] != 0:
        raise ValueError(f"{error_prefix}clip spans must start at frame 0")
    expected_start = 0
    for start, length in zip(clip_starts, clip_lengths, strict=True):
        if start != expected_start:
            raise ValueError(f"{error_prefix}clip spans must be contiguous")
        expected_start += length
    if expected_start != packed_frame_count:
        raise ValueError(
            f"{error_prefix}clip spans must cover packed frame count "
            f"{packed_frame_count}, got {expected_start}"
        )


def validate_minimum_clip_lengths(
    *,
    clip_lengths: Sequence[int],
    error_prefix: str = "",
    include_clip_id: bool = True,
) -> None:
    """Validate the schema-v1 minimum length for every rich clip."""
    for clip_id, clip_length in enumerate(clip_lengths):
        if clip_length < 3:
            clip_label = f"rich clip {clip_id}" if include_clip_id else "rich clip"
            raise ValueError(
                f"{error_prefix}{clip_label} must contain at least 3 frames"
            )


def validate_common_clip_fps(
    *,
    clip_fps: Sequence[float],
    fps: float,
    mismatch_error: str,
    error_prefix: str = "",
) -> None:
    """Validate that every clip uses the artifact's common FPS."""
    if any(value != fps for value in clip_fps):
        raise ValueError(f"{error_prefix}{mismatch_error}")


def validate_compact_clip_names(
    *,
    encoded_names: bytes,
    offsets: Sequence[int],
    clip_count: int,
    error_prefix: str = "",
) -> None:
    """Validate compact non-empty UTF-8 clip identity spans."""
    expected_offset_count = clip_count + 1
    if len(offsets) != expected_offset_count:
        raise ValueError(
            f"{error_prefix}clip_name_offsets must contain one more entry than "
            "the clip count"
        )
    byte_count = len(encoded_names)
    if offsets[0] != 0:
        raise ValueError(f"{error_prefix}clip_name_offsets must start at byte 0")
    if offsets[-1] != byte_count:
        raise ValueError(
            f"{error_prefix}clip_name_offsets must span clip_name_bytes length "
            f"{byte_count}, got end {offsets[-1]}"
        )
    for clip_id, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        if start >= end:
            raise ValueError(
                f"{error_prefix}clip names must be non-empty contiguous byte spans"
            )
        try:
            encoded_names[start:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{error_prefix}clip {clip_id} name is not valid UTF-8"
            ) from exc


def validate_axis_names(
    *,
    field_name: str,
    names: Sequence[object],
    expected_count: int,
    tensor_field_name: str,
    error_prefix: str = "",
) -> None:
    """Validate axis-name count and non-empty UTF-8 identity rules."""
    if len(names) != expected_count:
        raise ValueError(
            f"{error_prefix}{field_name} count must match {tensor_field_name} axis "
            f"{expected_count}, got {len(names)}"
        )
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{error_prefix}{field_name} must contain non-empty strings"
            )
        try:
            name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"{error_prefix}{field_name} must contain valid UTF-8 strings"
            ) from exc
