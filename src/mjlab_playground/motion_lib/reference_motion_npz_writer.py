from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ._reference_motion_npz_schema import REFERENCE_MOTION_NPZ_V1

if TYPE_CHECKING:
    from .motion_loader import ReferenceMotionState


class ReferenceMotionNpzWriter:
    """Write a canonical reference motion as a schema-versioned `.npz` artifact."""

    def write(
        self,
        motion: ReferenceMotionState,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        path = Path(output_path)
        if path.suffix != ".npz":
            raise ValueError(
                f"ReferenceMotionNpzWriter output path must end with .npz: {path}"
            )
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"ReferenceMotionNpzWriter output path already exists: {path}"
            )
        self._validate_train_ready(motion)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            field_name: self._to_numpy(getattr(motion, field_name))
            for field_name in REFERENCE_MOTION_NPZ_V1.required_tensor_keys
        }
        clip_starts, clip_lengths, clip_fps = self._clip_metadata(motion)
        clip_name_bytes, clip_name_offsets = self._clip_name_metadata(motion)
        payload.update(
            {
                REFERENCE_MOTION_NPZ_V1.version_key: np.asarray(
                    REFERENCE_MOTION_NPZ_V1.version, dtype=np.int64
                ),
                REFERENCE_MOTION_NPZ_V1.fps_key: np.asarray(
                    motion.fps, dtype=np.float32
                ),
                REFERENCE_MOTION_NPZ_V1.clip_starts_key: clip_starts,
                REFERENCE_MOTION_NPZ_V1.clip_lengths_key: clip_lengths,
                REFERENCE_MOTION_NPZ_V1.clip_fps_key: clip_fps,
                REFERENCE_MOTION_NPZ_V1.clip_name_bytes_key: clip_name_bytes,
                REFERENCE_MOTION_NPZ_V1.clip_name_offsets_key: clip_name_offsets,
                REFERENCE_MOTION_NPZ_V1.dof_names_key: np.asarray(
                    motion.dof_names, dtype=np.str_
                ),
                REFERENCE_MOTION_NPZ_V1.body_names_key: np.asarray(
                    motion.body_names, dtype=np.str_
                ),
            }
        )
        if motion.body_contacts is not None:
            payload["body_contacts"] = self._to_numpy(motion.body_contacts)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=path.parent,
                suffix=".npz",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                np.savez(temporary_file, allow_pickle=False, **payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            if overwrite:
                os.replace(temporary_path, path)
            else:
                try:
                    os.link(temporary_path, path)
                except FileExistsError:
                    raise FileExistsError(
                        f"ReferenceMotionNpzWriter output path already exists: {path}"
                    ) from None
                temporary_path.unlink()
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _validate_train_ready(self, motion: ReferenceMotionState) -> None:
        if motion.foot_contacts is not None:
            raise ValueError(
                "ReferenceMotionNpzWriter v1 does not serialize source foot_contacts; "
                "use body_contacts for train-ready robot contact labels"
            )
        for field_name in REFERENCE_MOTION_NPZ_V1.required_tensor_keys:
            if getattr(motion, field_name) is None:
                raise ValueError(
                    "ReferenceMotionNpzWriter requires train-ready rich reference "
                    f"field {field_name}"
                )
        if motion.dof_names is None:
            raise ValueError(
                "ReferenceMotionNpzWriter requires dof_names aligned to dof_pos"
            )
        if motion.body_names is None:
            raise ValueError(
                "ReferenceMotionNpzWriter requires body_names aligned to body tensors"
            )
        if motion.clip_lengths is None:
            if motion.root_pos.shape[0] < 3:
                raise ValueError(
                    "ReferenceMotionNpzWriter rich clip must contain at least 3 frames"
                )
            return
        short_clip_ids = torch.nonzero(motion.clip_lengths < 3).flatten()
        if short_clip_ids.numel() > 0:
            clip_id = int(short_clip_ids[0].item())
            raise ValueError(
                f"ReferenceMotionNpzWriter rich clip {clip_id} must contain at least "
                "3 frames"
            )
        assert motion.clip_fps is not None
        if not torch.equal(
            motion.clip_fps,
            torch.full_like(motion.clip_fps, motion.fps),
        ):
            raise ValueError(
                "ReferenceMotionNpzWriter schema version 1 requires one common FPS"
            )

    @classmethod
    def _clip_metadata(
        cls,
        motion: ReferenceMotionState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if motion.clip_starts is None:
            return (
                np.asarray([0], dtype=np.int64),
                np.asarray([motion.root_pos.shape[0]], dtype=np.int64),
                np.asarray([motion.fps], dtype=np.float32),
            )
        assert motion.clip_lengths is not None
        assert motion.clip_fps is not None
        return (
            cls._to_numpy(motion.clip_starts).astype(np.int64, copy=False),
            cls._to_numpy(motion.clip_lengths).astype(np.int64, copy=False),
            cls._to_numpy(motion.clip_fps).astype(np.float32, copy=False),
        )

    @staticmethod
    def _clip_name_metadata(
        motion: ReferenceMotionState,
    ) -> tuple[np.ndarray, np.ndarray]:
        if motion.clip_name_bytes is not None:
            assert motion.clip_name_offsets is not None
            return (
                motion.clip_name_bytes.numpy(),
                motion.clip_name_offsets.numpy().astype(np.int64, copy=False),
            )
        if motion.clip_starts is not None:
            raise ValueError(
                "ReferenceMotionNpzWriter requires clip identities for explicit "
                "packaged motion metadata"
            )
        if not motion.name:
            raise ValueError(
                "ReferenceMotionNpzWriter requires a clip name for versioned output"
            )
        try:
            encoded = motion.name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "ReferenceMotionNpzWriter clip name must be valid UTF-8"
            ) from exc
        return (
            np.frombuffer(encoded, dtype=np.uint8),
            np.asarray([0, len(encoded)], dtype=np.int64),
        )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perform reference motion assembly over versioned reference motion "
            "artifacts and write the resulting versioned reference motion artifact."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Versioned reference motion artifact or flat directory of versioned "
            "reference motion artifacts."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination versioned reference motion artifact.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Device used to load artifacts and perform reference motion assembly "
            "(default: cpu)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing versioned reference motion artifact.",
    )
    return parser.parse_args(argv)


def package_reference_motions(
    input_path: str | Path,
    output_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
) -> None:
    """Perform reference motion assembly over versioned reference motion artifacts."""
    from .motion_loader import MotionLoader, ReferenceMotion

    source_path = Path(input_path)
    clips = MotionLoader.load(
        source_path,
        motion_format="mjlab",
        device=device,
    )
    input_paths = [
        source_path / (clip.name or f"clip-{clip_id}")
        if source_path.is_dir()
        else source_path
        for clip_id, clip in enumerate(clips)
    ]
    for clip, failing_input in zip(clips, input_paths, strict=True):
        if clip.clip_starts is None:
            raise ValueError(
                f"{failing_input}: reference motion assembly requires a versioned "
                "reference motion artifact with schema version 1; legacy rich "
                "reference artifacts are not accepted"
            )
        if clip.clip_starts.numel() != 1:
            raise ValueError(
                f"{failing_input}: reference motion assembly accepts only one-clip "
                "versioned reference motion artifacts; the input reference motion "
                "contains multiple clips"
            )

    try:
        assembled_motion = ReferenceMotion.from_clips(clips, device=device)
    except (TypeError, ValueError) as exc:
        input_order = ", ".join(
            f"clip {clip_id}={path}" for clip_id, path in enumerate(input_paths)
        )
        raise type(exc)(f"{input_order}: {exc}") from exc
    ReferenceMotionNpzWriter().write(assembled_motion, output_path, overwrite=overwrite)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    package_reference_motions(
        args.input,
        args.output,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
