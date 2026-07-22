from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

MotionLibRunnerFormat = Literal["pyroki", "proto"]
MotionLibRunnerRobot = Literal["astro"]

MOTION_FORMAT_CHOICES = ("pyroki", "proto")
ROBOT_CHOICES = ("astro",)


@dataclass(frozen=True)
class MotionLibRunResult:
    """Published output of the source-to-reference-motion pipeline."""

    assembled_motion: Any
    written_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transform source motion clips and publish one assembled versioned "
            "reference motion artifact."
        )
    )
    parser.add_argument(
        "--motion-files",
        required=True,
        type=Path,
        help="Input motion file or flat directory of source motion files.",
    )
    parser.add_argument(
        "--format",
        choices=MOTION_FORMAT_CHOICES,
        default="pyroki",
        dest="motion_format",
        help="Input motion source format.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="Source motion FPS.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=50.0,
        help="Output reference motion FPS after resampling.",
    )
    parser.add_argument(
        "--robot",
        choices=ROBOT_CHOICES,
        default="astro",
        help="Robot model used for simulator enrichment.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device used for motion tensors."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Destination versioned reference motion artifact.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing destination artifact.",
    )
    return parser.parse_args(argv)


def _load_motion_lib_classes() -> tuple[type[Any], type[Any]]:
    from mjlab_playground.motion_lib.motion_lib import MotionLib, MotionLibCfg

    return MotionLib, MotionLibCfg


def _load_writer_class() -> type[Any]:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    return ReferenceMotionNpzWriter


def _load_reference_motion_class() -> type[Any]:
    from mjlab_playground.motion_lib.motion_loader import ReferenceMotion

    return ReferenceMotion


def _motion_identity(motion: Any) -> str:
    name = getattr(motion, "name", None)
    return str(name) if name else "<unnamed clip>"


def run_pipeline(
    motion_files: str | Path,
    *,
    output_file: str | Path,
    motion_format: MotionLibRunnerFormat = "pyroki",
    source_fps: float = 30.0,
    output_fps: float = 50.0,
    robot: MotionLibRunnerRobot = "astro",
    device: str = "cpu",
    overwrite: bool = False,
    motion_lib_cls: type[Any] | None = None,
    motion_lib_cfg_cls: type[Any] | None = None,
    writer_cls: type[Any] | None = None,
) -> MotionLibRunResult:
    """Transform source clips and publish one assembled reference motion."""
    if motion_lib_cls is None or motion_lib_cfg_cls is None:
        default_motion_lib_cls, default_motion_lib_cfg_cls = _load_motion_lib_classes()
    else:
        default_motion_lib_cls = None
        default_motion_lib_cfg_cls = None
    if motion_lib_cls is None:
        assert default_motion_lib_cls is not None
        motion_lib_cls = default_motion_lib_cls
    if motion_lib_cfg_cls is None:
        assert default_motion_lib_cfg_cls is not None
        motion_lib_cfg_cls = default_motion_lib_cfg_cls

    cfg = motion_lib_cfg_cls(
        source_format=motion_format,
        source_fps=source_fps,
        output_fps=output_fps,
        robot=robot,
        device=device,
    )
    motion_lib = motion_lib_cls(cfg)

    source_path = Path(motion_files)
    try:
        source_motions = list(motion_lib.load(motion_files))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load source motions from {source_path}: {exc}"
        ) from exc
    rich_motions: list[Any] = []
    for clip_index, source_motion in enumerate(source_motions):
        identity = _motion_identity(source_motion)
        try:
            resampled_motion = motion_lib.resample(source_motion)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resample clip {clip_index} ({identity}): {exc}"
            ) from exc
        try:
            rich_motion = motion_lib.enrich(resampled_motion)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to enrich clip {clip_index} ({identity}): {exc}"
            ) from exc
        rich_motions.append(rich_motion)

    ordered_identities = ", ".join(
        f"{clip_index}: {_motion_identity(motion)}"
        for clip_index, motion in enumerate(rich_motions)
    )
    try:
        assembled_motion = _load_reference_motion_class().from_clips(
            rich_motions,
            device=device,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed during reference motion assembly for clips in loader order "
            f"[{ordered_identities}]: {exc}"
        ) from exc
    written_path = Path(output_file)
    writer_cls = writer_cls or _load_writer_class()
    try:
        writer_cls().write(assembled_motion, written_path, overwrite=overwrite)
    except Exception as exc:
        raise RuntimeError(
            "Failed to write versioned reference motion artifact to "
            f"{written_path}: {exc}"
        ) from exc
    return MotionLibRunResult(
        assembled_motion=assembled_motion,
        written_path=written_path,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    motion_lib_cls: type[Any] | None = None,
    motion_lib_cfg_cls: type[Any] | None = None,
    writer_cls: type[Any] | None = None,
) -> MotionLibRunResult:
    args = parse_args(argv)
    result = run_pipeline(
        args.motion_files,
        output_file=args.output_file,
        motion_format=args.motion_format,
        source_fps=args.source_fps,
        output_fps=args.output_fps,
        robot=args.robot,
        device=args.device,
        overwrite=args.overwrite,
        motion_lib_cls=motion_lib_cls,
        motion_lib_cfg_cls=motion_lib_cfg_cls,
        writer_cls=writer_cls,
    )
    print(f"wrote {result.written_path}")
    return result


if __name__ == "__main__":
    main()
