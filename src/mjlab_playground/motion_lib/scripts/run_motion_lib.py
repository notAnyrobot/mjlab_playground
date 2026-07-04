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
    """In-memory output of the explicit MotionLib stage pipeline."""

    source_motions: list[Any]
    resampled_motions: list[Any]
    rich_motions: list[Any]
    written_paths: list[Path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MotionLib load -> resample -> enrich reference motion pipeline."
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
    parser.add_argument("--device", default="cpu", help="Device used for motion tensors.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where exported rich reference .npz clips are written.",
    )
    return parser.parse_args(argv)


def _load_motion_lib_classes() -> tuple[type[Any], type[Any]]:
    from mjlab_playground.motion_lib.motion_lib import MotionLib, MotionLibCfg

    return MotionLib, MotionLibCfg


def _load_writer_class() -> type[Any]:
    from mjlab_playground.motion_lib import ReferenceMotionNpzWriter

    return ReferenceMotionNpzWriter


def run_pipeline(
    motion_files: str | Path,
    *,
    motion_format: MotionLibRunnerFormat = "pyroki",
    source_fps: float = 30.0,
    output_fps: float = 50.0,
    robot: MotionLibRunnerRobot = "astro",
    device: str = "cpu",
    output_dir: str | Path | None = None,
    motion_lib_cls: type[Any] | None = None,
    motion_lib_cfg_cls: type[Any] | None = None,
    writer_cls: type[Any] | None = None,
) -> MotionLibRunResult:
    """Run MotionLib's explicit load, resample, and enrich stages."""
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

    source_motions = list(motion_lib.load(motion_files))
    resampled_motions = list(motion_lib.resample(source_motions))
    rich_motions = list(motion_lib.enrich(resampled_motions))
    writer_cls = writer_cls or _load_writer_class()
    writer = writer_cls()
    export_dir = (
        Path(output_dir)
        if output_dir is not None
        else _default_output_dir(motion_files)
    )
    written_paths: list[Path] = []
    for motion in rich_motions:
        output_path = export_dir / _output_filename_for_motion(motion)
        try:
            writer.write(motion, output_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to export {_motion_identity(motion)} to {output_path}: {exc}"
            ) from exc
        written_paths.append(output_path)
    return MotionLibRunResult(
        source_motions=source_motions,
        resampled_motions=resampled_motions,
        rich_motions=rich_motions,
        written_paths=written_paths,
    )


def _output_filename_for_motion(motion: Any) -> str:
    identity = _motion_identity(motion)
    return Path(identity).with_suffix(".npz").name


def _motion_identity(motion: Any) -> str:
    identity = getattr(motion, "name", None) or getattr(motion, "display_name", None)
    if not identity:
        raise ValueError("rich reference motion must expose a name for export")
    return str(identity)


def _default_output_dir(motion_files: str | Path) -> Path:
    source_path = Path(motion_files)
    source_root = source_path if source_path.is_dir() else source_path.parent
    return source_root.parent / "mjlab-astro"


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
        motion_format=args.motion_format,
        source_fps=args.source_fps,
        output_fps=args.output_fps,
        robot=args.robot,
        device=args.device,
        output_dir=args.output_dir,
        motion_lib_cls=motion_lib_cls,
        motion_lib_cfg_cls=motion_lib_cfg_cls,
        writer_cls=writer_cls,
    )
    _report_written_paths(result.written_paths)
    return result


def _report_written_paths(written_paths: Sequence[Path]) -> None:
    for path in written_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
