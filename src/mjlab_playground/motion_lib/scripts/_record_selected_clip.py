from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from mjlab_playground.motion_lib.motion_viewer import (
    MotionViewer,
    TerminalStatusReporter,
)
from mjlab_playground.motion_lib.recording import (
    RecordingRequest,
    RecordingRequestTarget,
)
from mjlab_playground.motion_lib.scripts import launch_motion_viewer


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Internal selected reference-clip recording child."
    )
    parser.add_argument("--motion-file", required=True, type=Path)
    parser.add_argument(
        "--format",
        choices=launch_motion_viewer.MOTION_FORMAT_CHOICES,
        required=True,
        dest="motion_format",
    )
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument(
        "--robot",
        choices=launch_motion_viewer.ROBOT_CHOICES,
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--clip-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.clip_id < 0:
        parser.error("--clip-id must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    motions = launch_motion_viewer._load_reference_motions(
        args.motion_file,
        motion_format=args.motion_format,
        fps=args.fps,
        device=args.device,
    )
    if len(motions) != 1:
        raise ValueError(
            "selected-clip recording child requires one source artifact to load "
            "exactly one reference motion"
        )

    scene = launch_motion_viewer.build_scene_adapter(
        args.robot,
        output_fps=args.fps,
        device=args.device,
    )
    status_reporter = TerminalStatusReporter()
    deterministic_recorder = launch_motion_viewer._create_deterministic_recorder(
        scene,
        status_reporter=status_reporter,
    )
    request = RecordingRequest(
        targets=(
            RecordingRequestTarget(
                reference_motion_index=0,
                clip_id=args.clip_id,
                source_path=args.motion_file,
                output_path=args.output,
            ),
        )
    )
    MotionViewer(
        motions,
        scene,
        deterministic_recorder=deterministic_recorder,
    ).record(request)


if __name__ == "__main__":
    main()
