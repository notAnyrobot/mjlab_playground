from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Sequence

from mjlab_playground.motion_lib.motion_viewer import (
    MotionViewer,
    TerminalStatusReporter,
)
from mjlab_playground.motion_lib.mujoco_scene_adapter import (
    MujocoSceneAdapter,
    MujocoSceneAdapterCfg,
)
from mjlab_playground.motion_lib.viewers import (
    VIEWER_MODE_CHOICES,
    create_viewer_adapter,
    resolve_viewer_mode,
)

__all__ = [
    "MotionViewer",
    "MotionViewerVerificationError",
    "TerminalStatusReporter",
    "build_scene_adapter",
    "main",
    "parse_args",
    "verify_motion_viewer_path",
]

MOTION_FORMAT_CHOICES = ("pyroki", "proto", "mjlab")
ROBOT_CHOICES = ("astro",)


class MotionViewerVerificationError(RuntimeError):
    """Bounded viewer verification failed at a labeled workflow phase."""


def _load_motion_loader_module() -> Any:
    module_name = "_mjlab_playground_motion_viewer_motion_loader"
    if module_name in sys.modules:
        return sys.modules[module_name]

    loader_path = Path(__file__).resolve().parents[1] / "motion_loader.py"
    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load motion loader from {loader_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def _load_recording_module() -> Any:
    module_name = "_mjlab_playground_motion_viewer_recording"
    if module_name in sys.modules:
        return sys.modules[module_name]

    recording_path = Path(__file__).resolve().parents[1] / "recording.py"
    spec = importlib.util.spec_from_file_location(module_name, recording_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load recording helpers from {recording_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def _load_astro_constants_module() -> Any:
    module_name = "_mjlab_playground_motion_viewer_astro_constants"
    if module_name in sys.modules:
        return sys.modules[module_name]

    src_path = Path(__file__).resolve().parents[2]
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


def build_scene_adapter(
    robot: str,
    *,
    output_fps: float,
    device: str = "cpu",
) -> MujocoSceneAdapter:
    if robot != "astro":
        raise ValueError(
            f"Unsupported robot {robot!r}. Supported robots: {', '.join(ROBOT_CHOICES)}"
        )

    robot_cfg = _load_astro_constants_module().get_astro_robot_cfg()
    mujoco_scene_cfg = MujocoSceneAdapterCfg(
        robot_cfg=robot_cfg,
        output_fps=output_fps,
        device=device,
    )
    mujoco_scene_adapter = MujocoSceneAdapter(mujoco_scene_cfg)
    return mujoco_scene_adapter


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View reference motions with native MuJoCo or browser-based Viser."
    )
    parser.add_argument(
        "--motion-files",
        required=True,
        type=Path,
        help="Input motion file or flat directory of motion files.",
    )
    parser.add_argument(
        "--format",
        choices=MOTION_FORMAT_CHOICES,
        default="pyroki",
        dest="motion_format",
        help="Motion source format.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Motion playback FPS.")
    parser.add_argument(
        "--robot",
        choices=ROBOT_CHOICES,
        default="astro",
        help="Robot model to view.",
    )
    parser.add_argument("--device", default="cpu", help="Device used for scene state.")
    parser.add_argument(
        "--viewer",
        choices=VIEWER_MODE_CHOICES,
        default="auto",
        help=(
            "Interactive presentation: auto selects Viser on macOS and Linux "
            "without a display, or native MuJoCo on Linux with a display."
        ),
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help=(
            "Enable one-shot '\\' recording of the selected full clip in a "
            "background child process while interactive playback continues."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Record every loaded clip deterministically and exit without launching "
            "an interactive viewer. Requires --record-video."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override recording output directory. Defaults to sibling "
            "renderings/<timestamp>/ next to the motion directory."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Load motions, build the selected robot scene, apply one frame, "
            "and exit without launching the interactive viewer."
        ),
    )
    args = parser.parse_args(argv)
    if args.headless and not args.record_video:
        parser.error("--headless requires --record-video in v1")
    if args.output_dir is not None and not args.record_video:
        parser.error("--output-dir requires --record-video")
    if args.smoke_test and args.record_video:
        parser.error("--smoke-test cannot be combined with --record-video")
    if (args.headless or args.smoke_test) and args.viewer != "auto":
        parser.error(
            "explicit --viewer selection requires interactive presentation; "
            "use --viewer auto with --headless or --smoke-test"
        )
    return args


def _load_reference_motions(
    motion_files: str | Path,
    *,
    motion_format: str,
    fps: float,
    device: str,
) -> list[Any]:
    MotionLoader = _load_motion_loader_module().MotionLoader

    return MotionLoader.load(
        motion_files,
        motion_format=motion_format,
        fps=fps,
        device=device,
    )


def verify_motion_viewer_path(
    motion_files: str | Path,
    *,
    motion_format: str = "pyroki",
    fps: float = 30.0,
    robot: str = "astro",
    device: str = "cpu",
    load_motions: Callable[..., Sequence[Any]] = _load_reference_motions,
    scene_adapter_builder: Callable[..., Any] = build_scene_adapter,
) -> None:
    """Load, construct, apply one frame, and return without launching a viewer."""
    if motion_format not in MOTION_FORMAT_CHOICES:
        raise MotionViewerVerificationError(
            f"Unsupported motion format {motion_format!r}. "
            f"Supported formats: {', '.join(MOTION_FORMAT_CHOICES)}"
        )

    try:
        motions = list(
            load_motions(
                motion_files,
                motion_format=motion_format,
                fps=fps,
                device=device,
            )
        )
    except NotImplementedError as exc:
        raise MotionViewerVerificationError(
            f"Unsupported motion format {motion_format!r}: {exc}"
        ) from exc
    except Exception as exc:
        raise MotionViewerVerificationError(
            f"Failed to load {motion_format} motion source data: {exc}"
        ) from exc

    if not motions:
        raise MotionViewerVerificationError(
            f"Failed to load {motion_format} motion source data: "
            "no reference motions were produced"
        )

    if robot not in ROBOT_CHOICES:
        raise MotionViewerVerificationError(
            f"Unsupported robot selection {robot!r}. "
            f"Supported robots: {', '.join(ROBOT_CHOICES)}"
        )

    try:
        scene_adapter = scene_adapter_builder(robot, output_fps=fps, device=device)
    except Exception as exc:
        raise MotionViewerVerificationError(
            f"Failed to construct {robot} scene adapter: {exc}"
        ) from exc

    try:
        scene_adapter.apply_reference_frame(motions[0], 0)
    except ValueError as exc:
        if robot == "astro" and (
            "DOF count" in str(exc) or "DOFs from robot joint order" in str(exc)
        ):
            raise MotionViewerVerificationError(
                f"Incompatible Astro reference motion DOF count: {exc}"
            ) from exc
        raise MotionViewerVerificationError(
            f"Failed to apply one reference motion frame to {robot} scene: {exc}"
        ) from exc
    except Exception as exc:
        raise MotionViewerVerificationError(
            f"Failed to apply one reference motion frame to {robot} scene: {exc}"
        ) from exc


def _recording_motion_names(motions: Sequence[Any]) -> list[str]:
    names = []
    for motion in motions:
        name = getattr(motion, "name", None)
        if not name:
            raise ValueError("loaded reference motions must expose names for recording")
        names.append(str(name))
    return names


def _display_is_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _report_viewer_selection(mode: str) -> None:
    print(f"Selected viewer: {mode}", flush=True)


def _create_deterministic_recorder(scene: Any, **kwargs: Any) -> Any:
    return _load_recording_module().create_mjlab_deterministic_recorder(scene, **kwargs)


def _create_background_recorder(**kwargs: Any) -> Any:
    return _load_recording_module().create_subprocess_background_recorder(**kwargs)


def main(
    argv: Sequence[str] | None = None,
    *,
    load_motions: Callable[..., Sequence[Any]] = _load_reference_motions,
    scene_adapter_builder: Callable[..., Any] = build_scene_adapter,
    motion_viewer_factory: Callable[..., MotionViewer] = MotionViewer,
    verify_viewer_path: Callable[..., None] = verify_motion_viewer_path,
    viewer_adapter_factory: Callable[..., Any] = create_viewer_adapter,
    deterministic_recorder_factory: Callable[..., Any] = _create_deterministic_recorder,
    background_recorder_factory: Callable[..., Any] = _create_background_recorder,
    platform_name: str | None = None,
    display_available: bool | None = None,
    report_viewer_selection: Callable[[str], None] = _report_viewer_selection,
) -> None:
    args = parse_args(argv)
    if args.smoke_test:
        verify_viewer_path(
            args.motion_files,
            motion_format=args.motion_format,
            fps=args.fps,
            robot=args.robot,
            device=args.device,
        )
        return

    resolved_viewer_mode: str | None = None
    if not args.headless:
        resolved_viewer_mode = resolve_viewer_mode(
            args.viewer,
            platform_name=sys.platform if platform_name is None else platform_name,
            display_available=(
                _display_is_available()
                if display_available is None
                else display_available
            ),
        )
        report_viewer_selection(resolved_viewer_mode)

    motions = list(
        load_motions(
            args.motion_files,
            motion_format=args.motion_format,
            fps=args.fps,
            device=args.device,
        )
    )
    scene = scene_adapter_builder(args.robot, output_fps=args.fps, device=args.device)

    if args.headless:
        status_reporter = TerminalStatusReporter()
        recording_module = _load_recording_module()
        output_plan = recording_module.plan_recording_outputs(
            args.motion_files,
            output_dir=args.output_dir,
            motion_names=_recording_motion_names(motions),
        )
        deterministic_recorder = deterministic_recorder_factory(
            scene,
            status_reporter=status_reporter,
        )
        motion_viewer_factory(
            motions,
            scene,
            deterministic_recorder=deterministic_recorder,
        ).record(output_plan.as_request())
        return

    status_reporter = TerminalStatusReporter()
    adapter_options: dict[str, Any] = {
        "status_reporter": status_reporter,
        "recording_enabled": args.record_video,
    }
    background_recorder = None
    if args.record_video:
        recording_module = _load_recording_module()
        output_plan = recording_module.plan_recording_outputs(
            args.motion_files,
            output_dir=args.output_dir,
            motion_names=_recording_motion_names(motions),
        )
        background_recorder = background_recorder_factory(
            targets=output_plan.as_request().targets,
            motion_format=args.motion_format,
            fps=args.fps,
            robot=args.robot,
            device=args.device,
        )

    assert resolved_viewer_mode is not None
    viewer_adapter = viewer_adapter_factory(
        resolved_viewer_mode,
        scene=scene,
        **adapter_options,
    )
    motion_viewer_factory(
        motions,
        scene,
        viewer_adapter=viewer_adapter,
        background_recorder=background_recorder,
        status_reporter=status_reporter,
    ).run()


if __name__ == "__main__":
    main()
