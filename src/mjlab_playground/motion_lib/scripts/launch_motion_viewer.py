from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from mjlab_playground.motion_lib.motion_viewer import (
    PLAYBACK_SPEEDS,
    PlaybackController,
    SceneAdapter,
    TerminalStatusReporter,
)
from mjlab_playground.motion_lib.motion_viewer import (
    MotionViewer as BaseMotionViewer,
)
from mjlab_playground.motion_lib.mujoco_scene_adapter import (
    MujocoSceneAdapter,
    MujocoSceneAdapterCfg,
)

__all__ = [
    "DEFAULT_CAMERA_CONFIG",
    "ViewerCameraConfig",
    "MotionViewer",
    "MotionViewerVerificationError",
    "OffscreenFrameRenderer",
    "PLAYBACK_SPEEDS",
    "PlaybackController",
    "SceneAdapter",
    "TerminalStatusReporter",
    "build_scene_adapter",
    "create_motion_viewer",
    "main",
    "parse_args",
    "verify_motion_viewer_path",
]

MOTION_FORMAT_CHOICES = ("pyroki", "proto", "mjlab")
ROBOT_CHOICES = ("astro",)


@dataclass(frozen=True)
class ViewerCameraConfig:
    distance: float = 2.0
    elevation: float = -5.0
    azimuth: float = 20.0


DEFAULT_CAMERA_CONFIG = ViewerCameraConfig()


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
) -> SceneAdapter:
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


class OffscreenFrameRenderer:
    """Lazy frame-renderer adapter for interactive recording."""

    def __init__(
        self,
        scene_adapter: SceneAdapter,
        *,
        offscreen_renderer_cls: Any | None = None,
        viewer_config_cls: Any | None = None,
    ) -> None:
        if offscreen_renderer_cls is None or viewer_config_cls is None:
            from mjlab.viewer import ViewerConfig
            from mjlab.viewer.offscreen_renderer import OffscreenRenderer

            offscreen_renderer_cls = OffscreenRenderer
            viewer_config_cls = ViewerConfig

        cfg = viewer_config_cls(
            height=480,
            width=640,
            origin_type=viewer_config_cls.OriginType.ASSET_ROOT,
            entity_name="robot",
            distance=DEFAULT_CAMERA_CONFIG.distance,
            elevation=DEFAULT_CAMERA_CONFIG.elevation,
            azimuth=DEFAULT_CAMERA_CONFIG.azimuth,
        )
        sim = scene_adapter.sim  # type: ignore[attr-defined]
        self._renderer = offscreen_renderer_cls(
            model=scene_adapter.mj_model,
            cfg=cfg,
            scene=scene_adapter.scene,  # type: ignore[attr-defined]
            sim_model=getattr(sim, "model", None),
        )
        self._sim_data = sim.data
        self._renderer.initialize()

    def render_frame(self) -> Any:
        self._renderer.update(self._sim_data)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


class MotionViewer(BaseMotionViewer):
    """Launch-runner viewer with concrete renderer defaults."""

    def __init__(self, motions: Sequence[Any], scene_adapter: SceneAdapter) -> None:
        super().__init__(
            motions,
            scene_adapter,
            frame_renderer_factory=lambda: OffscreenFrameRenderer(scene_adapter),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View reference motions in MuJoCo.")
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
        "--record-video",
        action="store_true",
        help=(
            "Enable '\\' recording selection. Stopping an interactive recording "
            "pauses viewer rendering, records the selected MP4 headlessly, "
            "then resumes the same viewer."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Record all loaded motions and exit without launching a passive viewer.",
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


def create_motion_viewer(
    motion_files: str | Path,
    *,
    motion_format: str = "pyroki",
    fps: float = 30.0,
    robot: str = "astro",
    device: str = "cpu",
    load_motions: Callable[..., Sequence[Any]] = _load_reference_motions,
    scene_adapter_builder: Callable[..., SceneAdapter] = build_scene_adapter,
) -> MotionViewer:
    motions = list(
        load_motions(
            motion_files,
            motion_format=motion_format,
            fps=fps,
            device=device,
        )
    )
    scene_adapter = scene_adapter_builder(robot, output_fps=fps, device=device)
    return MotionViewer(motions, scene_adapter)


def verify_motion_viewer_path(
    motion_files: str | Path,
    *,
    motion_format: str = "pyroki",
    fps: float = 30.0,
    robot: str = "astro",
    device: str = "cpu",
    load_motions: Callable[..., Sequence[Any]] = _load_reference_motions,
    scene_adapter_builder: Callable[..., SceneAdapter] = build_scene_adapter,
) -> MotionViewer:
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

    viewer = MotionViewer(motions, scene_adapter)
    try:
        viewer.render_current_frame()
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

    return viewer


def _recording_motion_names(motions: Sequence[Any]) -> list[str]:
    names = []
    for motion in motions:
        name = getattr(motion, "name", None)
        if not name:
            name = getattr(motion, "display_name", None)
        if not name:
            raise ValueError("loaded reference motions must expose names for recording")
        names.append(str(name))
    return names


def _build_interactive_recording_executor(
    *,
    targets: Sequence[Any],
    motion_format: str,
    fps: float,
    robot: str,
    device: str,
) -> Callable[[int, Path], None]:
    """Run selected interactive recordings in a child process to isolate GL contexts."""

    def recording_executor(motion_index: int, output_path: Path) -> None:
        target = targets[motion_index]
        command = [
            sys.executable,
            "-m",
            "mjlab_playground.motion_lib.scripts.launch_motion_viewer",
            "--motion-files",
            str(target.source_path),
            "--format",
            motion_format,
            "--fps",
            str(float(fps)),
            "--robot",
            robot,
            "--device",
            device,
            "--headless",
            "--record-video",
            "--output-dir",
            str(Path(output_path).parent),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "interactive recording subprocess failed with exit code "
                f"{result.returncode}: {' '.join(command)}"
            )

    return recording_executor


def _viewer_handle_configurator(viewer: Any) -> Callable[[Any], None] | None:
    scene_adapter = getattr(viewer, "scene_adapter", None)
    configure_tracking_camera = getattr(
        scene_adapter, "configure_tracking_camera", None
    )
    if configure_tracking_camera is None:
        return None

    def configure(viewer_handle: Any) -> None:
        configure_tracking_camera(viewer_handle, DEFAULT_CAMERA_CONFIG)

    return configure


def _format_camera_value(value: Any) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _viewer_handle_camera_status(viewer_handle: Any) -> str:
    cam = viewer_handle.cam
    return (
        f"camera distance={_format_camera_value(cam.distance)} "
        f"elevation={_format_camera_value(cam.elevation)} "
        f"azimuth={_format_camera_value(cam.azimuth)}"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    create_viewer: Callable[..., MotionViewer] = create_motion_viewer,
    run_viewer: Callable[[MotionViewer], None] | None = None,
    verify_viewer_path: Callable[..., MotionViewer] = verify_motion_viewer_path,
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

    viewer = create_viewer(
        args.motion_files,
        motion_format=args.motion_format,
        fps=args.fps,
        robot=args.robot,
        device=args.device,
    )
    if args.headless:
        status_reporter = TerminalStatusReporter()
        recording_module = _load_recording_module()
        output_plan = recording_module.plan_recording_outputs(
            args.motion_files,
            output_dir=args.output_dir,
            motion_names=_recording_motion_names(viewer.motions),
        )
        viewer.record_headless(
            status_reporter=status_reporter,
            recording_attachment_factory=recording_module.RecordingAttachment,
            recording_output_paths=[
                target.output_path for target in output_plan.targets
            ],
            frame_renderer_factory=lambda: OffscreenFrameRenderer(viewer.scene_adapter),
        )
        return

    if run_viewer is None:

        def run_viewer(built_viewer: MotionViewer) -> None:
            status_reporter = TerminalStatusReporter()
            if args.record_video:
                recording_module = _load_recording_module()
                output_plan = recording_module.plan_recording_outputs(
                    args.motion_files,
                    output_dir=args.output_dir,
                    motion_names=_recording_motion_names(built_viewer.motions),
                )
                built_viewer.run_interactive(
                    status_reporter=status_reporter,
                    recording_output_paths=[
                        target.output_path for target in output_plan.targets
                    ],
                    recording_executor=_build_interactive_recording_executor(
                        targets=output_plan.targets,
                        motion_format=args.motion_format,
                        fps=args.fps,
                        robot=args.robot,
                        device=args.device,
                    ),
                    viewer_handle_configurator=_viewer_handle_configurator(
                        built_viewer
                    ),
                    viewer_handle_status_provider=_viewer_handle_camera_status,
                )
                return

            viewer_handle_configurator = _viewer_handle_configurator(built_viewer)
            if viewer_handle_configurator is None:
                built_viewer.run_interactive(
                    status_reporter=status_reporter,
                    viewer_handle_status_provider=_viewer_handle_camera_status,
                )
            else:
                built_viewer.run_interactive(
                    status_reporter=status_reporter,
                    viewer_handle_configurator=viewer_handle_configurator,
                    viewer_handle_status_provider=_viewer_handle_camera_status,
                )

    run_viewer(viewer)


if __name__ == "__main__":
    main()
