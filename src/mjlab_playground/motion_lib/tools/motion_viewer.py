from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, TextIO

import torch

MOTION_FORMAT_CHOICES = ("pyroki", "proto")
ROBOT_CHOICES = ("astro",)
PLAYBACK_SPEEDS = (0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DEFAULT_PLAYBACK_SPEED_INDEX = PLAYBACK_SPEEDS.index(1.0)
STATUS_PROGRESS_BAR_WIDTH = 20
PlaybackAction = str
KEY_TO_PLAYBACK_ACTION: dict[int, PlaybackAction] = {
    32: "space",
    262: "right",
    263: "left",
    264: "down",
    265: "up",
}


class MotionViewerVerificationError(RuntimeError):
    """Bounded viewer verification failed at a labeled workflow phase."""


class SceneAdapter(Protocol):
    """Minimal scene boundary used by the source-agnostic viewer."""

    @property
    def mj_model(self) -> Any:
        """MuJoCo model rendered by the passive viewer."""

    @property
    def mj_data(self) -> Any:
        """MuJoCo data rendered by the passive viewer."""

    def apply(self, motion: Any, frame_index: int) -> None:
        """Apply one reference motion frame to the scene."""

    def sync_display_data(self) -> None:
        """Copy the latest applied scene state into ``mj_data`` for display."""


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
        package_stub.MJLAB_PLAYGROUND_SRC_PATH = src_path
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


class AstroMujocoSceneAdapter:
    """Apply reference motion frames to a one-env Astro MuJoCo scene."""

    def __init__(
        self,
        *,
        scene: Any,
        sim: Any,
        robot: Any,
        joint_names: Sequence[str],
        device: str,
    ) -> None:
        self.scene = scene
        self.sim = sim
        self.robot = robot
        self.joint_names = tuple(joint_names)
        self.device = device
        self.joint_ids = self._resolve_joint_ids()

    @property
    def mj_model(self) -> Any:
        return self.sim.mj_model

    @property
    def mj_data(self) -> Any:
        return self.sim.mj_data

    @classmethod
    def create(
        cls,
        *,
        device: str = "cpu",
        scene_cls: Any | None = None,
        scene_cfg_cls: Any | None = None,
        simulation_cls: Any | None = None,
        simulation_cfg_cls: Any | None = None,
        robot_cfg_factory: Any | None = None,
        joint_names: Sequence[str] | None = None,
    ) -> "AstroMujocoSceneAdapter":
        if (
            scene_cls is None
            or scene_cfg_cls is None
            or simulation_cls is None
            or simulation_cfg_cls is None
            or robot_cfg_factory is None
            or joint_names is None
        ):
            astro_constants = _load_astro_constants_module()
            from mjlab.scene import Scene, SceneCfg
            from mjlab.sim import Simulation, SimulationCfg

            scene_cls = Scene
            scene_cfg_cls = SceneCfg
            simulation_cls = Simulation
            simulation_cfg_cls = SimulationCfg
            robot_cfg_factory = astro_constants.get_astro_robot_cfg
            joint_names = astro_constants.ASTRO_JOINT_NAMES

        scene_cfg = scene_cfg_cls(num_envs=1, entities={"robot": robot_cfg_factory()})
        scene = scene_cls(scene_cfg, device=device)
        model = scene.compile()
        sim = simulation_cls(
            num_envs=1,
            cfg=simulation_cfg_cls(),
            model=model,
            device=device,
        )
        scene.initialize(sim.mj_model, sim.model, sim.data)
        return cls(
            scene=scene,
            sim=sim,
            robot=scene["robot"],
            joint_names=joint_names,
            device=device,
        )

    def apply(self, motion: Any, frame_index: int) -> None:
        dof_frame = self._frame(motion.dof_pos, frame_index, field_name="dof_pos")
        expected_dofs = len(self.joint_names)
        actual_dofs = int(dof_frame.shape[-1])
        if actual_dofs != expected_dofs:
            raise ValueError(
                "Astro reference motion DOF count must be "
                f"{expected_dofs}, got {actual_dofs}"
            )

        root_pos = self._frame(motion.root_pos, frame_index, field_name="root_pos")
        root_rot = self._frame(motion.root_rot, frame_index, field_name="root_rot")
        if int(root_pos.shape[-1]) != 3:
            raise ValueError(f"root_pos frame must have 3 values, got {root_pos.shape[-1]}")
        if int(root_rot.shape[-1]) != 4:
            raise ValueError(f"root_rot frame must have 4 values, got {root_rot.shape[-1]}")

        root_pose = torch.cat((root_pos, root_rot), dim=-1).reshape(1, 7)
        joint_pos = dof_frame.reshape(1, expected_dofs)
        joint_vel = torch.zeros_like(joint_pos)

        self.robot.write_root_link_pose_to_sim(root_pose)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self.joint_ids)
        self.sim.forward()
        self.scene.update(float(self.sim.mj_model.opt.timestep))

    def sync_display_data(self) -> None:
        sim_data = self.sim.data
        mj_data = self.sim.mj_data
        self._copy_env0(sim_data.qpos, mj_data.qpos)
        self._copy_env0(sim_data.qvel, mj_data.qvel)
        for field_name in ("ctrl", "mocap_pos", "mocap_quat", "xfrc_applied"):
            self._copy_optional_env0(sim_data, mj_data, field_name)

        try:
            import mujoco
        except ImportError:
            return
        try:
            mujoco.mj_forward(self.sim.mj_model, mj_data)
        except TypeError:
            return

    def _resolve_joint_ids(self) -> list[int]:
        joint_ids, matched_names = self.robot.find_joints(
            self.joint_names,
            preserve_order=True,
        )
        if tuple(matched_names) != self.joint_names:
            raise ValueError(
                "Astro robot joint mapping mismatch: "
                f"expected {self.joint_names}, got {tuple(matched_names)}"
            )
        return list(joint_ids)

    @staticmethod
    def _copy_env0(source: Any, target: Any) -> None:
        source_env0 = source[0]
        if hasattr(source_env0, "detach"):
            source_env0 = source_env0.detach()
        if hasattr(source_env0, "cpu"):
            source_env0 = source_env0.cpu()
        if hasattr(source_env0, "numpy"):
            source_env0 = source_env0.numpy()
        target[...] = source_env0

    def _copy_optional_env0(self, sim_data: Any, mj_data: Any, field_name: str) -> None:
        target = getattr(mj_data, field_name, None)
        if target is None or int(getattr(target, "size", 0)) == 0:
            return

        try:
            source = getattr(sim_data, field_name)
        except (AttributeError, TypeError):
            return

        self._copy_env0(source, target)

    def _frame(self, values: Any, frame_index: int, *, field_name: str) -> torch.Tensor:
        try:
            frame = values[frame_index]
        except IndexError as exc:
            raise IndexError(f"{field_name} has no frame {frame_index}") from exc
        return torch.as_tensor(frame, dtype=torch.float32, device=self.device)


def build_scene_adapter(
    robot: str,
    *,
    device: str = "cpu",
    scene_cls: Any | None = None,
    scene_cfg_cls: Any | None = None,
    simulation_cls: Any | None = None,
    simulation_cfg_cls: Any | None = None,
    robot_cfg_factory: Any | None = None,
    joint_names: Sequence[str] | None = None,
) -> SceneAdapter:
    if robot == "astro":
        return AstroMujocoSceneAdapter.create(
            device=device,
            scene_cls=scene_cls,
            scene_cfg_cls=scene_cfg_cls,
            simulation_cls=simulation_cls,
            simulation_cfg_cls=simulation_cfg_cls,
            robot_cfg_factory=robot_cfg_factory,
            joint_names=joint_names,
        )
    raise ValueError(f"Unsupported robot {robot!r}. Supported robots: {', '.join(ROBOT_CHOICES)}")


class PlaybackController:
    """Pure playback state for source-agnostic reference motion clips."""

    def __init__(self, *, motion_count: int) -> None:
        if motion_count <= 0:
            raise ValueError("motion_count must be positive")
        self.motion_count = motion_count
        self.selected_motion_index = 0
        self.frame_position = 0.0
        self.paused = False
        self.playback_speed_index = DEFAULT_PLAYBACK_SPEED_INDEX

    @property
    def playback_speed(self) -> float:
        return PLAYBACK_SPEEDS[self.playback_speed_index]

    def increase_speed(self) -> None:
        self.playback_speed_index = min(
            self.playback_speed_index + 1,
            len(PLAYBACK_SPEEDS) - 1,
        )

    def decrease_speed(self) -> None:
        self.playback_speed_index = max(self.playback_speed_index - 1, 0)

    def select_next_motion(self) -> None:
        self.selected_motion_index = (self.selected_motion_index + 1) % self.motion_count
        self.reset_frame()

    def select_previous_motion(self) -> None:
        self.selected_motion_index = (self.selected_motion_index - 1) % self.motion_count
        self.reset_frame()

    def reset_frame(self) -> None:
        self.frame_position = 0.0

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def handle_action(self, action: PlaybackAction) -> None:
        if action == "left":
            self.select_previous_motion()
        elif action == "right":
            self.select_next_motion()
        elif action == "up":
            self.increase_speed()
        elif action == "down":
            self.decrease_speed()
        elif action == "space":
            self.toggle_pause()

    def advance(self, *, frame_count: int, frames: float = 1.0) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.paused:
            return
        if not math.isfinite(frames):
            raise ValueError("frames must be finite")
        self.frame_position = (self.frame_position + frames * self.playback_speed) % frame_count

    def current_frame_index(self, *, frame_count: int) -> int:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        return int(self.frame_position) % frame_count


class TerminalStatusReporter:
    """Render playback status as a single updating terminal line."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self._last_status: str | None = None

    def __call__(self, status: str) -> None:
        if status == self._last_status:
            return
        padding = ""
        if self._last_status is not None:
            padding = " " * max(0, len(self._last_status) - len(status))
        print(f"\r{status}{padding}", end="", file=self.stream, flush=True)
        self._last_status = status

    def close(self) -> None:
        if self._last_status is not None:
            print(file=self.stream, flush=True)
            self._last_status = None


class MotionViewer:
    """Source-agnostic reference motion viewer."""

    def __init__(self, motions: Sequence[Any], scene_adapter: SceneAdapter) -> None:
        if not motions:
            raise ValueError("MotionViewer requires at least one reference motion")
        self.motions = list(motions)
        self.scene_adapter = scene_adapter
        self.controller = PlaybackController(motion_count=len(self.motions))

    @property
    def selected_motion(self) -> Any:
        return self.motions[self.controller.selected_motion_index]

    def handle_action(self, action: PlaybackAction) -> None:
        self.controller.handle_action(action)

    def handle_key(self, key: int) -> None:
        action = KEY_TO_PLAYBACK_ACTION.get(key)
        if action is not None:
            self.handle_action(action)

    def playback_status(self) -> str:
        motion_index = self.controller.selected_motion_index
        motion = self.selected_motion
        frame_count = self._frame_count(motion)
        frame_index = self.controller.current_frame_index(frame_count=frame_count)
        state = "paused" if self.controller.paused else "playing"
        percent = 100.0 * frame_index / frame_count
        return (
            f"Motion {motion_index + 1}/{len(self.motions)} "
            f"| {self._motion_name(motion)} "
            f"| {self._progress_bar(frame_index=frame_index, frame_count=frame_count)} "
            f"{frame_index}/{frame_count} ({percent:.1f}%) "
            f"| speed {self.controller.playback_speed:g}x "
            f"| {state}"
        )

    def render_current_frame(self) -> None:
        motion = self.selected_motion
        frame_index = self.controller.current_frame_index(
            frame_count=self._frame_count(motion),
        )
        self.scene_adapter.apply(motion, frame_index)

    def advance(self, *, frames: float = 1.0) -> None:
        self.controller.advance(
            frame_count=self._frame_count(self.selected_motion),
            frames=frames,
        )

    @staticmethod
    def _frame_count(motion: Any) -> int:
        try:
            frame_count = int(motion.root_pos.shape[0])
        except AttributeError as exc:
            raise TypeError("reference motion must expose root_pos frames") from exc
        if frame_count <= 0:
            raise ValueError("reference motion must contain at least one frame")
        return frame_count

    @staticmethod
    def _motion_name(motion: Any) -> str:
        name = getattr(motion, "name", None)
        if name:
            return MotionViewer._compact_motion_name(str(name))
        return "unnamed"

    @staticmethod
    def _compact_motion_name(name: str) -> str:
        suffixes = (
            "_poses_keypoints_retargeted.npz",
            ".npz",
            ".motion",
            ".pt",
        )
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @staticmethod
    def _progress_bar(*, frame_index: int, frame_count: int) -> str:
        filled = round(STATUS_PROGRESS_BAR_WIDTH * frame_index / frame_count)
        filled = max(0, min(STATUS_PROGRESS_BAR_WIDTH, filled))
        empty = STATUS_PROGRESS_BAR_WIDTH - filled
        return f"[{'=' * filled}{'-' * empty}]"

    def run_interactive(
        self,
        *,
        launch_passive: Callable[..., Any] | None = None,
        status_reporter: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        target_refresh_rate: float = 60.0,
    ) -> None:
        if launch_passive is None:
            import mujoco.viewer

            launch_passive = mujoco.viewer.launch_passive

        handle = launch_passive(
            self.scene_adapter.mj_model,
            self.scene_adapter.mj_data,
            key_callback=self.handle_key,
            show_left_ui=False,
            show_right_ui=False,
        )
        if handle is None:
            raise RuntimeError("Failed to launch MuJoCo viewer")

        previous_time = clock()
        min_frame_time = 1.0 / target_refresh_rate if target_refresh_rate > 0.0 else 0.0
        try:
            while handle.is_running():
                self.render_current_frame()
                self.scene_adapter.sync_display_data()
                if status_reporter is not None:
                    status_reporter(self.playback_status())
                handle.sync()

                current_time = clock()
                elapsed_seconds = max(0.0, current_time - previous_time)
                previous_time = current_time
                self.advance(frames=elapsed_seconds * self._motion_fps(self.selected_motion))

                sleep_for = min_frame_time - elapsed_seconds
                if sleep_for > 0.0:
                    sleep(sleep_for)
        finally:
            close_status_reporter = getattr(status_reporter, "close", None)
            if close_status_reporter is not None:
                close_status_reporter()
            handle.close()

    @staticmethod
    def _motion_fps(motion: Any) -> float:
        fps = float(getattr(motion, "fps", 30.0))
        if fps <= 0.0 or not math.isfinite(fps):
            raise ValueError("reference motion fps must be positive and finite")
        return fps


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
    parser.add_argument("--robot", default="astro", help="Robot model to view.")
    parser.add_argument("--device", default="cpu", help="Device used for scene state.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Load motions, build the selected robot scene, apply one frame, "
            "and exit without launching the interactive viewer."
        ),
    )
    return parser.parse_args(argv)


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
    scene_adapter = scene_adapter_builder(robot, device=device)
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
        scene_adapter = scene_adapter_builder(robot, device=device)
    except Exception as exc:
        raise MotionViewerVerificationError(
            f"Failed to construct {robot} scene adapter: {exc}"
        ) from exc

    viewer = MotionViewer(motions, scene_adapter)
    try:
        viewer.render_current_frame()
    except ValueError as exc:
        if robot == "astro" and "DOF count" in str(exc):
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
    if run_viewer is None:
        def run_viewer(built_viewer: MotionViewer) -> None:
            built_viewer.run_interactive(status_reporter=TerminalStatusReporter())

    run_viewer(viewer)


if __name__ == "__main__":
    main()
