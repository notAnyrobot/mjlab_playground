from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import torch

from mjlab_playground.motion_lib.motion_viewer import SceneAdapter

__all__ = [
    "AstroMujocoSceneAdapter",
    "ROBOT_CHOICES",
    "build_scene_adapter",
]

ROBOT_CHOICES = ("astro",)


def _load_astro_constants_module() -> Any:
    module_name = "_mjlab_playground_motion_viewer_astro_constants"
    if module_name in sys.modules:
        return sys.modules[module_name]

    src_path = Path(__file__).resolve().parents[1]
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
