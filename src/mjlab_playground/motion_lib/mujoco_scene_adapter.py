from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .motion_loader import ReferenceFrame, ReferenceMotion

__all__ = ["MujocoSceneAdapter", "MujocoSceneAdapterCfg"]


@dataclass(frozen=True, kw_only=True)
class MujocoSceneAdapterCfg:
    """Configuration for constructing a one-env MuJoCo reference-motion scene."""

    robot_cfg: Any
    output_fps: float
    device: str | torch.device = "cpu"


class MujocoSceneAdapter:
    """Canonical one-env MuJoCo scene adapter for reference motions."""

    def __init__(
        self,
        cfg: MujocoSceneAdapterCfg,
    ) -> None:
        self.device = torch.device(cfg.device)
        scene_cls, scene_cfg_cls, simulation_cls, simulation_cfg_cls, terrain_cfg = (
            self._load_mjlab_defaults()
        )

        scene_cfg = scene_cfg_cls(
            num_envs=1,
            terrain=terrain_cfg,
            entities={"robot": cfg.robot_cfg},
        )
        self.scene = scene_cls(scene_cfg, device=str(cfg.device))
        model = self.scene.compile()

        sim_cfg = simulation_cfg_cls()
        sim_cfg.mujoco.timestep = 1.0 / float(cfg.output_fps)
        self.sim = simulation_cls(
            num_envs=1,
            cfg=sim_cfg,
            model=model,
            device=str(cfg.device),
        )
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot = self.scene["robot"]
        self.joint_names = tuple(self.robot.joint_names)
        self.joint_ids = self._resolve_joint_ids()
        self.body_names = tuple(getattr(self.robot, "body_names", ()))

    @classmethod
    def from_existing_scene(
        cls,
        *,
        scene: Any,
        sim: Any,
        robot: Any,
        device: str | torch.device = "cpu",
    ) -> "MujocoSceneAdapter":
        """Adapt a prebuilt one-env scene into the canonical adapter behavior."""
        adapter = cls.__new__(cls)
        adapter._initialize_existing_scene(
            scene=scene,
            sim=sim,
            robot=robot,
            device=device,
        )
        return adapter

    def _initialize_existing_scene(
        self,
        *,
        scene: Any,
        sim: Any,
        robot: Any,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.scene = scene
        self.sim = sim
        self.robot = robot
        self.joint_names = tuple(self.robot.joint_names)
        self.joint_ids = self._resolve_joint_ids()
        self.body_names = tuple(getattr(self.robot, "body_names", ()))

    @property
    def mj_model(self) -> Any:
        return self.sim.mj_model

    @property
    def mj_data(self) -> Any:
        return self.sim.mj_data

    def apply(self, motion: ReferenceMotion, frame_index: int) -> None:
        """Apply one frame from a reference motion for viewer playback."""
        self.apply_frame(motion.frame(frame_index))

    def apply_frame(self, frame: ReferenceFrame) -> None:
        """Write one reference frame and update the scene."""
        dof_width = int(frame.dof_pos.shape[-1])
        expected_dofs = len(self.joint_names)
        if dof_width != expected_dofs:
            raise ValueError(
                "MujocoSceneAdapter expected "
                f"{expected_dofs} DOFs from robot joint order, got {dof_width}"
            )

        root_state = self.robot.data.default_root_state.clone()
        root_state[:, 0:3] = self._row(frame.root_pos)
        root_state[:, :2] += self.scene.env_origins[:, :2]
        root_state[:, 3:7] = self._row(frame.root_rot)
        if frame.root_lin_vel is not None:
            root_state[:, 7:10] = self._row(frame.root_lin_vel)
        if frame.root_ang_vel is not None:
            root_state[:, 10:13] = self._row(frame.root_ang_vel)
        self.robot.write_root_state_to_sim(root_state)

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        joint_pos[:, self.joint_ids] = self._row(frame.dof_pos)
        if frame.dof_vel is not None:
            joint_vel[:, self.joint_ids] = self._row(frame.dof_vel)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        self.sim.forward()
        self.scene.update(float(self.sim.mj_model.opt.timestep))

    def read_robot_state(self) -> ReferenceFrame:
        """Read the current MuJoCo robot state into one reference frame."""
        return ReferenceFrame(
            root_pos=self._root_pos_frame(),
            root_rot=self._env0_frame("root_link_quat_w", width=4),
            dof_pos=self._joint_frame("joint_pos"),
            root_lin_vel=self._env0_frame("root_link_lin_vel_w", width=3),
            root_ang_vel=self._env0_frame("root_link_ang_vel_w", width=3),
            dof_vel=self._joint_frame("joint_vel"),
            body_pos=self._body_frame("body_link_pos_w", width=3),
            body_rot=self._body_frame("body_link_quat_w", width=4),
            body_lin_vel=self._body_frame("body_link_lin_vel_w", width=3),
            body_ang_vel=self._body_frame("body_link_ang_vel_w", width=3),
        )

    def sync_display_data(self) -> None:
        """Copy env-0 simulation data into MuJoCo display data."""
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

    def configure_tracking_camera(self, viewer_handle: Any, camera_config: Any) -> None:
        """Configure a passive viewer camera to track the robot root body."""
        import mujoco

        cam = viewer_handle.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        cam.trackbodyid = int(self.robot.indexing.root_body_id)
        cam.fixedcamid = -1
        cam.distance = float(camera_config.distance)
        cam.elevation = float(camera_config.elevation)
        cam.azimuth = float(camera_config.azimuth)

    def _resolve_joint_ids(self) -> list[int]:
        joint_ids, matched_names = self.robot.find_joints(
            self.joint_names,
            preserve_order=True,
        )
        if tuple(matched_names) != self.joint_names:
            raise ValueError(
                "MujocoSceneAdapter robot joint order mismatch: "
                f"expected {self.joint_names}, got {tuple(matched_names)}"
            )
        return list(joint_ids)

    def _body_frame(self, field_name: str, *, width: int) -> torch.Tensor:
        try:
            values = getattr(self.robot.data, field_name)
        except AttributeError as exc:
            raise AttributeError(
                f"MujocoSceneAdapter robot data is missing {field_name}"
            ) from exc

        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.ndim != 3:
            raise ValueError(
                f"{field_name} must have shape (N, B, D), got {tuple(tensor.shape)}"
            )
        if int(tensor.shape[-1]) != width:
            raise ValueError(
                f"{field_name} must have last dimension {width}, "
                f"got {tensor.shape[-1]}"
            )
        if self.body_names and int(tensor.shape[1]) != len(self.body_names):
            raise ValueError(
                f"{field_name} body count must match robot.body_names "
                f"({len(self.body_names)}), got {tensor.shape[1]}"
            )
        return tensor[0].clone()

    def _root_pos_frame(self) -> torch.Tensor:
        root_pos = self._env0_frame("root_link_pos_w", width=3)
        env_origins = torch.as_tensor(
            self.scene.env_origins,
            dtype=root_pos.dtype,
            device=root_pos.device,
        )
        if env_origins.ndim != 2 or int(env_origins.shape[-1]) < 2:
            raise ValueError(
                "scene.env_origins must have shape (N, >=2), "
                f"got {tuple(env_origins.shape)}"
            )
        root_pos = root_pos.clone()
        root_pos[:2] -= env_origins[0, :2]
        return root_pos

    def _joint_frame(self, field_name: str) -> torch.Tensor:
        values = self._env0_frame(field_name, width=None)
        max_joint_id = max(self.joint_ids, default=-1)
        if max_joint_id >= int(values.shape[0]):
            raise ValueError(
                f"{field_name} must contain resolved joint id {max_joint_id}, "
                f"got {values.shape[0]}"
            )
        return values[self.joint_ids].clone()

    def _env0_frame(self, field_name: str, *, width: int | None) -> torch.Tensor:
        try:
            values = getattr(self.robot.data, field_name)
        except AttributeError as exc:
            raise AttributeError(
                f"MujocoSceneAdapter robot data is missing {field_name}"
            ) from exc

        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.ndim != 2:
            raise ValueError(
                f"{field_name} must have shape (N, D), got {tuple(tensor.shape)}"
            )
        if width is not None and int(tensor.shape[-1]) != width:
            raise ValueError(
                f"{field_name} must have last dimension {width}, "
                f"got {tensor.shape[-1]}"
            )
        return tensor[0].clone()

    def _row(self, values: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(
            1, -1
        )

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

    @staticmethod
    def _load_mjlab_defaults() -> tuple[Any, Any, Any, Any, Any]:
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import Simulation, SimulationCfg
        from mjlab.terrains import TerrainEntityCfg

        terrain_cfg = TerrainEntityCfg(terrain_type="plane")
        return Scene, SceneCfg, Simulation, SimulationCfg, terrain_cfg
