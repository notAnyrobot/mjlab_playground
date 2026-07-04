from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import torch

__all__ = ["AstroSimulatorEnrichmentAdapter"]


def _load_astro_constants_module() -> Any:
    module_name = "_mjlab_playground_motion_lib_astro_constants"
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


class AstroSimulatorEnrichmentAdapter:
    """Derive Astro body trajectories from resampled generalized coordinates."""

    def __init__(
        self,
        *,
        scene: Any,
        sim: Any,
        robot: Any,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        device: str | torch.device,
    ) -> None:
        self.scene = scene
        self.sim = sim
        self.robot = robot
        self.joint_names = tuple(joint_names)
        self.body_names = tuple(body_names)
        self.device = torch.device(device)
        self.joint_ids = self._resolve_joint_ids()
        self.body_ids = self._resolve_body_ids()

    @classmethod
    def create(
        cls,
        *,
        device: str | torch.device = "cpu",
        scene_cls: Any | None = None,
        scene_cfg_cls: Any | None = None,
        simulation_cls: Any | None = None,
        simulation_cfg_cls: Any | None = None,
        robot_cfg_factory: Any | None = None,
        joint_names: Sequence[str] | None = None,
        body_names: Sequence[str] | None = None,
    ) -> "AstroSimulatorEnrichmentAdapter":
        if (
            scene_cls is None
            or scene_cfg_cls is None
            or simulation_cls is None
            or simulation_cfg_cls is None
            or robot_cfg_factory is None
            or joint_names is None
            or body_names is None
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
            body_names = astro_constants.ASTRO_AMP_BODY_NAMES

        scene_cfg = scene_cfg_cls(num_envs=1, entities={"robot": robot_cfg_factory()})
        scene = scene_cls(scene_cfg, device=str(device))
        model = scene.compile()
        sim = simulation_cls(
            num_envs=1,
            cfg=simulation_cfg_cls(),
            model=model,
            device=str(device),
        )
        scene.initialize(sim.mj_model, sim.model, sim.data)
        return cls(
            scene=scene,
            sim=sim,
            robot=scene["robot"],
            joint_names=joint_names,
            body_names=body_names,
            device=device,
        )

    def enrich(self, motion: Any) -> Any:
        frame_count = int(motion.root_pos.shape[0])
        body_pos: list[torch.Tensor] = []
        body_rot: list[torch.Tensor] = []
        body_lin_vel: list[torch.Tensor] = []
        body_ang_vel: list[torch.Tensor] = []

        for frame_index in range(frame_count):
            self._apply_frame(motion, frame_index)
            self.sim.forward()
            self.scene.update(float(self.sim.mj_model.opt.timestep))

            body_pos.append(self._body_frame("body_link_pos_w", frame_index, width=3))
            body_rot.append(self._body_frame("body_link_quat_w", frame_index, width=4))
            body_lin_vel.append(
                self._body_frame("body_link_lin_vel_w", frame_index, width=3)
            )
            body_ang_vel.append(
                self._body_frame("body_link_ang_vel_w", frame_index, width=3)
            )

        return type(motion)(
            root_pos=motion.root_pos,
            root_rot=motion.root_rot,
            dof_pos=motion.dof_pos,
            name=motion.name,
            display_name=motion.display_name,
            fps=motion.fps,
            root_lin_vel=motion.root_lin_vel,
            root_ang_vel=motion.root_ang_vel,
            dof_vel=motion.dof_vel,
            body_pos=torch.stack(body_pos, dim=0),
            body_rot=torch.stack(body_rot, dim=0),
            body_lin_vel=torch.stack(body_lin_vel, dim=0),
            body_ang_vel=torch.stack(body_ang_vel, dim=0),
            body_contacts=motion.body_contacts,
            foot_contacts=motion.foot_contacts,
        )

    def _apply_frame(self, motion: Any, frame_index: int) -> None:
        root_pos = self._frame(motion.root_pos, frame_index, field_name="root_pos")
        root_rot = self._frame(motion.root_rot, frame_index, field_name="root_rot")
        dof_pos = self._frame(motion.dof_pos, frame_index, field_name="dof_pos")
        expected_dofs = len(self.joint_names)
        actual_dofs = int(dof_pos.shape[-1])
        if actual_dofs != expected_dofs:
            raise ValueError(
                "Astro enrichment expected "
                f"{expected_dofs} DOFs from joint contract, got {actual_dofs}"
            )

        self.robot.write_root_link_pose_to_sim(
            torch.cat((root_pos, root_rot), dim=-1).reshape(1, 7)
        )
        self.robot.write_root_link_velocity_to_sim(
            torch.cat(
                (
                    self._optional_velocity_frame(
                        motion,
                        "root_lin_vel",
                        frame_index,
                        width=3,
                    ),
                    self._optional_velocity_frame(
                        motion,
                        "root_ang_vel",
                        frame_index,
                        width=3,
                    ),
                ),
                dim=-1,
            ).reshape(1, 6)
        )
        self.robot.write_joint_state_to_sim(
            dof_pos.reshape(1, expected_dofs),
            self._optional_velocity_frame(
                motion,
                "dof_vel",
                frame_index,
                width=expected_dofs,
            ).reshape(1, expected_dofs),
            joint_ids=self.joint_ids,
        )

    def _body_frame(self, field_name: str, frame_index: int, *, width: int) -> torch.Tensor:
        try:
            values = getattr(self.robot.data, field_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Astro enrichment robot data is missing {field_name}"
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
        try:
            return tensor[0, self.body_ids].clone()
        except IndexError as exc:
            raise ValueError(
                "Astro body contract cannot be collected from robot data at "
                f"frame {frame_index}: requested body ids {self.body_ids}, "
                f"data shape {tuple(tensor.shape)}"
            ) from exc

    def _resolve_joint_ids(self) -> list[int]:
        joint_ids, matched_names = self.robot.find_joints(
            self.joint_names,
            preserve_order=True,
        )
        if tuple(matched_names) != self.joint_names:
            raise ValueError(
                "Astro joint contract mismatch: "
                f"expected {self.joint_names}, got {tuple(matched_names)}"
            )
        return list(joint_ids)

    def _resolve_body_ids(self) -> list[int]:
        body_ids, matched_names = self.robot.find_bodies(
            self.body_names,
            preserve_order=True,
        )
        if tuple(matched_names) != self.body_names:
            raise ValueError(
                "Astro body contract mismatch: "
                f"expected {self.body_names}, got {tuple(matched_names)}"
            )
        return list(body_ids)

    def _frame(self, values: Any, frame_index: int, *, field_name: str) -> torch.Tensor:
        try:
            frame = values[frame_index]
        except IndexError as exc:
            raise IndexError(f"{field_name} has no frame {frame_index}") from exc
        return torch.as_tensor(frame, dtype=torch.float32, device=self.device)

    def _optional_velocity_frame(
        self,
        motion: Any,
        field_name: str,
        frame_index: int,
        *,
        width: int,
    ) -> torch.Tensor:
        values = getattr(motion, field_name)
        if values is None:
            return torch.zeros(width, dtype=torch.float32, device=self.device)
        frame = self._frame(values, frame_index, field_name=field_name)
        actual_width = int(frame.shape[-1])
        if actual_width != width:
            raise ValueError(
                f"{field_name} frame must have {width} values, got {actual_width}"
            )
        return frame
