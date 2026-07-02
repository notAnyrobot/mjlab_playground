from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from mjlab_playground.motion_lib.astro_scene_adapter import (
    AstroMujocoSceneAdapter,
    build_scene_adapter,
)


class FakeReferenceMotion:
    def __init__(self, frame_count: int, marker: float) -> None:
        self.root_pos = torch.full((frame_count, 3), marker)
        self.root_rot = torch.zeros(frame_count, 4)
        self.dof_pos = torch.zeros(frame_count, 29)


class FakeRobot:
    joint_names = ("hip", "knee", "ankle")

    def __init__(self) -> None:
        self.root_pose: torch.Tensor | None = None
        self.joint_pos: torch.Tensor | None = None
        self.joint_vel: torch.Tensor | None = None
        self.joint_ids = None

    def find_joints(
        self,
        name_keys: tuple[str, ...],
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        assert preserve_order is True
        name_to_id = {name: index for index, name in enumerate(self.joint_names)}
        return [name_to_id[name] for name in name_keys], list(name_keys)

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor) -> None:
        self.root_pose = root_pose

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: list[int],
    ) -> None:
        self.joint_pos = position
        self.joint_vel = velocity
        self.joint_ids = joint_ids


class FakeSimulation:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.model = object()
        self.data = object()
        self.mj_data = SimpleNamespace(
            qpos=torch.zeros(6).numpy(),
            qvel=torch.zeros(6).numpy(),
            ctrl=torch.zeros(2).numpy(),
            mocap_pos=torch.zeros(1, 3).numpy(),
            mocap_quat=torch.zeros(1, 4).numpy(),
            xfrc_applied=torch.zeros(1, 6).numpy(),
        )
        self.mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))

    def forward(self) -> None:
        self.forward_calls += 1


class FakeScene:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot
        self.update_dts: list[float] = []

    def __getitem__(self, key: str) -> FakeRobot:
        assert key == "robot"
        return self.robot

    def update(self, dt: float) -> None:
        self.update_dts.append(dt)


class ConstructedScene:
    created_cfg = None
    initialized_with = None

    def __init__(self, scene_cfg, device: str) -> None:
        self.scene_cfg = scene_cfg
        self.device = device
        self.robot = FakeRobot()
        self.num_envs = scene_cfg.num_envs
        ConstructedScene.created_cfg = scene_cfg

    def compile(self):
        return "compiled-model"

    def initialize(self, mj_model, model, data) -> None:
        ConstructedScene.initialized_with = (mj_model, model, data)

    def __getitem__(self, key: str) -> FakeRobot:
        assert key == "robot"
        return self.robot

    def update(self, dt: float) -> None:
        pass


class ConstructedSceneCfg:
    def __init__(self, *, num_envs: int, entities: dict[str, object]) -> None:
        self.num_envs = num_envs
        self.entities = entities


class ConstructedSimulation:
    def __init__(self, *, num_envs: int, cfg, model, device: str) -> None:
        self.num_envs = num_envs
        self.cfg = cfg
        self.compiled_model = model
        self.device = device
        self.model = "warp-model"
        self.data = "warp-data"
        self.mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=0.02))

    def forward(self) -> None:
        pass


class ConstructedSimulationCfg:
    pass


@dataclass
class FakeBatchedData:
    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor
    mocap_pos: torch.Tensor
    mocap_quat: torch.Tensor
    xfrc_applied: torch.Tensor


class FakeWarpBridgeLikeData:
    qpos = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    qvel = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    ctrl = torch.tensor([[0.7, 0.8]])

    @property
    def mocap_pos(self):
        raise TypeError("Cannot convert <class 'warp._src.types.vec3f'> to a Torch type")

    @property
    def mocap_quat(self):
        raise TypeError("Cannot convert <class 'warp._src.types.quatf'> to a Torch type")


def test_build_scene_adapter_selects_astro_scene_configuration() -> None:
    robot_cfg = object()

    adapter = build_scene_adapter(
        "astro",
        device="cpu",
        scene_cls=ConstructedScene,
        scene_cfg_cls=ConstructedSceneCfg,
        simulation_cls=ConstructedSimulation,
        simulation_cfg_cls=ConstructedSimulationCfg,
        robot_cfg_factory=lambda: robot_cfg,
        joint_names=("hip", "knee", "ankle"),
    )

    assert isinstance(adapter, AstroMujocoSceneAdapter)
    assert ConstructedScene.created_cfg.entities == {"robot": robot_cfg}
    assert ConstructedScene.initialized_with == (
        adapter.sim.mj_model,
        adapter.sim.model,
        adapter.sim.data,
    )


def test_build_scene_adapter_rejects_unknown_robot() -> None:
    try:
        build_scene_adapter("g1")
    except ValueError as exc:
        assert "Unsupported robot 'g1'" in str(exc)
    else:
        raise AssertionError("expected unsupported robot to fail")


def test_astro_scene_adapter_lives_outside_reusable_viewer_module() -> None:
    import mjlab_playground.motion_lib.motion_viewer as reference_viewer

    assert not hasattr(reference_viewer, "AstroMujocoSceneAdapter")
    assert not hasattr(reference_viewer, "build_scene_adapter")


def test_astro_scene_adapter_validates_reference_motion_dof_count() -> None:
    adapter = AstroMujocoSceneAdapter(
        scene=FakeScene(FakeRobot()),
        sim=FakeSimulation(),
        robot=FakeRobot(),
        joint_names=("hip", "knee", "ankle"),
        device="cpu",
    )
    motion = FakeReferenceMotion(frame_count=1, marker=0.0)
    motion.dof_pos = torch.zeros(1, 2)

    try:
        adapter.apply(motion, 0)
    except ValueError as exc:
        assert "Astro reference motion DOF count must be 3, got 2" in str(exc)
    else:
        raise AssertionError("expected DOF-count mismatch to fail")


def test_astro_scene_adapter_applies_one_frame_and_updates_scene() -> None:
    robot = FakeRobot()
    sim = FakeSimulation()
    scene = FakeScene(robot)
    adapter = AstroMujocoSceneAdapter(
        scene=scene,
        sim=sim,
        robot=robot,
        joint_names=("hip", "knee", "ankle"),
        device="cpu",
    )
    motion = FakeReferenceMotion(frame_count=2, marker=0.0)
    motion.root_pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    motion.root_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    motion.dof_pos = torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])

    adapter.apply(motion, 1)

    assert robot.root_pose is not None
    assert robot.joint_pos is not None
    assert robot.joint_vel is not None
    torch.testing.assert_close(
        robot.root_pose,
        torch.tensor([[4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(robot.joint_pos, torch.tensor([[1.1, 1.2, 1.3]]))
    torch.testing.assert_close(robot.joint_vel, torch.zeros(1, 3))
    assert robot.joint_ids == [0, 1, 2]
    assert sim.forward_calls == 1
    assert scene.update_dts == [0.01]


def test_astro_scene_adapter_syncs_env_zero_state_to_mujoco_display_data() -> None:
    robot = FakeRobot()
    sim = FakeSimulation()
    scene = FakeScene(robot)
    sim.data = FakeBatchedData(
        qpos=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        qvel=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        ctrl=torch.tensor([[0.7, 0.8]]),
        mocap_pos=torch.tensor([[[9.0, 10.0, 11.0]]]),
        mocap_quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        xfrc_applied=torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]]),
    )
    adapter = AstroMujocoSceneAdapter(
        scene=scene,
        sim=sim,
        robot=robot,
        joint_names=("hip", "knee", "ankle"),
        device="cpu",
    )

    adapter.sync_display_data()

    assert sim.mj_data.qpos.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert sim.mj_data.qvel.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert sim.mj_data.ctrl.tolist() == pytest.approx([0.7, 0.8])
    assert sim.mj_data.mocap_pos.tolist() == [[9.0, 10.0, 11.0]]
    assert sim.mj_data.mocap_quat.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert sim.mj_data.xfrc_applied.tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]


def test_astro_scene_adapter_skips_empty_optional_display_fields_without_source_access() -> None:
    robot = FakeRobot()
    sim = FakeSimulation()
    scene = FakeScene(robot)
    sim.data = FakeWarpBridgeLikeData()
    sim.mj_data.mocap_pos = torch.zeros(0, 3).numpy()
    sim.mj_data.mocap_quat = torch.zeros(0, 4).numpy()
    adapter = AstroMujocoSceneAdapter(
        scene=scene,
        sim=sim,
        robot=robot,
        joint_names=("hip", "knee", "ankle"),
        device="cpu",
    )

    adapter.sync_display_data()

    assert sim.mj_data.qpos.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert sim.mj_data.qvel.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert sim.mj_data.ctrl.tolist() == pytest.approx([0.7, 0.8])
