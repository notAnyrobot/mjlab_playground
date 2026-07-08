from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from mjlab_playground.motion_lib.motion_loader import ReferenceFrame, ReferenceMotion
from mjlab_playground.motion_lib.mujoco_scene_adapter import (
    MujocoSceneAdapter,
    MujocoSceneAdapterCfg,
)


def _identity_root_rot(num_frames: int) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_frames, 1)


def _source_motion(*, dof_count: int = 2, with_velocities: bool = True) -> ReferenceMotion:
    fields = {
        "name": "walk_001",
        "display_name": "Walk 001",
        "fps": 60.0,
        "root_pos": torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ),
        "root_rot": _identity_root_rot(3),
        "dof_pos": torch.tensor(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        )[:, :dof_count],
    }
    if with_velocities:
        fields.update(
            root_lin_vel=torch.tensor(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ]
            ),
            root_ang_vel=torch.tensor(
                [
                    [1.1, 1.2, 1.3],
                    [1.4, 1.5, 1.6],
                    [1.7, 1.8, 1.9],
                ]
            ),
            dof_vel=torch.tensor(
                [
                    [2.1, 2.2],
                    [2.3, 2.4],
                    [2.5, 2.6],
                ]
            )[:, :dof_count],
        )
    return ReferenceMotion(**fields)


@dataclass
class BodyFrame:
    pos: torch.Tensor
    rot: torch.Tensor
    lin_vel: torch.Tensor
    ang_vel: torch.Tensor


class FakeRobotData:
    def __init__(
        self,
        body_frame: BodyFrame,
        *,
        root_state: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
    ) -> None:
        self.default_root_state = torch.tensor(
            [[10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.0, 0.7, 0.8, 0.9, 1.7, 1.8, 1.9]]
        )
        self.default_joint_pos = torch.tensor([[9.0, 8.0, 7.0]])
        self.default_joint_vel = torch.tensor([[0.9, 0.8, 0.7]])
        self._body_frame = body_frame
        self._root_state = (
            root_state.clone() if root_state is not None else self.default_root_state
        )
        self.joint_pos = (
            joint_pos.clone() if joint_pos is not None else self.default_joint_pos
        )
        self.joint_vel = (
            joint_vel.clone() if joint_vel is not None else self.default_joint_vel
        )

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        return self._root_state[:, 0:3]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        return self._root_state[:, 3:7]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        return self._root_state[:, 7:10]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        return self._root_state[:, 10:13]

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        return self._body_frame.pos

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        return self._body_frame.rot

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        return self._body_frame.lin_vel

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        return self._body_frame.ang_vel


class FakeRobot:
    joint_names = ("hip", "knee")
    body_names = ("torso_link", "pelvis", "left_elbow_link")

    def __init__(self) -> None:
        self.indexing = SimpleNamespace(root_body_id=42)
        self.frame_outputs = [_body_frame(10.0), _body_frame(20.0), _body_frame(30.0)]
        self.frame_index = 0
        self.data = FakeRobotData(self.frame_outputs[self.frame_index])
        self.root_states: list[torch.Tensor] = []
        self.joint_positions: list[torch.Tensor] = []
        self.joint_velocities: list[torch.Tensor] = []
        self.joint_ids_seen: list[list[int] | None] = []

    def find_joints(
        self,
        name_keys: tuple[str, ...],
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        assert preserve_order is True
        name_to_id = {name: index for index, name in enumerate(self.joint_names)}
        matched = [name for name in name_keys if name in name_to_id]
        return [name_to_id[name] for name in matched], matched

    def write_root_state_to_sim(self, root_state: torch.Tensor) -> None:
        self.root_states.append(root_state.clone())

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: list[int] | None = None,
    ) -> None:
        self.joint_positions.append(position.clone())
        self.joint_velocities.append(velocity.clone())
        self.joint_ids_seen.append(list(joint_ids) if joint_ids is not None else None)


class FakeScene:
    created_cfg = None
    initialized_with = None
    current_instance = None

    def __init__(self, scene_cfg, device: str) -> None:
        self.scene_cfg = scene_cfg
        self.device = device
        self.robot = FakeRobot()
        self.env_origins = torch.tensor([[100.0, 200.0, 300.0]])
        self.update_dts: list[float] = []
        FakeScene.created_cfg = scene_cfg
        FakeScene.current_instance = self

    def compile(self):
        return "compiled-model"

    def initialize(self, mj_model, model, data) -> None:
        FakeScene.initialized_with = (mj_model, model, data)

    def __getitem__(self, key: str) -> FakeRobot:
        assert key == "robot"
        return self.robot

    def update(self, dt: float) -> None:
        self.update_dts.append(dt)


class FakeSceneCfg:
    def __init__(
        self,
        *,
        num_envs: int,
        entities: dict[str, object],
        terrain: object | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.entities = entities
        self.terrain = terrain


class FakeMujocoCfg:
    timestep = 0.002


class FakeSimulationCfg:
    def __init__(self) -> None:
        self.mujoco = FakeMujocoCfg()


class FakeSimulation:
    def __init__(self, *, num_envs: int, cfg, model, device: str) -> None:
        self.num_envs = num_envs
        self.cfg = cfg
        self.compiled_model = model
        self.device = device
        self.model = "warp-model"
        self.data = FakeBatchedData(
            qpos=torch.tensor([[1.0, 2.0, 3.0]]),
            qvel=torch.tensor([[0.1, 0.2, 0.3]]),
            ctrl=torch.tensor([[0.7, 0.8]]),
            mocap_pos=torch.tensor([[[9.0, 10.0, 11.0]]]),
            mocap_quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            xfrc_applied=torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]]),
        )
        self.mj_data = SimpleNamespace(
            qpos=torch.zeros(3).numpy(),
            qvel=torch.zeros(3).numpy(),
            ctrl=torch.zeros(2).numpy(),
            mocap_pos=torch.zeros(1, 3).numpy(),
            mocap_quat=torch.zeros(1, 4).numpy(),
            xfrc_applied=torch.zeros(1, 6).numpy(),
        )
        self.mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=cfg.mujoco.timestep))
        self.forward_calls = 0

    def forward(self) -> None:
        scene = FakeScene.current_instance
        assert scene is not None
        robot = scene.robot
        robot.frame_index = self.forward_calls
        robot.data = FakeRobotData(
            robot.frame_outputs[self.forward_calls],
            root_state=robot.root_states[-1],
            joint_pos=robot.joint_positions[-1],
            joint_vel=robot.joint_velocities[-1],
        )
        self.forward_calls += 1


@dataclass
class FakeBatchedData:
    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor
    mocap_pos: torch.Tensor
    mocap_quat: torch.Tensor
    xfrc_applied: torch.Tensor


class FakeWarpBridgeLikeData:
    qpos = torch.tensor([[1.0, 2.0, 3.0]])
    qvel = torch.tensor([[0.1, 0.2, 0.3]])
    ctrl = torch.tensor([[0.7, 0.8]])

    @property
    def mocap_pos(self):
        raise TypeError("Cannot convert <class 'warp._src.types.vec3f'> to a Torch type")

    @property
    def mocap_quat(self):
        raise TypeError("Cannot convert <class 'warp._src.types.quatf'> to a Torch type")


@pytest.fixture(autouse=True)
def _patch_mjlab_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MujocoSceneAdapter,
        "_load_mjlab_defaults",
        staticmethod(
            lambda: (
                FakeScene,
                FakeSceneCfg,
                FakeSimulation,
                FakeSimulationCfg,
                None,
            )
        ),
    )


def _body_frame(start: float) -> BodyFrame:
    return BodyFrame(
        pos=torch.tensor(
            [
                [
                    [start + 0.0, start + 0.1, start + 0.2],
                    [start + 1.0, start + 1.1, start + 1.2],
                    [start + 2.0, start + 2.1, start + 2.2],
                ]
            ]
        ),
        rot=torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        ),
        lin_vel=torch.tensor(
            [
                [
                    [start + 3.0, start + 3.1, start + 3.2],
                    [start + 4.0, start + 4.1, start + 4.2],
                    [start + 5.0, start + 5.1, start + 5.2],
                ]
            ]
        ),
        ang_vel=torch.tensor(
            [
                [
                    [start + 6.0, start + 6.1, start + 6.2],
                    [start + 7.0, start + 7.1, start + 7.2],
                    [start + 8.0, start + 8.1, start + 8.2],
                ]
            ]
        ),
    )


def _adapter(*, output_fps: float = 50.0) -> MujocoSceneAdapter:
    cfg = MujocoSceneAdapterCfg(
        robot_cfg="robot-cfg",
        output_fps=output_fps,
        device="cpu",
    )
    return MujocoSceneAdapter(cfg)


def test_adapter_constructs_one_env_scene_and_sets_mujoco_timestep() -> None:
    adapter = _adapter(output_fps=25.0)

    created_cfg = FakeScene.created_cfg
    assert created_cfg is not None
    assert created_cfg.num_envs == 1
    assert created_cfg.entities == {"robot": "robot-cfg"}
    assert created_cfg.terrain is None
    assert adapter.sim.num_envs == 1
    assert adapter.sim.cfg.mujoco.timestep == pytest.approx(0.04)
    assert adapter.sim.mj_model.opt.timestep == pytest.approx(0.04)
    assert FakeScene.initialized_with == (
        adapter.sim.mj_model,
        adapter.sim.model,
        adapter.sim.data,
    )


def test_apply_frame_writes_default_backed_state_and_updates_scene() -> None:
    adapter = _adapter()
    frame = ReferenceFrame(
        root_pos=torch.tensor([1.0, 2.0, 3.0]),
        root_rot=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        root_lin_vel=torch.tensor([0.1, 0.2, 0.3]),
        root_ang_vel=torch.tensor([1.1, 1.2, 1.3]),
        dof_pos=torch.tensor([0.4, 0.5]),
        dof_vel=torch.tensor([2.1, 2.2]),
    )

    adapter.apply_frame(frame)

    robot = adapter.robot
    torch.testing.assert_close(
        robot.root_states[0],
        torch.tensor([[101.0, 202.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.1, 1.2, 1.3]]),
    )
    torch.testing.assert_close(robot.joint_positions[0], torch.tensor([[0.4, 0.5, 7.0]]))
    torch.testing.assert_close(robot.joint_velocities[0], torch.tensor([[2.1, 2.2, 0.7]]))
    assert robot.joint_ids_seen[0] is None
    assert adapter.sim.forward_calls == 1
    assert adapter.scene.update_dts == [0.02]


def test_apply_frame_uses_default_velocities_when_optional_fields_are_missing() -> None:
    adapter = _adapter()
    frame = ReferenceFrame(
        root_pos=torch.tensor([1.0, 2.0, 3.0]),
        root_rot=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        dof_pos=torch.tensor([0.4, 0.5]),
    )

    adapter.apply_frame(frame)

    torch.testing.assert_close(
        adapter.robot.root_states[0],
        torch.tensor([[101.0, 202.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.7, 0.8, 0.9, 1.7, 1.8, 1.9]]),
    )
    torch.testing.assert_close(
        adapter.robot.joint_velocities[0],
        torch.tensor([[0.9, 0.8, 0.7]]),
    )


def test_apply_frame_rejects_dof_count_mismatch() -> None:
    adapter = _adapter()
    frame = ReferenceFrame(
        root_pos=torch.tensor([1.0, 2.0, 3.0]),
        root_rot=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        dof_pos=torch.tensor([0.4]),
    )

    with pytest.raises(ValueError, match="expected 2 DOFs from robot joint order, got 1"):
        adapter.apply_frame(frame)


def test_apply_uses_reference_motion_frame() -> None:
    adapter = _adapter()

    adapter.apply(_source_motion(), 1)

    torch.testing.assert_close(adapter.robot.joint_positions[0], torch.tensor([[0.3, 0.4, 7.0]]))


def test_mujoco_scene_adapter_does_not_own_motion_enrichment() -> None:
    assert not hasattr(MujocoSceneAdapter, "enrich")


def test_read_robot_state_reads_full_robot_state_after_frame_application() -> None:
    adapter = _adapter()
    source_motion = _source_motion()

    adapter.apply_frame(source_motion.frame(1))
    frame = adapter.read_robot_state()

    assert isinstance(frame, ReferenceFrame)
    torch.testing.assert_close(frame.root_pos, source_motion.root_pos[1])
    torch.testing.assert_close(frame.root_rot, source_motion.root_rot[1])
    torch.testing.assert_close(frame.root_lin_vel, source_motion.root_lin_vel[1])
    torch.testing.assert_close(frame.root_ang_vel, source_motion.root_ang_vel[1])
    torch.testing.assert_close(frame.dof_pos, source_motion.dof_pos[1])
    torch.testing.assert_close(frame.dof_vel, source_motion.dof_vel[1])
    assert frame.body_pos is not None
    assert frame.body_rot is not None
    assert frame.body_lin_vel is not None
    assert frame.body_ang_vel is not None
    assert frame.body_pos.shape == (3, 3)
    assert frame.body_rot.shape == (3, 4)
    assert frame.body_lin_vel.shape == (3, 3)
    assert frame.body_ang_vel.shape == (3, 3)
    torch.testing.assert_close(
        frame.body_pos,
        torch.tensor(
            [[10.0, 10.1, 10.2], [11.0, 11.1, 11.2], [12.0, 12.1, 12.2]]
        ),
    )
    assert adapter.sim.forward_calls == 1
    assert adapter.scene.update_dts == [0.02]


def test_sync_display_data_copies_env_zero_and_skips_empty_warp_vector_fields() -> None:
    adapter = _adapter()
    adapter.sync_display_data()

    assert adapter.sim.mj_data.qpos.tolist() == [1.0, 2.0, 3.0]
    assert adapter.sim.mj_data.qvel.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert adapter.sim.mj_data.ctrl.tolist() == pytest.approx([0.7, 0.8])
    assert adapter.sim.mj_data.mocap_pos.tolist() == [[9.0, 10.0, 11.0]]
    assert adapter.sim.mj_data.mocap_quat.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert adapter.sim.mj_data.xfrc_applied.tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]

    adapter.sim.data = FakeWarpBridgeLikeData()
    adapter.sim.mj_data.mocap_pos = torch.zeros(0, 3).numpy()
    adapter.sim.mj_data.mocap_quat = torch.zeros(0, 4).numpy()

    adapter.sync_display_data()

    assert adapter.sim.mj_data.qpos.tolist() == [1.0, 2.0, 3.0]
    assert adapter.sim.mj_data.qvel.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_configure_tracking_camera_tracks_robot_root_body() -> None:
    adapter = _adapter()
    viewer_handle = SimpleNamespace(
        cam=SimpleNamespace(
            type=None,
            trackbodyid=None,
            fixedcamid=7,
            distance=0.0,
            elevation=0.0,
            azimuth=0.0,
        )
    )
    camera_config = SimpleNamespace(distance=2.0, elevation=-5.0, azimuth=20.0)

    adapter.configure_tracking_camera(viewer_handle, camera_config)

    assert viewer_handle.cam.type == 1
    assert viewer_handle.cam.trackbodyid == 42
    assert viewer_handle.cam.fixedcamid == -1
    assert viewer_handle.cam.distance == 2.0
    assert viewer_handle.cam.elevation == -5.0
    assert viewer_handle.cam.azimuth == 20.0
