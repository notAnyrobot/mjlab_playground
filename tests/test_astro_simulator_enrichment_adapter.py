from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    ROOT / "src" / "mjlab_playground" / "motion_lib" / "astro_enrichment_adapter.py"
)


def _load_adapter_class():
    spec = importlib.util.spec_from_file_location(
        "astro_enrichment_adapter_under_test",
        ADAPTER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.AstroSimulatorEnrichmentAdapter


AstroSimulatorEnrichmentAdapter = _load_adapter_class()


def _identity_root_rot(num_frames: int) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_frames, 1)


@dataclass(frozen=True, kw_only=True)
class FakeReferenceMotionState:
    root_pos: torch.Tensor
    root_rot: torch.Tensor
    dof_pos: torch.Tensor
    name: str | None = None
    display_name: str | None = None
    fps: float = 30.0
    root_lin_vel: torch.Tensor | None = None
    root_ang_vel: torch.Tensor | None = None
    dof_vel: torch.Tensor | None = None
    body_pos: torch.Tensor | None = None
    body_rot: torch.Tensor | None = None
    body_lin_vel: torch.Tensor | None = None
    body_ang_vel: torch.Tensor | None = None
    body_contacts: torch.Tensor | None = None
    foot_contacts: torch.Tensor | None = None


def _source_motion(*, dof_count: int = 2) -> FakeReferenceMotionState:
    return FakeReferenceMotionState(
        name="walk_001",
        display_name="Walk 001",
        fps=60.0,
        root_pos=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ),
        root_rot=_identity_root_rot(3),
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
        dof_pos=torch.tensor(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        )[:, :dof_count],
        dof_vel=torch.tensor(
            [
                [2.1, 2.2],
                [2.3, 2.4],
                [2.5, 2.6],
            ]
        )[:, :dof_count],
    )


@dataclass
class BodyFrame:
    pos: torch.Tensor
    rot: torch.Tensor
    lin_vel: torch.Tensor
    ang_vel: torch.Tensor


class FakeRobotData:
    def __init__(self, body_frame: BodyFrame) -> None:
        self._body_frame = body_frame

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
    body_names = ("torso_link", "pelvis", "left_elbow_link")
    joint_names = ("hip", "knee")

    def __init__(self, frame_outputs: list[BodyFrame]) -> None:
        self.frame_outputs = frame_outputs
        self.frame_index = 0
        self.data = FakeRobotData(self.frame_outputs[self.frame_index])
        self.root_poses: list[torch.Tensor] = []
        self.root_velocities: list[torch.Tensor] = []
        self.joint_positions: list[torch.Tensor] = []
        self.joint_velocities: list[torch.Tensor] = []
        self.joint_ids_seen: list[list[int]] = []

    def find_bodies(
        self,
        name_keys: tuple[str, ...],
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        assert preserve_order is True
        name_to_id = {name: index for index, name in enumerate(self.body_names)}
        matched = [name for name in name_keys if name in name_to_id]
        return [name_to_id[name] for name in matched], matched

    def find_joints(
        self,
        name_keys: tuple[str, ...],
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        assert preserve_order is True
        name_to_id = {name: index for index, name in enumerate(self.joint_names)}
        matched = [name for name in name_keys if name in name_to_id]
        return [name_to_id[name] for name in matched], matched

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor) -> None:
        self.root_poses.append(root_pose.clone())

    def write_root_link_velocity_to_sim(self, root_velocity: torch.Tensor) -> None:
        self.root_velocities.append(root_velocity.clone())

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: list[int],
    ) -> None:
        self.joint_positions.append(position.clone())
        self.joint_velocities.append(velocity.clone())
        self.joint_ids_seen.append(list(joint_ids))


class FakeSimulation:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot
        self.forward_calls = 0
        self.mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=0.02))

    def forward(self) -> None:
        self.robot.frame_index = self.forward_calls
        self.robot.data = FakeRobotData(self.robot.frame_outputs[self.forward_calls])
        self.forward_calls += 1


class FakeScene:
    def __init__(self) -> None:
        self.update_dts: list[float] = []

    def update(self, dt: float) -> None:
        self.update_dts.append(dt)


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


def test_astro_enrichment_adapter_enriches_one_clip_with_canonical_body_order() -> None:
    frame_outputs = [_body_frame(10.0), _body_frame(20.0), _body_frame(30.0)]
    robot = FakeRobot(frame_outputs)
    scene = FakeScene()
    sim = FakeSimulation(robot)
    adapter = AstroSimulatorEnrichmentAdapter(
        scene=scene,
        sim=sim,
        robot=robot,
        joint_names=("hip", "knee"),
        body_names=("pelvis", "torso_link"),
        device="cpu",
    )

    rich = adapter.enrich(_source_motion())

    assert rich.name == "walk_001"
    assert rich.display_name == "Walk 001"
    assert rich.fps == 60.0
    assert rich.root_pos is not None
    torch.testing.assert_close(rich.root_pos, _source_motion().root_pos)
    torch.testing.assert_close(rich.dof_pos, _source_motion().dof_pos)
    torch.testing.assert_close(rich.dof_vel, _source_motion().dof_vel)
    assert rich.body_pos is not None
    assert rich.body_rot is not None
    assert rich.body_lin_vel is not None
    assert rich.body_ang_vel is not None
    assert rich.body_pos.shape == (3, 2, 3)
    assert rich.body_rot.shape == (3, 2, 4)
    assert rich.body_lin_vel.shape == (3, 2, 3)
    assert rich.body_ang_vel.shape == (3, 2, 3)
    torch.testing.assert_close(
        rich.body_pos,
        torch.tensor(
            [
                [[11.0, 11.1, 11.2], [10.0, 10.1, 10.2]],
                [[21.0, 21.1, 21.2], [20.0, 20.1, 20.2]],
                [[31.0, 31.1, 31.2], [30.0, 30.1, 30.2]],
            ]
        ),
    )
    torch.testing.assert_close(
        rich.body_rot,
        torch.tensor(
            [
                [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            ]
        ),
    )
    assert sim.forward_calls == 3
    assert scene.update_dts == [0.02, 0.02, 0.02]
    assert len(robot.root_poses) == 3
    torch.testing.assert_close(
        robot.root_velocities[1],
        torch.tensor([[0.4, 0.5, 0.6, 1.4, 1.5, 1.6]]),
    )
    torch.testing.assert_close(robot.joint_velocities[2], torch.tensor([[2.5, 2.6]]))
    assert robot.joint_ids_seen == [[0, 1], [0, 1], [0, 1]]


def test_astro_enrichment_adapter_uses_zero_velocities_when_optional_fields_are_missing() -> None:
    robot = FakeRobot([_body_frame(10.0), _body_frame(20.0), _body_frame(30.0)])
    adapter = AstroSimulatorEnrichmentAdapter(
        scene=FakeScene(),
        sim=FakeSimulation(robot),
        robot=robot,
        joint_names=("hip", "knee"),
        body_names=("pelvis", "torso_link"),
        device="cpu",
    )
    motion = FakeReferenceMotionState(
        root_pos=_source_motion().root_pos,
        root_rot=_source_motion().root_rot,
        dof_pos=_source_motion().dof_pos,
        fps=60.0,
    )

    adapter.enrich(motion)

    torch.testing.assert_close(robot.root_velocities[0], torch.zeros(1, 6))
    torch.testing.assert_close(robot.joint_velocities[0], torch.zeros(1, 2))


def test_astro_enrichment_adapter_rejects_dof_count_mismatch() -> None:
    robot = FakeRobot([_body_frame(10.0), _body_frame(20.0), _body_frame(30.0)])
    adapter = AstroSimulatorEnrichmentAdapter(
        scene=FakeScene(),
        sim=FakeSimulation(robot),
        robot=robot,
        joint_names=("hip", "knee"),
        body_names=("pelvis", "torso_link"),
        device="cpu",
    )

    with pytest.raises(ValueError, match="expected 2 DOFs from joint contract, got 1"):
        adapter.enrich(_source_motion(dof_count=1))


def test_astro_enrichment_adapter_rejects_body_contract_mismatch() -> None:
    robot = FakeRobot([_body_frame(10.0), _body_frame(20.0), _body_frame(30.0)])

    with pytest.raises(ValueError, match="Astro body contract mismatch"):
        AstroSimulatorEnrichmentAdapter(
            scene=FakeScene(),
            sim=FakeSimulation(robot),
            robot=robot,
            joint_names=("hip", "knee"),
            body_names=("pelvis", "missing_body"),
            device="cpu",
        )
