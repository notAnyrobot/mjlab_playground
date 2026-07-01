from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MOTION_VIEWER_PATH = (
    ROOT
    / "src"
    / "mjlab_playground"
    / "motion_lib"
    / "tools"
    / "motion_viewer.py"
)


def _load_motion_viewer_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("motion_viewer", MOTION_VIEWER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_motion_viewer = _load_motion_viewer_module()
MotionViewer = _motion_viewer.MotionViewer
AstroMujocoSceneAdapter = _motion_viewer.AstroMujocoSceneAdapter
PlaybackController = _motion_viewer.PlaybackController
PLAYBACK_SPEEDS = _motion_viewer.PLAYBACK_SPEEDS
build_scene_adapter = _motion_viewer.build_scene_adapter


class FakeReferenceMotion:
    def __init__(self, frame_count: int, marker: float) -> None:
        self.root_pos = torch.full((frame_count, 3), marker)
        self.root_rot = torch.zeros(frame_count, 4)
        self.dof_pos = torch.zeros(frame_count, 29)


def _write_pyroki_npz(path: Path, *, dof_count: int = 29) -> None:
    np.savez(
        path,
        base_frame_pos=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        base_frame_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        joint_angles=np.zeros((1, dof_count), dtype=np.float64),
    )


class FakeSceneAdapter:
    def __init__(self) -> None:
        self.applied: list[tuple[FakeReferenceMotion, int]] = []
        self.mj_model = object()
        self.mj_data = object()
        self.display_syncs = 0

    def apply(self, motion: FakeReferenceMotion, frame_index: int) -> None:
        self.applied.append((motion, frame_index))

    def sync_display_data(self) -> None:
        self.display_syncs += 1


class FakeViewerHandle:
    def __init__(self, *, running_checks: int) -> None:
        self.running_checks = running_checks
        self.sync_calls = 0
        self.closed = False

    def is_running(self) -> bool:
        self.running_checks -= 1
        return self.running_checks >= 0

    def sync(self) -> None:
        self.sync_calls += 1

    def close(self) -> None:
        self.closed = True


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
        self.mj_data = type(
            "MjData",
            (),
            {
                "qpos": torch.zeros(6).numpy(),
                "qvel": torch.zeros(6).numpy(),
                "ctrl": torch.zeros(2).numpy(),
                "mocap_pos": torch.zeros(1, 3).numpy(),
                "mocap_quat": torch.zeros(1, 4).numpy(),
                "xfrc_applied": torch.zeros(1, 6).numpy(),
            },
        )()
        self.mj_model = type("MjModel", (), {"opt": type("Opt", (), {"timestep": 0.01})()})()

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
        self.mj_model = "mj-model"
        self.model = "warp-model"
        self.data = "warp-data"
        self.mj_model = type("MjModel", (), {"opt": type("Opt", (), {"timestep": 0.02})()})()

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


def test_playback_controller_uses_fixed_speed_ladder_and_clamps() -> None:
    controller = PlaybackController(motion_count=2)

    assert PLAYBACK_SPEEDS == (0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    assert controller.playback_speed == 1.0

    for _ in range(10):
        controller.decrease_speed()

    assert controller.playback_speed == 0.1
    assert controller.playback_speed_index == 0

    for _ in range(10):
        controller.increase_speed()

    assert controller.playback_speed == 2.0
    assert controller.playback_speed_index == len(PLAYBACK_SPEEDS) - 1


def test_playback_controller_wraps_motion_selection_and_resets_frame() -> None:
    controller = PlaybackController(motion_count=3)
    controller.advance(frame_count=5, frames=2.0)

    controller.select_previous_motion()

    assert controller.selected_motion_index == 2
    assert controller.frame_position == 0.0

    controller.advance(frame_count=5, frames=3.0)
    controller.select_next_motion()

    assert controller.selected_motion_index == 0
    assert controller.frame_position == 0.0


def test_playback_controller_space_action_toggles_pause() -> None:
    controller = PlaybackController(motion_count=1)

    controller.handle_action("space")

    assert controller.paused is True

    controller.handle_action("space")

    assert controller.paused is False


def test_motion_viewer_applies_reference_motion_without_source_metadata() -> None:
    motions = [
        FakeReferenceMotion(frame_count=2, marker=1.0),
        FakeReferenceMotion(frame_count=3, marker=2.0),
    ]
    scene_adapter = FakeSceneAdapter()
    viewer = MotionViewer(motions, scene_adapter)

    viewer.render_current_frame()
    viewer.handle_action("right")
    viewer.render_current_frame()
    viewer.controller.advance(frame_count=3, frames=1.0)
    viewer.render_current_frame()

    assert scene_adapter.applied == [
        (motions[0], 0),
        (motions[1], 0),
        (motions[1], 1),
    ]


def test_motion_viewer_interactive_loop_applies_frames_and_syncs_passive_viewer() -> None:
    motion = FakeReferenceMotion(frame_count=3, marker=1.0)
    motion.fps = 2.0
    scene_adapter = FakeSceneAdapter()
    viewer = MotionViewer([motion], scene_adapter)
    handle = FakeViewerHandle(running_checks=3)
    launched_with = {}
    status_updates: list[str] = []
    clock_values = iter([10.0, 10.5, 11.0, 11.5])

    def launch_passive(model, data, *, key_callback, show_left_ui, show_right_ui):
        launched_with.update(
            model=model,
            data=data,
            key_callback=key_callback,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        )
        return handle

    viewer.run_interactive(
        launch_passive=launch_passive,
        clock=lambda: next(clock_values),
        sleep=lambda _: None,
        status_reporter=status_updates.append,
    )

    assert launched_with == {
        "model": scene_adapter.mj_model,
        "data": scene_adapter.mj_data,
        "key_callback": viewer.handle_key,
        "show_left_ui": False,
        "show_right_ui": False,
    }
    assert scene_adapter.applied == [(motion, 0), (motion, 1), (motion, 2)]
    assert scene_adapter.display_syncs == 3
    assert status_updates == [
        "Motion 1/1 | unnamed | [--------------------] 0/3 (0.0%) | speed 1x | playing",
        "Motion 1/1 | unnamed | [=======-------------] 1/3 (33.3%) | speed 1x | playing",
        "Motion 1/1 | unnamed | [=============-------] 2/3 (66.7%) | speed 1x | playing",
    ]
    assert handle.sync_calls == 3
    assert handle.closed is True


def test_motion_viewer_key_callback_maps_controls_to_playback_actions() -> None:
    motions = [
        FakeReferenceMotion(frame_count=2, marker=1.0),
        FakeReferenceMotion(frame_count=2, marker=2.0),
    ]
    viewer = MotionViewer(motions, FakeSceneAdapter())

    viewer.handle_key(262)
    assert viewer.controller.selected_motion_index == 1

    viewer.handle_key(263)
    assert viewer.controller.selected_motion_index == 0

    viewer.handle_key(265)
    assert viewer.controller.playback_speed == 1.25

    viewer.handle_key(264)
    assert viewer.controller.playback_speed == 1.0

    viewer.handle_key(32)
    assert viewer.controller.paused is True


def test_motion_viewer_reports_current_motion_speed_and_progress() -> None:
    motions = [
        FakeReferenceMotion(frame_count=2, marker=1.0),
        FakeReferenceMotion(frame_count=3, marker=2.0),
    ]
    motions[1].name = "walking.npz"
    viewer = MotionViewer(motions, FakeSceneAdapter())

    viewer.handle_action("right")
    viewer.handle_action("up")
    viewer.advance(frames=1.0)

    assert (
        viewer.playback_status()
        == "Motion 2/2 | walking | [=======-------------] 1/3 (33.3%) | speed 1.25x | playing"
    )

    viewer.handle_action("space")

    assert (
        viewer.playback_status()
        == "Motion 2/2 | walking | [=======-------------] 1/3 (33.3%) | speed 1.25x | paused"
    )


def test_motion_viewer_status_simplifies_pyroki_retargeted_filenames() -> None:
    motion = FakeReferenceMotion(frame_count=984, marker=1.0)
    motion.name = "0007_0007_Walking001_poses_keypoints_retargeted.npz"
    viewer = MotionViewer([motion], FakeSceneAdapter())
    viewer.advance(frames=131.0)

    assert (
        viewer.playback_status()
        == "Motion 1/1 | 0007_0007_Walking001 | [===-----------------] 131/984 (13.3%) | speed 1x | playing"
    )


def test_create_motion_viewer_loads_motions_and_builds_selected_robot_adapter(
    tmp_path: Path,
) -> None:
    create_motion_viewer = _motion_viewer.create_motion_viewer
    motions = [FakeReferenceMotion(frame_count=1, marker=1.0)]
    scene_adapter = FakeSceneAdapter()
    calls = {}

    def load_motions(*args, **kwargs):
        calls["load_motions"] = (args, kwargs)
        return motions

    def build_adapter(*args, **kwargs):
        calls["scene_adapter_builder"] = (args, kwargs)
        return scene_adapter

    viewer = create_motion_viewer(
        tmp_path / "motion.npz",
        motion_format="pyroki",
        fps=50.0,
        robot="astro",
        device="cpu",
        load_motions=load_motions,
        scene_adapter_builder=build_adapter,
    )

    assert viewer.motions == motions
    assert viewer.scene_adapter is scene_adapter
    assert calls["load_motions"] == (
        (tmp_path / "motion.npz",),
        {"motion_format": "pyroki", "fps": 50.0, "device": "cpu"},
    )
    assert calls["scene_adapter_builder"] == (
        ("astro",),
        {"device": "cpu"},
    )


def test_motion_viewer_main_launches_interactive_viewer(tmp_path: Path) -> None:
    motion_viewer_main = _motion_viewer.main
    motion = FakeReferenceMotion(frame_count=1, marker=1.0)
    scene_adapter = FakeSceneAdapter()
    viewer = MotionViewer([motion], scene_adapter)
    calls = {}

    def create_viewer(*args, **kwargs):
        calls["create_viewer"] = (args, kwargs)
        return viewer

    motion_viewer_main(
        ["--motion-files", str(tmp_path / "motion.npz"), "--fps", "50"],
        create_viewer=create_viewer,
        run_viewer=lambda built_viewer: calls.setdefault("run_viewer", built_viewer),
    )

    assert calls["create_viewer"] == (
        (tmp_path / "motion.npz",),
        {
            "motion_format": "pyroki",
            "fps": 50.0,
            "robot": "astro",
            "device": "cpu",
        },
    )
    assert calls["run_viewer"] is viewer


def test_motion_viewer_main_reports_terminal_status_by_default(
    tmp_path: Path,
    capsys,
) -> None:
    motion_viewer_main = _motion_viewer.main

    class FakeInteractiveViewer:
        def run_interactive(self, *, status_reporter):
            status_reporter(
                "Motion 1/1 | walking | [--------------------] 0/3 (0.0%) | speed 1x | playing"
            )

    motion_viewer_main(
        ["--motion-files", str(tmp_path / "motion.npz")],
        create_viewer=lambda *_, **__: FakeInteractiveViewer(),
    )

    assert (
        "\rMotion 1/1 | walking | [--------------------] 0/3 (0.0%) | speed 1x | playing"
        in capsys.readouterr().out
    )


def test_motion_viewer_main_smoke_test_verifies_one_frame_without_interactive_viewer(
    tmp_path: Path,
) -> None:
    motion_viewer_main = _motion_viewer.main
    calls = {}

    def verify_viewer_path(*args, **kwargs):
        calls["verify_viewer_path"] = (args, kwargs)

    def create_viewer(*args, **kwargs):
        raise AssertionError("smoke test must not create an interactive viewer")

    motion_viewer_main(
        [
            "--motion-files",
            str(tmp_path / "motion.npz"),
            "--fps",
            "50",
            "--smoke-test",
        ],
        create_viewer=create_viewer,
        run_viewer=lambda _: calls.setdefault("run_viewer", True),
        verify_viewer_path=verify_viewer_path,
    )

    assert calls == {
        "verify_viewer_path": (
            (tmp_path / "motion.npz",),
            {
                "motion_format": "pyroki",
                "fps": 50.0,
                "robot": "astro",
                "device": "cpu",
            },
        )
    }


def test_verify_motion_viewer_path_loads_pyroki_builds_astro_and_applies_one_frame(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_pyroki_npz(motion_path)
    scene_adapter = FakeSceneAdapter()
    calls = {}

    def build_adapter(*args, **kwargs):
        calls["build_adapter"] = (args, kwargs)
        return scene_adapter

    viewer = _motion_viewer.verify_motion_viewer_path(
        motion_path,
        motion_format="pyroki",
        fps=30.0,
        robot="astro",
        device="cpu",
        scene_adapter_builder=build_adapter,
    )

    assert calls["build_adapter"] == (("astro",), {"device": "cpu"})
    assert len(viewer.motions) == 1
    assert scene_adapter.applied == [(viewer.motions[0], 0)]


def test_verify_motion_viewer_path_labels_bad_pyroki_source_data(tmp_path: Path) -> None:
    motion_path = tmp_path / "bad.npz"
    np.savez(motion_path, base_frame_pos=np.zeros((1, 3)))

    with pytest.raises(
        _motion_viewer.MotionViewerVerificationError,
        match="Failed to load pyroki motion source data",
    ):
        _motion_viewer.verify_motion_viewer_path(
            motion_path,
            scene_adapter_builder=lambda *_, **__: FakeSceneAdapter(),
        )


def test_verify_motion_viewer_path_labels_unsupported_format_selection(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        _motion_viewer.MotionViewerVerificationError,
        match="Unsupported motion format 'proto'",
    ):
        _motion_viewer.verify_motion_viewer_path(
            tmp_path / "motion.motion",
            motion_format="proto",
            load_motions=lambda *_, **__: (_ for _ in ()).throw(
                NotImplementedError("proto motion loading is not implemented yet")
            ),
            scene_adapter_builder=lambda *_, **__: FakeSceneAdapter(),
        )


def test_verify_motion_viewer_path_labels_astro_dof_mismatch(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_pyroki_npz(motion_path, dof_count=2)

    def build_adapter(*_, **__):
        return AstroMujocoSceneAdapter(
            scene=FakeScene(FakeRobot()),
            sim=FakeSimulation(),
            robot=FakeRobot(),
            joint_names=("hip", "knee", "ankle"),
            device="cpu",
        )

    with pytest.raises(
        _motion_viewer.MotionViewerVerificationError,
        match="Incompatible Astro reference motion DOF count",
    ):
        _motion_viewer.verify_motion_viewer_path(
            motion_path,
            scene_adapter_builder=build_adapter,
        )


def test_verify_motion_viewer_path_labels_scene_construction_errors(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_pyroki_npz(motion_path)

    with pytest.raises(
        _motion_viewer.MotionViewerVerificationError,
        match="Failed to construct astro scene adapter",
    ):
        _motion_viewer.verify_motion_viewer_path(
            motion_path,
            scene_adapter_builder=lambda *_, **__: (_ for _ in ()).throw(
                RuntimeError("scene setup failed")
            ),
        )


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
