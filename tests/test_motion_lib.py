from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_pyroki_npz(path: Path) -> None:
    np.savez(
        path,
        base_frame_pos=np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        base_frame_wxyz=np.array(
            [[1.0, 0.0, 0.0, 0.0]] * 3,
            dtype=np.float64,
        ),
        joint_angles=np.array(
            [[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]],
            dtype=np.float64,
        ),
    )


def _reference_motion(*, fps: float = 60.0, name: str = "walk_retargeted.npz"):
    from mjlab_playground.motion_lib import ReferenceMotion

    return ReferenceMotion(
        name=name,
        fps=fps,
        root_pos=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        dof_pos=torch.tensor([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]]),
        root_lin_vel=torch.tensor([[30.0, 0.0, 0.0]] * 3),
        root_ang_vel=torch.zeros(3, 3),
        dof_vel=torch.tensor([[30.0, 30.0]] * 3),
    )


def _package_reference_motion():
    from mjlab_playground.motion_lib import ReferenceMotion

    return ReferenceMotion(
        name="package",
        fps=10.0,
        root_pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
            ]
        ),
        root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(6, 1),
        dof_pos=torch.tensor(
            [
                [0.0, 100.0],
                [1.0, 101.0],
                [2.0, 102.0],
                [10.0, 110.0],
                [11.0, 111.0],
                [12.0, 112.0],
            ]
        ),
        clip_starts=torch.tensor([0, 3]),
        clip_lengths=torch.tensor([3, 3]),
        clip_fps=torch.tensor([10.0, 10.0]),
    )


def _z_quat(degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    return torch.tensor([math.cos(radians / 2.0), 0.0, 0.0, math.sin(radians / 2.0)])


def _reject_adapter_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    import mjlab_playground.motion_lib.motion_lib as motion_lib_module

    class RejectingMujocoSceneAdapter:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("invalid enrich input should not create adapter")

    monkeypatch.setattr(
        motion_lib_module,
        "MujocoSceneAdapter",
        RejectingMujocoSceneAdapter,
        raising=False,
    )


def test_motion_lib_query_reads_exact_frames_from_multi_clip_package() -> None:
    from mjlab_playground.motion_lib import (
        MotionLib,
        MotionLibCfg,
        ReferenceMotion,
    )

    motion_lib = MotionLib(MotionLibCfg(output_fps=10.0))
    package = _package_reference_motion()

    result = motion_lib.query(
        package,
        motion_ids=torch.tensor([0, 1, 1]),
        motion_times=torch.tensor([0.1, 0.0, 0.2]),
    )

    assert isinstance(result, ReferenceMotion)
    assert result.clip_starts is None
    assert result.clip_lengths is None
    assert result.clip_fps is None
    torch.testing.assert_close(
        result.root_pos,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
            ]
        ),
    )
    torch.testing.assert_close(
        result.dof_pos,
        torch.tensor(
            [
                [1.0, 101.0],
                [10.0, 110.0],
                [12.0, 112.0],
            ]
        ),
    )


def test_motion_lib_query_interpolates_clip_local_times() -> None:
    from mjlab_playground.motion_lib import (
        MotionLib,
        MotionLibCfg,
        ReferenceMotion,
    )

    motion_lib = MotionLib(MotionLibCfg(output_fps=10.0))
    package = ReferenceMotion(
        fps=10.0,
        root_pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 4.0, 0.0]]),
        root_rot=torch.stack([_z_quat(0.0), _z_quat(90.0), _z_quat(180.0)]),
        dof_pos=torch.tensor([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]]),
    )

    result = motion_lib.query(
        package,
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.05]),
    )

    torch.testing.assert_close(result.root_pos, torch.tensor([[0.5, 1.0, 0.0]]))
    torch.testing.assert_close(result.dof_pos, torch.tensor([[1.0, 15.0]]))
    torch.testing.assert_close(result.root_rot, _z_quat(45.0).unsqueeze(0))


def test_motion_lib_query_treats_missing_package_metadata_as_single_clip() -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_lib = MotionLib(MotionLibCfg(output_fps=10.0))
    motion = _package_reference_motion()
    motion = dataclasses.replace(
        motion,
        clip_starts=None,
        clip_lengths=None,
        clip_fps=None,
    )

    result = motion_lib.query(
        motion,
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.1]),
    )

    torch.testing.assert_close(result.root_pos, torch.tensor([[1.0, 0.0, 0.0]]))
    assert result.clip_starts is None
    assert result.clip_lengths is None
    assert result.clip_fps is None


def test_motion_lib_query_rejects_times_past_clip_boundary() -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_lib = MotionLib(MotionLibCfg(output_fps=10.0))
    package = _package_reference_motion()

    with pytest.raises(ValueError, match="motion_times.*out-of-range"):
        motion_lib.query(
            package,
            motion_ids=torch.tensor([0]),
            motion_times=torch.tensor([0.25]),
        )


def test_motion_lib_query_interpolates_optional_reference_fields() -> None:
    from mjlab_playground.motion_lib import (
        MotionLib,
        MotionLibCfg,
        ReferenceMotion,
    )

    motion_lib = MotionLib(MotionLibCfg(output_fps=10.0))
    package = ReferenceMotion(
        fps=10.0,
        root_pos=torch.zeros(3, 3),
        root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        dof_pos=torch.zeros(3, 2),
        root_lin_vel=torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0], [4.0, 8.0, 0.0]]),
        root_ang_vel=torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 4.0], [0.0, 4.0, 8.0]]),
        dof_vel=torch.tensor([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]]),
        body_pos=torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[2.0, 4.0, 6.0]],
                [[4.0, 8.0, 12.0]],
            ]
        ),
        body_rot=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(3, 1, 1),
        body_lin_vel=torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[2.0, 4.0, 6.0]],
                [[4.0, 8.0, 12.0]],
            ]
        ),
        body_ang_vel=torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[6.0, 4.0, 2.0]],
                [[12.0, 8.0, 4.0]],
            ]
        ),
    )

    result = motion_lib.query(
        package,
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.05]),
    )

    assert result.root_lin_vel is not None
    assert result.root_ang_vel is not None
    assert result.dof_vel is not None
    assert result.body_pos is not None
    assert result.body_rot is not None
    assert result.body_lin_vel is not None
    assert result.body_ang_vel is not None
    torch.testing.assert_close(result.root_lin_vel, torch.tensor([[1.0, 2.0, 0.0]]))
    torch.testing.assert_close(result.root_ang_vel, torch.tensor([[0.0, 1.0, 2.0]]))
    torch.testing.assert_close(result.dof_vel, torch.tensor([[1.0, 15.0]]))
    torch.testing.assert_close(result.body_pos, torch.tensor([[[1.0, 2.0, 3.0]]]))
    torch.testing.assert_close(result.body_rot, torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))
    torch.testing.assert_close(result.body_lin_vel, torch.tensor([[[1.0, 2.0, 3.0]]]))
    torch.testing.assert_close(result.body_ang_vel, torch.tensor([[[3.0, 2.0, 1.0]]]))


def test_motion_lib_query_rejects_out_of_range_motion_ids() -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_lib = MotionLib(MotionLibCfg(output_fps=10.0))
    package = _package_reference_motion()

    with pytest.raises(ValueError, match="motion_ids.*out-of-range"):
        motion_lib.query(
            package,
            motion_ids=torch.tensor([2]),
            motion_times=torch.tensor([0.0]),
        )


@pytest.mark.parametrize(
    ("motion_ids", "motion_times", "match"),
    [
        (torch.tensor([[0]]), torch.tensor([0.0]), "motion_ids must be 1D"),
        (torch.tensor([0]), torch.tensor([[0.0]]), "motion_times must be 1D"),
        (
            torch.tensor([0, 1]),
            torch.tensor([0.0]),
            "motion_ids and motion_times must have matching shapes",
        ),
        (torch.tensor([0.0]), torch.tensor([0.0]), "motion_ids must be an integer"),
        (torch.tensor([False]), torch.tensor([0.0]), "motion_ids must be an integer"),
        (torch.tensor([0]), torch.tensor([0]), "motion_times must be a floating point"),
    ],
)
def test_motion_lib_query_rejects_malformed_id_and_time_tensors(
    motion_ids: torch.Tensor,
    motion_times: torch.Tensor,
    match: str,
) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_lib = MotionLib(MotionLibCfg(output_fps=10.0))
    package = _package_reference_motion()

    with pytest.raises((TypeError, ValueError), match=match):
        motion_lib.query(
            package,
            motion_ids=motion_ids,
            motion_times=motion_times,
        )


def test_motion_lib_query_returns_contact_fields() -> None:
    from mjlab_playground.motion_lib import (
        MotionLib,
        MotionLibCfg,
        ReferenceMotion,
    )

    motion_lib = MotionLib(MotionLibCfg(output_fps=10.0))
    package = ReferenceMotion(
        fps=10.0,
        root_pos=torch.zeros(3, 3),
        root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        dof_pos=torch.zeros(3, 2),
        body_contacts=torch.tensor(
            [[False, True], [True, False], [True, True]],
            dtype=torch.bool,
        ),
        foot_contacts=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=torch.float32,
        ),
    )

    result = motion_lib.query(
        package,
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.1]),
    )

    assert result.body_contacts is not None
    assert result.foot_contacts is not None
    torch.testing.assert_close(result.body_contacts, torch.tensor([[True, False]]))
    torch.testing.assert_close(result.foot_contacts, torch.tensor([[1.0, 0.0]]))


def test_motion_lib_public_exports_construct_astro_pyroki_pipeline() -> None:
    import mjlab_playground.motion_lib as motion_lib_package
    from mjlab_playground.motion_lib import (
        MotionLib,
        MotionLibCfg,
    )

    cfg = MotionLibCfg(
        source_format="pyroki",
        source_fps=30.0,
        output_fps=60.0,
        robot="astro",
        device="cpu",
    )
    motion_lib = MotionLib(cfg)

    assert motion_lib.cfg == cfg
    assert dataclasses.is_dataclass(cfg)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.output_fps = 120.0
    assert MotionLib.__name__ in motion_lib_package.__all__
    assert MotionLibCfg.__name__ in motion_lib_package.__all__
    assert set(motion_lib_package.__all__) == {
        "ClipWeighting",
        "MimicMotionManager",
        "MotionLib",
        "MotionLibCfg",
        "MotionManager",
        "MotionManagerCfg",
        "ReferenceFrame",
        "ReferenceMotion",
        "ReferenceMotionSample",
        "ReferenceMotionState",
        "TimeSampling",
    }
    assert "MotionLoader" not in motion_lib_package.__all__
    assert "MotionResamplingCfg" not in motion_lib_package.__all__
    assert "ReferenceMotionNpzWriter" not in motion_lib_package.__all__
    assert "ReferenceMotionQueryResult" not in motion_lib_package.__all__
    with pytest.raises(ImportError):
        from mjlab_playground.motion_lib import MotionLoader  # noqa: F401
    with pytest.raises(ImportError):
        from mjlab_playground.motion_lib import MotionResamplingCfg  # noqa: F401
    with pytest.raises(ImportError):
        from mjlab_playground.motion_lib import ReferenceMotionNpzWriter  # noqa: F401
    with pytest.raises(ImportError):
        from mjlab_playground.motion_lib import ReferenceMotionQueryResult  # noqa: F401


def test_motion_lib_cfg_validates_v1_pipeline_options() -> None:
    from mjlab_playground.motion_lib import MotionLibCfg

    cfg_fields = {field.name for field in dataclasses.fields(MotionLibCfg)}
    assert "motion_files" not in cfg_fields
    assert "motion_file" not in cfg_fields

    with pytest.raises(ValueError, match="Only robot='astro' is supported"):
        MotionLibCfg(robot="g1")  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError, match="proto source format is reserved"):
        MotionLibCfg(source_format="proto")

    with pytest.raises(ValueError, match="source_fps must be integer-valued"):
        MotionLibCfg(source_fps=29.97)

    with pytest.raises(ValueError, match="output_fps must be positive and finite"):
        MotionLibCfg(output_fps=0.0)


def test_motion_lib_load_does_not_accept_contact_label_paths() -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_lib = MotionLib(MotionLibCfg())

    with pytest.raises(TypeError, match="contact_labels"):
        motion_lib.load("motion.npz", contact_labels="contacts")  # type: ignore[call-arg]


@pytest.mark.parametrize("method_name", ["resample", "enrich"])
def test_motion_lib_scalar_transformations_reject_collection_inputs(
    method_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    _reject_adapter_creation(monkeypatch)
    motion_lib = MotionLib(MotionLibCfg(output_fps=60.0))

    with pytest.raises(TypeError, match=r"one ReferenceMotion$"):
        getattr(motion_lib, method_name)([_reference_motion()])


def test_motion_lib_loads_pyroki_source_clips_without_contact_labels(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_path = tmp_path / "walk_retargeted.npz"
    _write_pyroki_npz(motion_path)
    motion_lib = MotionLib(
        MotionLibCfg(source_format="pyroki", source_fps=50.0, device="cpu")
    )

    motions = motion_lib.load(motion_path)

    assert len(motions) == 1
    motion = motions[0]
    assert motion.name == "walk_retargeted.npz"
    assert not hasattr(motion, "display_name")
    assert motion.fps == 50.0
    assert motion.root_pos.device == torch.device("cpu")
    assert motion.foot_contacts is None


def test_motion_lib_resamples_loaded_source_clip_to_output_fps(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_path = tmp_path / "walk_retargeted.npz"
    _write_pyroki_npz(motion_path)
    motion_lib = MotionLib(
        MotionLibCfg(source_format="pyroki", source_fps=30.0, output_fps=60.0)
    )
    source_motion = motion_lib.load(motion_path)[0]

    motion = motion_lib.resample(source_motion)

    assert motion is not source_motion
    assert motion.name == "walk_retargeted.npz"
    assert not hasattr(motion, "display_name")
    assert motion.fps == 60.0
    torch.testing.assert_close(
        motion.root_pos[:, 0],
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
    )
    torch.testing.assert_close(
        motion.dof_pos,
        torch.tensor([[0.0, 1.0], [0.25, 1.25], [0.5, 1.5], [0.75, 1.75], [1.0, 2.0]]),
    )
    torch.testing.assert_close(
        motion.root_lin_vel, torch.tensor([[15.0, 0.0, 0.0]] * 5)
    )
    torch.testing.assert_close(motion.root_ang_vel, torch.zeros(5, 3))
    torch.testing.assert_close(motion.dof_vel, torch.tensor([[15.0, 15.0]] * 5))


def test_motion_lib_resample_rejects_contact_bearing_clip_with_name(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_path = tmp_path / "contact_walk_retargeted.npz"
    _write_pyroki_npz(motion_path)
    motion_lib = MotionLib(
        MotionLibCfg(source_format="pyroki", source_fps=30.0, output_fps=60.0)
    )
    source_motion = motion_lib.load(motion_path)[0]
    contact_motion = dataclasses.replace(
        source_motion,
        foot_contacts=torch.zeros(source_motion.root_pos.shape[0], 2),
    )

    with pytest.raises(
        ValueError, match="contact_walk_retargeted\\.npz.*foot_contacts"
    ):
        motion_lib.resample(contact_motion)


def test_motion_lib_resample_rejects_rich_clip_with_name(tmp_path: Path) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_path = tmp_path / "rich_walk_retargeted.npz"
    _write_pyroki_npz(motion_path)
    motion_lib = MotionLib(
        MotionLibCfg(source_format="pyroki", source_fps=30.0, output_fps=60.0)
    )
    source_motion = motion_lib.load(motion_path)[0]
    rich_motion = dataclasses.replace(
        source_motion,
        body_pos=torch.zeros(source_motion.root_pos.shape[0], 1, 3),
    )

    with pytest.raises(ValueError, match="rich_walk_retargeted\\.npz.*body_pos"):
        motion_lib.resample(rich_motion)


def test_motion_lib_full_pipeline_enriches_resampled_clip_without_writing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mjlab_playground.motion_lib.motion_lib as motion_lib_module
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_path = tmp_path / "walk_retargeted.npz"
    _write_pyroki_npz(motion_path)

    class FakeMujocoSceneAdapter:
        init_calls = []

        def __init__(self, cfg) -> None:
            from mjlab_playground.motion_lib import ReferenceFrame

            self.init_calls.append(
                {
                    "robot_cfg": cfg.robot_cfg,
                    "output_fps": cfg.output_fps,
                    "device": cfg.device,
                }
            )
            self._reference_frame_cls = ReferenceFrame
            self.current_frame = None
            self.joint_names = ("left_hip", "right_hip")
            self.body_names = ("pelvis", "torso")

        def apply_frame(self, frame):
            assert frame.root_lin_vel is not None
            assert frame.root_ang_vel is not None
            assert frame.dof_vel is not None
            self.current_frame = frame

        def read_robot_state(self):
            frame = self.current_frame
            assert frame is not None
            return self._reference_frame_cls(
                root_pos=frame.root_pos,
                root_rot=frame.root_rot,
                dof_pos=frame.dof_pos,
                root_lin_vel=frame.root_lin_vel,
                root_ang_vel=frame.root_ang_vel,
                dof_vel=frame.dof_vel,
                body_pos=torch.ones(2, 3),
                body_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
                body_lin_vel=torch.full((2, 3), 2.0),
                body_ang_vel=torch.full((2, 3), 3.0),
            )

    monkeypatch.setattr(
        motion_lib_module,
        "MujocoSceneAdapter",
        FakeMujocoSceneAdapter,
        raising=False,
    )
    monkeypatch.setattr(
        motion_lib_module,
        "_load_astro_constants_module",
        lambda: SimpleNamespace(get_astro_robot_cfg=lambda: "astro-robot-cfg"),
        raising=False,
    )
    motion_lib = MotionLib(
        MotionLibCfg(source_format="pyroki", source_fps=30.0, output_fps=60.0)
    )

    rich = motion_lib.enrich(motion_lib.resample(motion_lib.load(motion_path)[0]))

    assert FakeMujocoSceneAdapter.init_calls == [
        {
            "robot_cfg": "astro-robot-cfg",
            "output_fps": 60.0,
            "device": torch.device("cpu"),
        }
    ]
    assert rich.name == "walk_retargeted.npz"
    assert not hasattr(rich, "display_name")
    assert rich.fps == 60.0
    torch.testing.assert_close(
        rich.root_pos[:, 0],
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
    )
    torch.testing.assert_close(rich.dof_vel, torch.tensor([[15.0, 15.0]] * 5))
    assert rich.body_pos is not None
    assert rich.body_pos.shape == (5, 2, 3)
    assert rich.body_rot is not None
    assert rich.body_rot.shape == (5, 2, 4)
    assert rich.body_lin_vel is not None
    assert rich.body_lin_vel.shape == (5, 2, 3)
    assert rich.body_ang_vel is not None
    assert rich.body_ang_vel.shape == (5, 2, 3)
    assert rich.dof_names == ("left_hip", "right_hip")
    assert rich.body_names == ("pelvis", "torso")
    assert {path.name for path in tmp_path.iterdir()} == {"walk_retargeted.npz"}


def test_motion_lib_enrich_rejects_source_rate_clip_with_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    _reject_adapter_creation(monkeypatch)
    motion_lib = MotionLib(
        MotionLibCfg(source_format="pyroki", source_fps=30.0, output_fps=60.0)
    )

    with pytest.raises(
        ValueError,
        match="walk_retargeted\\.npz.*cfg\\.output_fps=60\\.0.*motion\\.fps=30\\.0",
    ):
        motion_lib.enrich(_reference_motion(fps=30.0))


def test_motion_lib_enrich_uses_unnamed_diagnostic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    _reject_adapter_creation(monkeypatch)
    motion_lib = MotionLib(MotionLibCfg(output_fps=60.0))

    with pytest.raises(ValueError, match="<unnamed motion>"):
        motion_lib.enrich(dataclasses.replace(_reference_motion(fps=30.0), name=None))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("body_pos", torch.zeros(3, 1, 3)),
        ("body_rot", torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(3, 1, 1)),
        ("body_lin_vel", torch.zeros(3, 1, 3)),
        ("body_ang_vel", torch.zeros(3, 1, 3)),
    ],
)
def test_motion_lib_enrich_rejects_existing_rich_body_fields(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    value: torch.Tensor,
) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    _reject_adapter_creation(monkeypatch)
    motion_lib = MotionLib(MotionLibCfg(output_fps=60.0))
    motion = dataclasses.replace(_reference_motion(), **{field_name: value})

    with pytest.raises(ValueError, match=f"walk_retargeted\\.npz.*{field_name}"):
        motion_lib.enrich(motion)


def test_motion_lib_enrich_rejects_source_foot_contacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    _reject_adapter_creation(monkeypatch)
    motion_lib = MotionLib(MotionLibCfg(output_fps=60.0))
    motion = dataclasses.replace(
        _reference_motion(),
        foot_contacts=torch.zeros(3, 2),
    )

    with pytest.raises(ValueError, match="walk_retargeted\\.npz.*foot_contacts"):
        motion_lib.enrich(motion)


def test_motion_lib_reuses_adapter_across_scalar_enrich_calls_and_names_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mjlab_playground.motion_lib.motion_lib as motion_lib_module
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    class FailingMujocoSceneAdapter:
        init_count = 0

        def __init__(self, cfg) -> None:
            type(self).init_count += 1
            self.read_count = 0
            self.current_frame = None
            self.joint_names = ("hip", "knee")
            self.body_names = ("pelvis",)

        def apply_frame(self, frame):
            self.current_frame = frame

        def read_robot_state(self):
            from mjlab_playground.motion_lib import ReferenceFrame

            self.read_count += 1
            if self.read_count > 3:
                raise RuntimeError("simulator rejected pose")
            frame = self.current_frame
            assert frame is not None
            return ReferenceFrame(
                root_pos=frame.root_pos,
                root_rot=frame.root_rot,
                dof_pos=frame.dof_pos,
                root_lin_vel=frame.root_lin_vel,
                root_ang_vel=frame.root_ang_vel,
                dof_vel=frame.dof_vel,
                body_pos=torch.ones(1, 3),
                body_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                body_lin_vel=torch.zeros(1, 3),
                body_ang_vel=torch.zeros(1, 3),
            )

    monkeypatch.setattr(
        motion_lib_module,
        "MujocoSceneAdapter",
        FailingMujocoSceneAdapter,
        raising=False,
    )
    monkeypatch.setattr(
        motion_lib_module,
        "_load_astro_constants_module",
        lambda: SimpleNamespace(get_astro_robot_cfg=lambda: "astro-robot-cfg"),
        raising=False,
    )
    motion_lib = MotionLib(MotionLibCfg(output_fps=60.0))

    good = motion_lib.enrich(_reference_motion(name="good_walk.npz"))

    assert good.dof_names == ("hip", "knee")
    assert good.body_names == ("pelvis",)
    with pytest.raises(RuntimeError, match="bad_walk\\.npz.*simulator rejected pose"):
        motion_lib.enrich(_reference_motion(name="bad_walk.npz"))
    assert FailingMujocoSceneAdapter.init_count == 1
