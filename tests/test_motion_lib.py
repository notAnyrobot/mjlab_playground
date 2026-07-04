from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

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
    from mjlab_playground.motion_lib import ReferenceMotionState

    return ReferenceMotionState(
        name=name,
        display_name=name.removesuffix(".npz"),
        fps=fps,
        root_pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        dof_pos=torch.tensor([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]]),
        root_lin_vel=torch.tensor([[30.0, 0.0, 0.0]] * 3),
        root_ang_vel=torch.zeros(3, 3),
        dof_vel=torch.tensor([[30.0, 30.0]] * 3),
    )


def _reject_adapter_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    import mjlab_playground.motion_lib.motion_lib as motion_lib_module

    class RejectingAstroAdapter:
        @classmethod
        def create(cls, *, device: torch.device):
            raise AssertionError("invalid enrich input should not create adapter")

    monkeypatch.setattr(
        motion_lib_module,
        "AstroSimulatorEnrichmentAdapter",
        RejectingAstroAdapter,
    )


def test_motion_lib_public_exports_construct_astro_pyroki_pipeline() -> None:
    import mjlab_playground.motion_lib as motion_lib_package
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

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


def test_motion_lib_enrich_accepts_empty_batch_without_creating_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    _reject_adapter_creation(monkeypatch)
    motion_lib = MotionLib(MotionLibCfg())

    assert motion_lib.enrich([]) == []


def test_motion_lib_load_does_not_accept_contact_label_paths() -> None:
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_lib = MotionLib(MotionLibCfg())

    with pytest.raises(TypeError, match="contact_labels"):
        motion_lib.load("motion.npz", contact_labels="contacts")  # type: ignore[call-arg]


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
    assert motion.display_name == "walk_retargeted"
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

    resampled = motion_lib.resample([source_motion])

    assert len(resampled) == 1
    motion = resampled[0]
    assert motion is not source_motion
    assert motion.name == "walk_retargeted.npz"
    assert motion.display_name == "walk_retargeted"
    assert motion.fps == 60.0
    torch.testing.assert_close(
        motion.root_pos[:, 0],
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
    )
    torch.testing.assert_close(
        motion.dof_pos,
        torch.tensor(
            [[0.0, 1.0], [0.25, 1.25], [0.5, 1.5], [0.75, 1.75], [1.0, 2.0]]
        ),
    )
    torch.testing.assert_close(motion.root_lin_vel, torch.tensor([[15.0, 0.0, 0.0]] * 5))
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

    with pytest.raises(ValueError, match="contact_walk_retargeted\\.npz.*foot_contacts"):
        motion_lib.resample([contact_motion])


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
        motion_lib.resample([rich_motion])


def test_motion_lib_full_pipeline_enriches_resampled_clip_without_writing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mjlab_playground.motion_lib.motion_lib as motion_lib_module
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    motion_path = tmp_path / "walk_retargeted.npz"
    _write_pyroki_npz(motion_path)

    class FakeAstroAdapter:
        create_calls = 0

        @classmethod
        def create(cls, *, device: torch.device) -> "FakeAstroAdapter":
            cls.create_calls += 1
            assert device == torch.device("cpu")
            return cls()

        def enrich(self, motion):
            assert motion.fps == 60.0
            assert motion.root_lin_vel is not None
            assert motion.root_ang_vel is not None
            assert motion.dof_vel is not None
            frames = motion.root_pos.shape[0]
            return dataclasses.replace(
                motion,
                body_pos=torch.ones(frames, 2, 3),
                body_rot=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(frames, 2, 1),
                body_lin_vel=torch.full((frames, 2, 3), 2.0),
                body_ang_vel=torch.full((frames, 2, 3), 3.0),
            )

    monkeypatch.setattr(
        motion_lib_module,
        "AstroSimulatorEnrichmentAdapter",
        FakeAstroAdapter,
        raising=False,
    )
    motion_lib = MotionLib(
        MotionLibCfg(source_format="pyroki", source_fps=30.0, output_fps=60.0)
    )

    rich_motions = motion_lib.enrich(motion_lib.resample(motion_lib.load(motion_path)))

    assert FakeAstroAdapter.create_calls == 1
    assert len(rich_motions) == 1
    rich = rich_motions[0]
    assert rich.name == "walk_retargeted.npz"
    assert rich.display_name == "walk_retargeted"
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
        motion_lib.enrich([_reference_motion(fps=30.0)])


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
        motion_lib.enrich([motion])


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
        motion_lib.enrich([motion])


def test_motion_lib_enrich_batch_failure_names_offending_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mjlab_playground.motion_lib.motion_lib as motion_lib_module
    from mjlab_playground.motion_lib import MotionLib, MotionLibCfg

    class FailingAstroAdapter:
        @classmethod
        def create(cls, *, device: torch.device) -> "FailingAstroAdapter":
            return cls()

        def enrich(self, motion):
            if motion.name == "bad_walk.npz":
                raise RuntimeError("simulator rejected pose")
            frames = motion.root_pos.shape[0]
            return dataclasses.replace(
                motion,
                body_pos=torch.ones(frames, 1, 3),
                body_rot=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(frames, 1, 1),
                body_lin_vel=torch.zeros(frames, 1, 3),
                body_ang_vel=torch.zeros(frames, 1, 3),
            )

    monkeypatch.setattr(
        motion_lib_module,
        "AstroSimulatorEnrichmentAdapter",
        FailingAstroAdapter,
    )
    motion_lib = MotionLib(MotionLibCfg(output_fps=60.0))

    with pytest.raises(RuntimeError, match="bad_walk\\.npz.*simulator rejected pose"):
        motion_lib.enrich(
            [
                _reference_motion(name="good_walk.npz"),
                _reference_motion(name="bad_walk.npz"),
            ]
        )
