from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MOTION_LOADER_PATH = (
    ROOT / "src" / "mjlab_playground" / "motion_lib" / "motion_loader.py"
)


def _load_motion_loader_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("motion_loader", MOTION_LOADER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_motion_loader = _load_motion_loader_module()
ReferenceMotionState = _motion_loader.ReferenceMotionState


def test_reference_motion_state_defaults_to_30_fps_and_preserves_required_tensors() -> None:
    root_pos = torch.zeros(2, 3)
    root_rot = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    dof_pos = torch.zeros(2, 29)

    state = ReferenceMotionState(root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos)

    assert state.fps == 30.0
    assert state.root_pos is root_pos
    assert state.root_rot is root_rot
    assert state.dof_pos is dof_pos
    assert state.dof_vel is None
    assert state.body_pos is None
    assert state.body_contacts is None


def test_reference_motion_state_accepts_full_reference_motion_fields() -> None:
    state = ReferenceMotionState(
        fps=50.0,
        root_pos=torch.zeros(4, 3),
        root_rot=torch.zeros(4, 4),
        dof_pos=torch.zeros(4, 29),
        root_lin_vel=torch.zeros(4, 3),
        root_ang_vel=torch.zeros(4, 3),
        dof_vel=torch.zeros(4, 29),
        body_pos=torch.zeros(4, 31, 3),
        body_rot=torch.zeros(4, 31, 4),
        body_lin_vel=torch.zeros(4, 31, 3),
        body_ang_vel=torch.zeros(4, 31, 3),
        body_contacts=torch.zeros(4, 31, dtype=torch.bool),
    )

    assert state.fps == 50.0
    assert state.body_pos is not None
    assert state.body_pos.shape == (4, 31, 3)
    assert state.body_contacts is not None
    assert state.body_contacts.dtype == torch.bool


@pytest.mark.parametrize("fps", [0.0, -30.0, float("inf"), float("nan")])
def test_reference_motion_state_rejects_non_positive_or_non_finite_fps(
    fps: float,
) -> None:
    with pytest.raises(ValueError, match="fps must be positive and finite"):
        ReferenceMotionState(
            fps=fps,
            root_pos=torch.zeros(2, 3),
            root_rot=torch.zeros(2, 4),
            dof_pos=torch.zeros(2, 29),
        )


def test_motion_state_alias_is_not_exported_from_module_file() -> None:
    assert not hasattr(_motion_loader, "MotionState")
