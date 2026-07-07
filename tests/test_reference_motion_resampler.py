from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mjlab_playground.motion_lib import ReferenceMotionState  # noqa: E402
from mjlab_playground.motion_lib.motion_resampler import (  # noqa: E402
    MotionResamplingCfg,
    ReferenceMotionResampler,
)


def _z_quat(degrees: float, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    radians = math.radians(degrees)
    return torch.tensor(
        [math.cos(radians / 2.0), 0.0, 0.0, math.sin(radians / 2.0)],
        dtype=dtype,
    )


def _source_motion(*, fps: float = 30.0, dtype: torch.dtype = torch.float32) -> ReferenceMotionState:
    return ReferenceMotionState(
        name="motion.npz",
        display_name="motion",
        fps=fps,
        root_pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=dtype,
        ),
        root_rot=torch.stack(
            [_z_quat(0.0, dtype=dtype), _z_quat(90.0, dtype=dtype), _z_quat(180.0, dtype=dtype)]
        ),
        dof_pos=torch.tensor(
            [[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]],
            dtype=dtype,
        ),
    )


@pytest.mark.parametrize("output_fps", [0.0, -30.0, 29.97, float("inf"), float("nan")])
def test_motion_resampling_cfg_requires_positive_finite_integer_fps(
    output_fps: float,
) -> None:
    with pytest.raises(ValueError):
        MotionResamplingCfg(output_fps=output_fps)


def test_reference_motion_resampler_upsamples_coordinates_and_velocities() -> None:
    motion = _source_motion(dtype=torch.float64)
    result = ReferenceMotionResampler(MotionResamplingCfg(output_fps=60.0)).resample(motion)

    assert result is not motion
    assert result.name == "motion.npz"
    assert result.display_name == "motion"
    assert result.fps == 60.0
    assert result.root_pos.dtype == torch.float64
    assert result.root_pos.device == motion.root_pos.device
    assert result.root_pos.shape == (5, 3)

    torch.testing.assert_close(
        result.root_pos[:, 0],
        torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.dof_pos,
        torch.tensor(
            [[0.0, 10.0], [1.0, 15.0], [2.0, 20.0], [3.0, 25.0], [4.0, 30.0]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        result.root_rot,
        torch.stack(
            [
                _z_quat(0.0, dtype=torch.float64),
                _z_quat(45.0, dtype=torch.float64),
                _z_quat(90.0, dtype=torch.float64),
                _z_quat(135.0, dtype=torch.float64),
                _z_quat(180.0, dtype=torch.float64),
            ]
        ),
    )
    torch.testing.assert_close(
        result.root_lin_vel,
        torch.tensor([[30.0, 0.0, 0.0]] * 5, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.dof_vel,
        torch.tensor([[60.0, 300.0]] * 5, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.root_ang_vel,
        torch.tensor([[0.0, 0.0, 15.0 * math.pi]] * 5, dtype=torch.float64),
    )


def test_reference_motion_resampler_rejects_downsampling() -> None:
    with pytest.raises(ValueError, match="does not support downsampling"):
        ReferenceMotionResampler(MotionResamplingCfg(output_fps=30.0)).resample(
            _source_motion(fps=60.0)
        )


def test_reference_motion_resampler_slerp_uses_shortest_quaternion_path() -> None:
    motion = ReferenceMotionState(
        fps=30.0,
        root_pos=torch.zeros(3, 3),
        root_rot=torch.stack([_z_quat(0.0), -_z_quat(90.0), _z_quat(180.0)]),
        dof_pos=torch.zeros(3, 2),
    )

    result = ReferenceMotionResampler(MotionResamplingCfg(output_fps=60.0)).resample(motion)

    torch.testing.assert_close(result.root_rot[1], _z_quat(45.0))


def test_reference_motion_resampler_rejects_rich_reference_fields() -> None:
    motion = ReferenceMotionState(
        fps=30.0,
        root_pos=torch.zeros(3, 3),
        root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        dof_pos=torch.zeros(3, 2),
        body_pos=torch.zeros(3, 1, 3),
    )

    with pytest.raises(ValueError, match="body_pos"):
        ReferenceMotionResampler(MotionResamplingCfg(output_fps=60.0)).resample(motion)


def test_reference_motion_resampler_rejects_unresampled_foot_contacts() -> None:
    motion = ReferenceMotionState(
        fps=30.0,
        root_pos=torch.zeros(3, 3),
        root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        dof_pos=torch.zeros(3, 2),
        foot_contacts=torch.zeros(3, 2),
    )

    with pytest.raises(ValueError, match="foot_contacts"):
        ReferenceMotionResampler(MotionResamplingCfg(output_fps=60.0)).resample(motion)


def test_reference_motion_resampler_ignores_existing_velocity_fields() -> None:
    motion = ReferenceMotionState(
        fps=30.0,
        root_pos=torch.zeros(3, 3),
        root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        dof_pos=torch.zeros(3, 2),
        root_lin_vel=torch.ones(3, 3),
        root_ang_vel=torch.ones(3, 3),
        dof_vel=torch.ones(3, 2),
    )

    result = ReferenceMotionResampler(MotionResamplingCfg(output_fps=30.0)).resample(motion)

    assert result is not motion
    torch.testing.assert_close(result.root_lin_vel, torch.zeros(3, 3))
    torch.testing.assert_close(result.root_ang_vel, torch.zeros(3, 3))
    torch.testing.assert_close(result.dof_vel, torch.zeros(3, 2))
