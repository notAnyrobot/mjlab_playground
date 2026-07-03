from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mjlab_playground.motion_lib.math_util import (  # noqa: E402
    finite_difference_velocity,
    lerp_frame_values,
    quaternion_angular_velocity_wxyz,
    slerp_frame_quaternions_wxyz,
)


def z_quat(degrees: float, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    radians = math.radians(degrees)
    return torch.tensor(
        [math.cos(radians / 2.0), 0.0, 0.0, math.sin(radians / 2.0)],
        dtype=dtype,
    )


def test_lerp_frame_values_samples_fractional_frame_positions() -> None:
    values = torch.tensor([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]], dtype=torch.float64)
    frame_positions = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64)

    result = lerp_frame_values(values, frame_positions)

    torch.testing.assert_close(
        result,
        torch.tensor(
            [[0.0, 10.0], [1.0, 15.0], [2.0, 20.0], [3.0, 25.0], [4.0, 30.0]],
            dtype=torch.float64,
        ),
    )


def test_slerp_frame_quaternions_wxyz_uses_shortest_path() -> None:
    quaternions = torch.stack([z_quat(0.0), -z_quat(90.0), z_quat(180.0)])
    frame_positions = torch.tensor([0.5], dtype=torch.float64)

    result = slerp_frame_quaternions_wxyz(quaternions, frame_positions)

    torch.testing.assert_close(result[0], z_quat(45.0))


def test_finite_difference_velocity_matches_torch_gradient_edge_convention() -> None:
    values = torch.tensor([[0.0], [1.0], [4.0], [9.0]], dtype=torch.float64)

    result = finite_difference_velocity(values, fps=2.0)

    torch.testing.assert_close(
        result,
        torch.tensor([[2.0], [4.0], [8.0], [10.0]], dtype=torch.float64),
    )


def test_quaternion_angular_velocity_wxyz_uses_so3_differences() -> None:
    quaternions = torch.stack([z_quat(0.0), z_quat(90.0), z_quat(180.0)])

    result = quaternion_angular_velocity_wxyz(quaternions, fps=30.0)

    torch.testing.assert_close(
        result,
        torch.tensor([[0.0, 0.0, 15.0 * math.pi]] * 3, dtype=torch.float64),
    )
