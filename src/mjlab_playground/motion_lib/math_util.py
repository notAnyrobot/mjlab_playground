from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "FrameInterpolation",
    "axis_angle_from_quat_wxyz",
    "finite_difference_velocity",
    "frame_interpolation",
    "lerp_frame_values",
    "quat_conjugate_wxyz",
    "quat_multiply_wxyz",
    "quat_unique_wxyz",
    "quaternion_angular_velocity_wxyz",
    "relative_rotation_vector_wxyz",
    "slerp_frame_quaternions_wxyz",
]


@dataclass(frozen=True, kw_only=True)
class FrameInterpolation:
    lower: torch.Tensor
    upper: torch.Tensor
    blend: torch.Tensor


def frame_interpolation(
    frame_positions: torch.Tensor,
    num_frames: int,
) -> FrameInterpolation:
    lower = torch.floor(frame_positions).to(dtype=torch.long)
    upper = torch.clamp(lower + 1, max=num_frames - 1)
    blend = frame_positions - lower.to(dtype=frame_positions.dtype)
    return FrameInterpolation(lower=lower, upper=upper, blend=blend)


def lerp_frame_values(values: torch.Tensor, frame_positions: torch.Tensor) -> torch.Tensor:
    interpolation = frame_interpolation(frame_positions, values.shape[0])
    view_shape = (-1,) + (1,) * (values.ndim - 1)
    blend = interpolation.blend.reshape(view_shape)
    return values[interpolation.lower] * (1.0 - blend) + values[interpolation.upper] * blend


def slerp_frame_quaternions_wxyz(
    quaternions: torch.Tensor,
    frame_positions: torch.Tensor,
) -> torch.Tensor:
    interpolation = frame_interpolation(frame_positions, quaternions.shape[0])
    q0 = quaternions[interpolation.lower]
    q1 = quaternions[interpolation.upper]
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

    t = interpolation.blend.unsqueeze(-1)
    lerp_result = q0 * (1.0 - t) + q1 * t

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    slerp_result = (
        torch.sin((1.0 - t) * theta) / sin_theta * q0
        + torch.sin(t * theta) / sin_theta * q1
    )
    near_parallel = dot.abs() > 0.9995
    result = torch.where(near_parallel, lerp_result, slerp_result)
    return torch.nn.functional.normalize(result, dim=-1)


def finite_difference_velocity(values: torch.Tensor, fps: float) -> torch.Tensor:
    velocity = torch.empty_like(values)
    velocity[0] = (values[1] - values[0]) * fps
    velocity[1:-1] = (values[2:] - values[:-2]) * (fps / 2.0)
    velocity[-1] = (values[-1] - values[-2]) * fps
    return velocity


def quaternion_angular_velocity_wxyz(
    quaternions: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    angular_velocity = torch.empty(quaternions.shape[0], 3, dtype=quaternions.dtype)
    angular_velocity = angular_velocity.to(device=quaternions.device)
    angular_velocity[0] = relative_rotation_vector_wxyz(quaternions[1], quaternions[0]) * fps
    angular_velocity[1:-1] = relative_rotation_vector_wxyz(
        quaternions[2:],
        quaternions[:-2],
    ) * (fps / 2.0)
    angular_velocity[-1] = relative_rotation_vector_wxyz(quaternions[-1], quaternions[-2]) * fps
    return angular_velocity


def relative_rotation_vector_wxyz(q_next: torch.Tensor, q_prev: torch.Tensor) -> torch.Tensor:
    q_rel = quat_multiply_wxyz(q_next, quat_conjugate_wxyz(q_prev))
    q_rel = torch.nn.functional.normalize(q_rel, dim=-1)
    return axis_angle_from_quat_wxyz(quat_unique_wxyz(q_rel))


def quat_unique_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)


def quat_conjugate_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def axis_angle_from_quat_wxyz(
    quaternion: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    quaternion = quat_unique_wxyz(quaternion)
    mag = torch.linalg.norm(quaternion[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quaternion[..., 0])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps,
        torch.sin(half_angle) / angle,
        0.5 - angle * angle / 48,
    )
    return quaternion[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)
