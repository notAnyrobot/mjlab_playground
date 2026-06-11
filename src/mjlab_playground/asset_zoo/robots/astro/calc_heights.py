"""Compute Astro pelvis/torso heights from a keyframe-style robot configuration.

This script applies a selected initial-state keyframe, runs forward kinematics,
aligns both feet to the ground plane (z=0) by shifting floating-base z, then
reports pelvis and torso heights from ground.

Usage examples:

    uv run python -m mjlab_playground.asset_zoo.robots.astro.calc_heights
    uv run python -m mjlab_playground.asset_zoo.robots.astro.calc_heights --config home
    uv run python -m mjlab_playground.asset_zoo.robots.astro.calc_heights --config knees_bent
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.entity import EntityCfg

from mjlab_playground.asset_zoo.robots.astro.astro_constants import (
    _MAX_FOOT_COLLISION_GEOMS,
    HOME_KEYFRAME,
    KNEES_BENT_KEYFRAME,
    T_POSE_KEYFRAME,
    ZERO_KEYFRAME,
    get_spec,
)

BODY_PELVIS = "pelvis"
BODY_TORSO = "torso_link"
FLOATING_BASE_JOINT = "floating_base_joint"


@dataclass
class HeightResult:
    config_name: str
    pelvis_height: float
    torso_height: float
    base_z_after_alignment: float
    feet_min_z_before_alignment: float
    feet_min_z_after_alignment: float


def _joint_qpos_dof(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return 1
    raise ValueError(f"Unsupported joint type: {joint_type}")


def _set_named_joint_qpos(model: mujoco.MjModel, qpos: np.ndarray, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(f"Joint not found: {name}")

    dof = _joint_qpos_dof(int(model.jnt_type[joint_id]))
    if dof != 1:
        raise ValueError(f"Joint {name} has qpos dof={dof}; expected 1")

    qpos[int(model.jnt_qposadr[joint_id])] = value


def _set_floating_base_pos(model: mujoco.MjModel, qpos: np.ndarray, pos_xyz: tuple[float, float, float]) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FLOATING_BASE_JOINT)
    if joint_id < 0:
        raise KeyError(f"Floating-base joint not found: {FLOATING_BASE_JOINT}")

    qpos_adr = int(model.jnt_qposadr[joint_id])
    qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos_xyz, dtype=np.float64)


def _single_dof_joint_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        if _joint_qpos_dof(int(model.jnt_type[joint_id])) != 1:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            continue
        names.append(joint_name)
    return names


def _apply_keyframe_to_qpos(model: mujoco.MjModel, keyframe: EntityCfg.InitialStateCfg) -> np.ndarray:
    qpos = np.array(model.qpos0, copy=True)

    _set_floating_base_pos(model, qpos, tuple(float(x) for x in keyframe.pos))

    joint_names = _single_dof_joint_names(model)
    for pattern, value in keyframe.joint_pos.items():
        regex = re.compile(pattern)
        matched = [name for name in joint_names if regex.fullmatch(name)]
        for joint_name in matched:
            _set_named_joint_qpos(model, qpos, joint_name, float(value))

    return qpos


def _foot_collision_geom_ids(model: mujoco.MjModel) -> list[int]:
    geom_ids: list[int] = []
    for side in ("left", "right"):
        for i in range(1, _MAX_FOOT_COLLISION_GEOMS + 1):
            geom_name = f"{side}_foot{i}_collision"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id < 0:
                raise KeyError(f"Foot collision geom not found: {geom_name}")
            geom_ids.append(int(geom_id))
    return geom_ids


def _body_height(data: mujoco.MjData, model: mujoco.MjModel, body_name: str) -> float:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Body not found: {body_name}")
    return float(data.xpos[int(body_id), 2])


def _compute_heights(config_name: str, keyframe: EntityCfg.InitialStateCfg) -> HeightResult:
    model = get_spec().compile()
    data = mujoco.MjData(model)

    qpos = _apply_keyframe_to_qpos(model, keyframe)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    foot_geom_ids = _foot_collision_geom_ids(model)
    feet_min_z_before = min(float(data.geom_xpos[geom_id, 2]) for geom_id in foot_geom_ids)

    # Align ground plane to the lowest foot collision geom by shifting floating-base z.
    qpos_aligned = np.array(qpos, copy=True)
    floating_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FLOATING_BASE_JOINT)
    qpos_adr = int(model.jnt_qposadr[int(floating_joint_id)])
    qpos_aligned[qpos_adr + 2] -= feet_min_z_before

    data.qpos[:] = qpos_aligned
    mujoco.mj_forward(model, data)

    feet_min_z_after = min(float(data.geom_xpos[geom_id, 2]) for geom_id in foot_geom_ids)
    pelvis_height = _body_height(data, model, BODY_PELVIS)
    torso_height = _body_height(data, model, BODY_TORSO)

    return HeightResult(
        config_name=config_name,
        pelvis_height=pelvis_height,
        torso_height=torso_height,
        base_z_after_alignment=float(qpos_aligned[qpos_adr + 2]),
        feet_min_z_before_alignment=feet_min_z_before,
        feet_min_z_after_alignment=feet_min_z_after,
    )


def _keyframe_from_name(name: str) -> EntityCfg.InitialStateCfg:
    by_name: dict[str, EntityCfg.InitialStateCfg] = {
        "home": HOME_KEYFRAME,
        "zero": ZERO_KEYFRAME,
        "knees_bent": KNEES_BENT_KEYFRAME,
        "t_pose": T_POSE_KEYFRAME,
    }
    try:
        return by_name[name]
    except KeyError as exc:
        valid = ", ".join(sorted(by_name))
        raise ValueError(f"Unknown config {name!r}. Valid values: {valid}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="home",
        choices=("home", "zero", "knees_bent", "t_pose"),
        help="Initial-state config to evaluate.",
    )
    args = parser.parse_args()

    keyframe = _keyframe_from_name(args.config)
    result = _compute_heights(args.config, keyframe)

    print(f"config: {result.config_name}")
    print(f"pelvis_height_from_ground: {result.pelvis_height:.6f} m")
    print(f"torso_height_from_ground:  {result.torso_height:.6f} m")
    print(f"floating_base_z_after_alignment: {result.base_z_after_alignment:.6f} m")
    print(f"feet_min_z_before_alignment: {result.feet_min_z_before_alignment:.6f} m")
    print(f"feet_min_z_after_alignment:  {result.feet_min_z_after_alignment:.6f} m")


if __name__ == "__main__":
    main()
