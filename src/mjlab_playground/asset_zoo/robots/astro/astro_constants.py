"""Dobot Astro humanoid constants."""

import dataclasses
import math
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

from mjlab_playground import MJLAB_PLAYGROUND_SRC_PATH

# Shared foot collision constants (must be defined before imports that use them)
_MAX_FOOT_COLLISION_GEOMS = 12  # 36 for sphere foot, 12 for capsule foot
_FOOT_COLLISION_GEOMS_EXPR = (
    rf"^(left|right)_foot[1-{_MAX_FOOT_COLLISION_GEOMS}]_collision$"
)
FOOT_COLLISION_GEOMS = tuple(
    f"{side}_foot{i}_collision"
    for side in ("left", "right")
    for i in range(1, _MAX_FOOT_COLLISION_GEOMS + 1)
)


##
# MJCF and assets.
##

ASTRO_XML: Path = (
    MJLAB_PLAYGROUND_SRC_PATH
    / "asset_zoo"
    / "robots"
    / "astro"
    / "xmls"
    / "astro_v1_29dof.xml"
    # / "astro_v1.xml"
)
assert ASTRO_XML.exists(), f"Astro XML file does not exist: {ASTRO_XML}"


def get_spec() -> mujoco.MjSpec: # type: ignore
    return mujoco.MjSpec.from_file(str(ASTRO_XML)) # type: ignore


##
# Motion / CSV joint order (29 DOF). Order must match motion CSV columns. Head yaw is excluded purposefully.
##

ASTRO_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    # "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Canonical Astro AMP v1 discriminator body-state contract.  These body names
# are ordered and stable because they define the flattened discriminator feature
# layout used by policy and expert AMP observations.
ASTRO_AMP_ROOT_BODY_NAME = "pelvis"
ASTRO_AMP_BODY_NAMES: tuple[str, ...] = (
    ASTRO_AMP_ROOT_BODY_NAME,
    "torso_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
)
ASTRO_AMP_HISTORY_OFFSETS: tuple[int, ...] = (0, -1, -2, -4, -8, -16, -32)

# Safe pose constants used by motion conversion scripts.
# fmt: off
ASTRO_MOTION_SAFE_POSE_JOINTS = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
    0.0, 0.0, 0.0,                           # waist
    # 0.0,                                     # head_yaw
    0.2, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0,      # left arm
    0.2, -0.2, 0.0, 0.2, 0.0, 0.0, 0.0,     # right arm
])
# fmt: on
ASTRO_MOTION_SAFE_Z_HEIGHT = 0.745


##
# Keyframe config.
##

ZERO_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, ASTRO_MOTION_SAFE_Z_HEIGHT),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, ASTRO_MOTION_SAFE_Z_HEIGHT),
    joint_pos={
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        ".*_shoulder_pitch_joint": 0.2,
        ".*_elbow_joint": 1.5,
        "left_shoulder_roll_joint": 0.3,
        "right_shoulder_roll_joint": -0.3,
    },
    joint_vel={".*": 0.0},
)

T_POSE_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, ASTRO_MOTION_SAFE_Z_HEIGHT),
    joint_pos={
        "left_shoulder_roll_joint": 1.5,
        "right_shoulder_roll_joint": -1.5,
        ".*_elbow_joint": 1.5,
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
    },
    joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, ASTRO_MOTION_SAFE_Z_HEIGHT),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    },
    joint_vel={".*": 0.0},
)


##
# Actuator config.
##

# Motor specs (from Dobot).
ARMATURE_8514_25 = 81.431e-3
ARMATURE_5016_25 = 8.811e-3
ARMATURE_3907_36 = 2.387e-3  # kg*m^2

ACTUATOR_8514_25 = ElectricActuator(
    reflected_inertia=ARMATURE_8514_25,
    velocity_limit=18.85,
    effort_limit=130.0,
)
ACTUATOR_5016_25 = ElectricActuator(
    reflected_inertia=ARMATURE_5016_25,
    velocity_limit=26.18,
    effort_limit=30.0,
)
ACTUATOR_3907_36 = ElectricActuator(
    reflected_inertia=ARMATURE_3907_36,
    velocity_limit=20.94,
    effort_limit=10.0,
)

DEFAULT_NATURAL_FREQUENCY_HZ = 10.0
DAMPING_RATIO = 2.5 # Heavily overdamped to prevent oscillations, which can cause sim instability and don't look good in a humanoid.


def _compute_stiffness(
    actuator: ElectricActuator,
    armature_multiplier: float = 1.0,
    natural_frequency_hz: float = DEFAULT_NATURAL_FREQUENCY_HZ,
) -> float:
    """Stiffness k = I ω² with ω = 2π f; natural_frequency_hz is f in Hz."""
    omega = math.tau * natural_frequency_hz
    armature = actuator.reflected_inertia * armature_multiplier
    return armature * omega**2


def _compute_damping(
    actuator: ElectricActuator,
    armature_multiplier: float = 1.0,
    natural_frequency_hz: float = DEFAULT_NATURAL_FREQUENCY_HZ,
    damping_ratio: float = DAMPING_RATIO,
) -> float:
    """Damping c = 2 ζ I ω with ω = 2π f; natural_frequency_hz is f in Hz."""
    omega = math.tau * natural_frequency_hz
    armature = actuator.reflected_inertia * armature_multiplier
    return 2.0 * damping_ratio * armature * omega



ASTRO_ACTUATOR_8514_25 = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "waist_yaw_joint",
        ".*_hip_pitch_joint",
        ".*_hip_roll_joint",
        ".*_hip_yaw_joint",
        ".*_knee_joint",
    ),
    stiffness=_compute_stiffness(ACTUATOR_8514_25, natural_frequency_hz=9.0),
    damping=_compute_damping(ACTUATOR_8514_25, natural_frequency_hz=9.0),
    effort_limit=ACTUATOR_8514_25.effort_limit,
    armature=ACTUATOR_8514_25.reflected_inertia,
)
ASTRO_ACTUATOR_5016_25 = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
    ),
    stiffness=_compute_stiffness(ACTUATOR_5016_25),
    damping=_compute_damping(ACTUATOR_5016_25),
    effort_limit=ACTUATOR_5016_25.effort_limit,
    armature=ACTUATOR_5016_25.reflected_inertia,
)
ASTRO_ACTUATOR_3907_36 = BuiltinPositionActuatorCfg(
    target_names_expr=(
        # "head_yaw_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_yaw_joint",
    ),
    stiffness=_compute_stiffness(ACTUATOR_3907_36, natural_frequency_hz=12.0),
    damping=_compute_damping(ACTUATOR_3907_36, natural_frequency_hz=12.0),
    effort_limit=ACTUATOR_3907_36.effort_limit,
    armature=ACTUATOR_3907_36.reflected_inertia,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 5016_25 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.

ASTRO_ACTUATOR_WAIST_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_pitch_joint",),
    stiffness=_compute_stiffness(ACTUATOR_5016_25) * 2,
    damping=_compute_damping(ACTUATOR_5016_25, damping_ratio=3.0) * 2,
    effort_limit=ACTUATOR_5016_25.effort_limit * 2,
    armature=ACTUATOR_5016_25.reflected_inertia * 2,
)
ASTRO_ACTUATOR_WAIST_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_roll_joint",),
    stiffness=_compute_stiffness(ACTUATOR_5016_25) * 1.66,
    damping=_compute_damping(ACTUATOR_5016_25, damping_ratio=3.0) * 1.66,
    effort_limit=ACTUATOR_5016_25.effort_limit * 1.66,
    armature=ACTUATOR_5016_25.reflected_inertia * 1.66,
)
ASTRO_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint",),
    stiffness=_compute_stiffness(ACTUATOR_5016_25) * 2,
    damping=_compute_damping(ACTUATOR_5016_25) * 2,
    effort_limit=ACTUATOR_5016_25.effort_limit * 2,
    armature=ACTUATOR_5016_25.reflected_inertia * 2,
)
ASTRO_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_roll_joint",),
    stiffness=_compute_stiffness(ACTUATOR_5016_25) * 1.5,
    damping=_compute_damping(ACTUATOR_5016_25) * 1.5,
    effort_limit=ACTUATOR_5016_25.effort_limit * 1.5,
    armature=ACTUATOR_5016_25.reflected_inertia * 1.5,
)


##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_FOOT_COLLISION_GEOMS_EXPR: 3, ".*_collision": 1},
    priority={_FOOT_COLLISION_GEOMS_EXPR: 1},
    friction={_FOOT_COLLISION_GEOMS_EXPR: (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    condim={_FOOT_COLLISION_GEOMS_EXPR: 3, ".*_collision": 1},
    priority={_FOOT_COLLISION_GEOMS_EXPR: 1},
    friction={_FOOT_COLLISION_GEOMS_EXPR: (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_FOOT_COLLISION_GEOMS_EXPR,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)


##
# Final config.
##

ASTRO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        ASTRO_ACTUATOR_8514_25,
        ASTRO_ACTUATOR_5016_25,
        ASTRO_ACTUATOR_3907_36,
        ASTRO_ACTUATOR_WAIST_PITCH,
        ASTRO_ACTUATOR_WAIST_ROLL,
        ASTRO_ACTUATOR_ANKLE_PITCH,
        ASTRO_ACTUATOR_ANKLE_ROLL,
    ),
    soft_joint_pos_limit_factor=0.85,
)


def get_astro_robot_cfg(
    init_pos: str = "knees_bent", delayed_actuators: bool = False
) -> EntityCfg:
    """Get a fresh Astro robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """

    if init_pos == "zero":
        init_state = ZERO_KEYFRAME
    elif init_pos == "home":
        init_state = HOME_KEYFRAME
    elif init_pos == "knees_bent":
        init_state = KNEES_BENT_KEYFRAME
    elif init_pos == "t_pose":
        init_state = T_POSE_KEYFRAME
    else:
        raise ValueError(f"Invalid robot init_pos: {init_pos}")

    if delayed_actuators:
        _delay_kwargs = dict(
            delay_min_lag=2,
            delay_max_lag=5,
            delay_hold_prob=0.3,
            delay_update_period=10,
        )
        articulation = EntityArticulationInfoCfg(
            actuators=tuple(
                a if a.delay_max_lag > 0 else dataclasses.replace(a, **_delay_kwargs)
                for a in ASTRO_ARTICULATION.actuators
            ),
            soft_joint_pos_limit_factor=ASTRO_ARTICULATION.soft_joint_pos_limit_factor,
        )
    else:
        articulation = ASTRO_ARTICULATION

    return EntityCfg(
        init_state=init_state,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=articulation,
    )


ASTRO_ACTION_SCALE: dict[str, float] = {}
for a in ASTRO_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    assert a.effort_limit is not None
    for n in a.target_names_expr:
        ASTRO_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_astro_robot_cfg(init_pos="t_pose", delayed_actuators=False))

    viewer.launch(robot.spec.compile())
