"""Astro getup environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_playground.asset_zoo.robots.astro.astro_constants import get_astro_robot_cfg
from mjlab_playground.tasks.getup import mdp
from mjlab_playground.tasks.getup.component_factory import (
    build_clean_actor_obs,
    reduced_proprio_current,
)
from mjlab_playground.tasks.getup.getup_env_cfg import make_getup_env_cfg
from mjlab_playground.tasks.getup.mdp.actions import (
    SettleRelativeJointPositionActionCfg,
)

# Derived from default pose (knees bent) keyframe.
_TORSO_HEIGHT = 0.7515
_PELVIS_HEIGHT = 0.7145


def astro_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Astro getup task configuration."""
    cfg = make_getup_env_cfg()

    cfg.scene.entities = {"robot": get_astro_robot_cfg()}

    # self-collision sensor.
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)

    cfg.rewards["self_collision"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Torso + waist height. Waist reward prevents "sitting on booty or knees" local
    # minimum where torso is high but waist (pelvis) stays near ground.
    cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
    cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=("torso_link",)
    )
    cfg.rewards["waist_height"] = RewardTermCfg(
        func=mdp.height_reward,
        weight=1.0,
        params={
            "desired_height": _PELVIS_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
        },
    )

    cfg.metrics["getup_success"].params["desired_height"] = _PELVIS_HEIGHT
    cfg.metrics["getup_success"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=("torso_link",)
    )

    # Per-joint posture std: tight hips, medium knees and ankles, loose arms and waist.
    cfg.rewards["posture"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(".*",), body_names=("torso_link",)
    )
    cfg.rewards["posture"].params["std"] = {
        r".*_hip_roll_joint": 0.08,
        r".*_hip_yaw_joint": 0.08,
        r".*_hip_pitch_joint": 0.12,
        r".*_knee_joint": 0.15,
        r".*_ankle_pitch_joint": 0.2,
        r".*_ankle_roll_joint": 0.2,
        r"(waist_.*|.*_shoulder.*|.*_elbow.*|.*_wrist.*)": 0.5,
    }

    cfg.rewards["orientation"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=("torso_link",)
    )

    cfg.observations["actor"] = reduced_proprio_current(enable_corruption=True)
    cfg.observations["critic"] = reduced_proprio_current(
        enable_corruption=False,
        with_state_estimation=True,
    )
    # Override projected_gravity to use torso_link instead of pelvis (root).
    _torso_cfg = SceneEntityCfg("robot", body_names=("torso_link",))
    cfg.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
        func=mdp.body_projected_gravity,
        params={"asset_cfg": _torso_cfg},
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    cfg.observations["critic"].terms["projected_gravity"] = ObservationTermCfg(
        func=mdp.body_projected_gravity,
        params={"asset_cfg": _torso_cfg},
    )
    # Clean counterparts for L2C2
    build_clean_actor_obs(cfg)

    cfg.viewer.body_name = "torso_link"

    cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=("torso_link",)
    )

    cfg.events["geom_friction_slide"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
            "operation": "abs",
            "axes": [0],
            "ranges": (0.3, 1.5),
            "shared_random": True,
        },
    )

    cfg.events["reset_fallen_or_standing"].params["fall_height"] = 0.8
    cfg.events["reset_fallen_or_standing"].params["joint_range_scale"] = 0.5

    assert isinstance(cfg.actions["joint_pos"], SettleRelativeJointPositionActionCfg)
    cfg.actions["joint_pos"].settle_steps = 30
    cfg.terminations["energy"].params["settle_steps"] = 30

    cfg.curriculum = {}
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["joint_vel_l2"].weight = -0.0
    cfg.rewards["joint_vel_hinge"] = RewardTermCfg(
        func=mdp.joint_vel_hinge,
        weight=-0.1,
        params={"threshold": 2.0},  # rad/s
    )
    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events["reset_fallen_or_standing"].params["fall_probability"] = 1.0

    return cfg
