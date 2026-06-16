from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_playground.getup import mdp


def reduced_proprio_current(
    enable_corruption: bool = False,
    with_state_estimation: bool = False,
) -> ObservationGroupCfg:

    reduced_proprio_terms = {
        "base_pos": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_pos"},
        ),
        "base_ori": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_quat"},
        ),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
          func=mdp.builtin_sensor,
          params={"sensor_name": "robot/imu_ang_vel"},
          noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            params={"biased": True},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    if not with_state_estimation:
        reduced_proprio_terms.pop("base_pos", None)
        reduced_proprio_terms.pop("base_lin_vel", None)

    reduced_proprio_group = ObservationGroupCfg(
        terms=reduced_proprio_terms,
        concatenate_terms=True,
        enable_corruption=enable_corruption,
    )

    return reduced_proprio_group


def max_proprio_current(
    enable_corruption: bool = False,
    with_state_estimation: bool = True,
) -> ObservationGroupCfg:

    max_proprio_terms = {
        "base_pos": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_pos"},
        ),
        "base_ori": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_quat"},
        ),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"biased": True},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    max_proprio_group = ObservationGroupCfg(
        terms=max_proprio_terms,
        concatenate_terms=True,
        enable_corruption=enable_corruption,
    )

    return max_proprio_group


def clean_actor_obs_groups(
    observations: dict[str, ObservationGroupCfg],
) -> dict[str, ObservationGroupCfg]:
    clean_groups: dict[str, ObservationGroupCfg] = {}
    for source_name, source_group in observations.items():
        if source_name != "actor" and not source_name.startswith("actor_"):
            continue
        if source_name.endswith("_clean"):
            continue
        clean_group = deepcopy(source_group)
        clean_group.enable_corruption = False
        clean_groups[f"{source_name}_clean"] = clean_group
    return clean_groups


def build_clean_actor_obs(cfg: ManagerBasedRlEnvCfg) -> None:
    """Create clean (corruption-free) counterparts for all actor observation groups.

    Mirrors ``actor`` -> ``actor_clean`` and, when present,
    ``actor_history`` -> ``actor_history_clean`` and
    ``actor_future`` -> ``actor_future_clean``.

    Any existing ``_clean`` group whose source is no longer present is removed
    so stale observations do not leak into the compiled model.
    """

    clean_groups = clean_actor_obs_groups(cfg.observations)
    for obs_name in tuple(cfg.observations):
        if obs_name.startswith("actor_") and obs_name.endswith("_clean"):
            cfg.observations.pop(obs_name, None)
    cfg.observations.update(clean_groups)
