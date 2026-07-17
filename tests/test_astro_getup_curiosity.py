"""Tests for the Astro getup curiosity observation."""

from __future__ import annotations

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_playground.tasks.getup.config.astro.env_cfgs import astro_getup_env_cfg
from mjlab_playground.tasks.getup.mdp.observations import (
    body_height,
    body_projected_gravity,
)

_NUM_ENVS = 5


@pytest.fixture(scope="module")
def astro_env() -> ManagerBasedRlEnv:
    cfg = astro_getup_env_cfg()
    cfg.scene.num_envs = _NUM_ENVS
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    return env


@pytest.mark.parametrize("body_name", ["torso_link", "pelvis"])
def test_astro_body_height_is_one_finite_value_per_environment(
    astro_env: ManagerBasedRlEnv,
    body_name: str,
) -> None:
    asset_cfg = SceneEntityCfg("robot", body_names=(body_name,))
    asset_cfg.resolve(astro_env.scene)

    height = body_height(astro_env, asset_cfg)

    assert height.shape == (_NUM_ENVS, 1)
    assert torch.isfinite(height).all()


def test_astro_curiosity_state_has_ordered_normalized_runtime_values(
    astro_env: ManagerBasedRlEnv,
) -> None:
    torso_cfg = SceneEntityCfg("robot", body_names=("torso_link",))
    torso_cfg.resolve(astro_env.scene)
    pelvis_cfg = SceneEntityCfg("robot", body_names=("pelvis",))
    pelvis_cfg.resolve(astro_env.scene)

    observations = astro_env.get_observations()
    rnd_state = observations["rnd_state"]
    expected = torch.cat(
        (
            body_projected_gravity(astro_env, torso_cfg),
            (body_height(astro_env, torso_cfg) / 0.7515).clamp(0.0, 1.25),
            (body_height(astro_env, pelvis_cfg) / 0.7145).clamp(0.0, 1.25),
        ),
        dim=-1,
    )

    assert rnd_state.shape == (_NUM_ENVS, 5)
    assert torch.isfinite(rnd_state).all()
    torch.testing.assert_close(rnd_state, expected)


def test_astro_curiosity_heights_clip_after_normalization(
    astro_env: ManagerBasedRlEnv,
) -> None:
    robot = astro_env.scene["robot"]
    root_pose = robot.data.root_link_pose_w.clone()
    root_pose[0, 2] = 2.0
    root_pose[1, 2] = -2.0
    robot.write_root_link_pose_to_sim(root_pose)
    astro_env.sim.forward()

    try:
        observations = astro_env.observation_manager.compute(update_history=True)
        rnd_state = observations["rnd_state"]
        assert isinstance(rnd_state, torch.Tensor)
        heights = rnd_state[:, 3:]

        torch.testing.assert_close(heights[0], torch.tensor([1.25, 1.25]))
        torch.testing.assert_close(heights[1], torch.tensor([0.0, 0.0]))
    finally:
        astro_env.reset()
