"""Tests for the opt-in Astro getup RND experiment."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import cast

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_playground.rl_extensions import (
    MjPgOnPolicyRunnerCfg,
    MjPgPpo,
    MjPgPpoAlgorithmCfg,
)
from mjlab_playground.tasks.getup.config.astro.rl_cfg import (
    astro_getup_ppo_runner_cfg,
)
from mjlab_playground.tasks.getup.rl import GetupOnPolicyRunner

_BASELINE_TASK = "Mjlab-Getup-Flat-Astro"
_RND_TASK = "Mjlab-Getup-Flat-Astro-RND"
_NUM_ENVS = 4


def _make_bounded_runner(
    task_id: str = _RND_TASK,
) -> tuple[GetupOnPolicyRunner, RslRlVecEnvWrapper]:
    torch.manual_seed(0)
    env_cfg = load_env_cfg(task_id)
    env_cfg.seed = 0
    env_cfg.scene.num_envs = _NUM_ENVS
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device="cpu"))

    runner_cfg = cast(MjPgOnPolicyRunnerCfg, load_rl_cfg(task_id))
    algorithm_cfg = cast(MjPgPpoAlgorithmCfg, runner_cfg.algorithm)
    runner_cfg.num_steps_per_env = 2
    runner_cfg.actor.hidden_dims = (32,)
    runner_cfg.critic.hidden_dims = (32,)
    algorithm_cfg.num_learning_epochs = 1
    algorithm_cfg.num_mini_batches = 1
    return GetupOnPolicyRunner(env, asdict(runner_cfg), device="cpu"), env


def test_astro_runner_factory_disables_extensions_by_default() -> None:
    cfg = astro_getup_ppo_runner_cfg()
    algorithm_cfg = cast(MjPgPpoAlgorithmCfg, cfg.algorithm)

    assert algorithm_cfg.l2c2_cfg is None
    assert algorithm_cfg.rnd_cfg is None
    with pytest.raises(TypeError):
        astro_getup_ppo_runner_cfg(True, True)  # type: ignore[call-arg]


def test_astro_runner_factory_enables_complete_rnd_and_l2c2_config() -> None:
    cfg = astro_getup_ppo_runner_cfg(l2c2=True, rnd=True)
    algorithm_cfg = cast(MjPgPpoAlgorithmCfg, cfg.algorithm)

    assert algorithm_cfg.l2c2_cfg is not None
    assert algorithm_cfg.l2c2_cfg.enable is True
    assert algorithm_cfg.l2c2_cfg.lambda_l2c2 == 0.1
    assert algorithm_cfg.l2c2_cfg.clean_obs_suffix == "_clean"

    rnd_cfg = algorithm_cfg.rnd_cfg
    assert rnd_cfg is not None
    assert rnd_cfg.num_outputs == 1
    assert rnd_cfg.predictor_hidden_dims == (-1, -1)
    assert rnd_cfg.target_hidden_dims == (-1,)
    assert rnd_cfg.activation == "elu"
    assert rnd_cfg.state_normalization is True
    assert rnd_cfg.reward_normalization is False
    assert rnd_cfg.weight == 1.0
    assert rnd_cfg.weight_schedule is None
    assert rnd_cfg.learning_rate == 0.001


def test_astro_registrations_select_only_rnd_as_the_experimental_variable() -> None:
    baseline_rl = cast(MjPgOnPolicyRunnerCfg, load_rl_cfg("Mjlab-Getup-Flat-Astro"))
    rnd_rl = cast(MjPgOnPolicyRunnerCfg, load_rl_cfg("Mjlab-Getup-Flat-Astro-RND"))
    baseline_algorithm = cast(MjPgPpoAlgorithmCfg, baseline_rl.algorithm)
    rnd_algorithm = cast(MjPgPpoAlgorithmCfg, rnd_rl.algorithm)

    assert baseline_algorithm.l2c2_cfg is not None
    assert baseline_algorithm.rnd_cfg is None
    assert rnd_algorithm.l2c2_cfg is not None
    assert rnd_algorithm.rnd_cfg is not None
    assert load_runner_cls("Mjlab-Getup-Flat-Astro") is GetupOnPolicyRunner
    assert load_runner_cls("Mjlab-Getup-Flat-Astro-RND") is GetupOnPolicyRunner

    for play in (False, True):
        baseline_env = load_env_cfg("Mjlab-Getup-Flat-Astro", play=play)
        rnd_env = load_env_cfg("Mjlab-Getup-Flat-Astro-RND", play=play)

        assert type(rnd_env) is type(baseline_env)
        assert rnd_env.decimation == baseline_env.decimation == 4
        assert rnd_env.sim.mujoco.timestep == baseline_env.sim.mujoco.timestep == 0.005
        assert tuple(rnd_env.observations) == tuple(baseline_env.observations)
        assert rnd_env.scene.num_envs == baseline_env.scene.num_envs


def test_registered_astro_rnd_task_constructs_live_extensions() -> None:
    runner, env = _make_bounded_runner()
    alg = cast(MjPgPpo, runner.alg)

    try:
        assert alg.l2c2 is not None
        assert alg.rnd is not None
        assert alg.rnd.num_states == 5
        assert alg.rnd.initial_weight == pytest.approx(0.02)
    finally:
        env.close()


def test_registered_astro_baseline_task_constructs_l2c2_without_rnd() -> None:
    runner, env = _make_bounded_runner(_BASELINE_TASK)
    alg = cast(MjPgPpo, runner.alg)

    try:
        assert alg.rnd is None
        assert alg.l2c2 is not None
    finally:
        env.close()


def test_astro_rnd_rollout_and_update_are_finite_across_auto_reset() -> None:
    runner, env = _make_bounded_runner()
    alg = cast(MjPgPpo, runner.alg)
    assert alg.rnd is not None
    assert alg.l2c2 is not None

    try:
        obs = env.get_observations()
        env.episode_length_buf[0] = env.max_episode_length - 1
        initial_rnd_updates = alg.rnd.update_counter
        intrinsic_by_step: list[torch.Tensor] = []
        dones_by_step: list[torch.Tensor] = []

        with torch.inference_mode():
            for _ in range(runner.cfg["num_steps_per_env"]):
                actions = alg.act(obs)
                obs, rewards, dones, extras = env.step(actions)
                task_rewards = rewards.clone()
                values = alg.transition.values
                assert values is not None
                values = values.clone()
                storage_step = alg.storage.step

                alg.process_env_step(obs, rewards, dones, extras)

                intrinsic = alg.intrinsic_rewards.clone()
                expected_stored_rewards = task_rewards + intrinsic
                if "time_outs" in extras:
                    expected_stored_rewards += alg.gamma * torch.squeeze(
                        values * extras["time_outs"].unsqueeze(1), 1
                    )
                torch.testing.assert_close(
                    alg.storage.rewards[storage_step].squeeze(-1),
                    expected_stored_rewards,
                )
                torch.testing.assert_close(rewards, task_rewards)
                intrinsic_by_step.append(intrinsic)
                dones_by_step.append(dones.clone())

            alg.compute_returns(obs)

        losses = alg.update()
        intrinsic_rewards = torch.stack(intrinsic_by_step)
        dones = torch.stack(dones_by_step).bool()
        terminal_intrinsic = intrinsic_rewards[dones]
        continuing_intrinsic = intrinsic_rewards[~dones]

        assert alg.rnd.update_counter - initial_rnd_updates == len(intrinsic_by_step)
        assert torch.isfinite(intrinsic_rewards).all()
        assert dones.any(), "bounded smoke path must exercise terminal auto-reset"
        assert terminal_intrinsic.numel() > 0
        assert continuing_intrinsic.numel() > 0
        assert math.isfinite(losses["rnd"])
        assert math.isfinite(losses["l2c2"])
        print(
            "auto-reset observation: "
            f"terminal_count={dones.sum().item()}, "
            f"terminal_intrinsic_max={terminal_intrinsic.max().item():.8f}, "
            f"continuing_intrinsic_max={continuing_intrinsic.max().item():.8f}, "
            f"rnd_loss={losses['rnd']:.8f}, "
            f"l2c2_loss={losses['l2c2']:.8f}"
        )
    finally:
        env.close()
