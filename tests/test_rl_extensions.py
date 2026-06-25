"""Tests for RL extension wiring."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import mjlab_playground.rl_extensions.algorithms.ppo as ppo_mod
import pytest
import torch
from mjlab_playground.rl_extensions import MjPgOnPolicyRunnerCfg, MjPgPpo
from mjlab_playground.rl_extensions.l2c2 import L2C2, L2C2Cfg, resolve_l2c2_config
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

_ROOT = Path(__file__).resolve().parents[1]
_PPO_SOURCE = _ROOT / "src/mjlab_playground/rl_extensions/algorithms/ppo.py"
_L2C2_INIT_SOURCE = _ROOT / "src/mjlab_playground/rl_extensions/l2c2/__init__.py"


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _minimal_mjpgppo() -> MjPgPpo:
    alg = object.__new__(MjPgPpo)
    alg.clip_param = 0.2
    alg.use_clipped_value_loss = True
    alg.value_loss_coef = 2.0
    alg.entropy_coef = 0.05
    alg.rnd = None
    alg.symmetry = None
    alg.l2c2 = None
    return alg


def _loss_batch() -> RolloutStorage.Batch:
    return RolloutStorage.Batch(
        observations=TensorDict({"actor": torch.zeros(2, 1)}, batch_size=[2]),
        values=torch.tensor([[0.2], [0.8]]),
        advantages=torch.tensor([[1.0], [-0.5]]),
        returns=torch.tensor([[0.5], [1.0]]),
        old_actions_log_prob=torch.tensor([[0.0], [0.1]]),
    )


def test_l2c2_config_is_consumed_as_resolved_dict() -> None:
    tree = ast.parse(_PPO_SOURCE.read_text())

    normalize_refs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "normalize_l2c2_config"
    ]
    assert normalize_refs == []

    l2c2_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _name(node.func) == "L2C2"
    ]
    assert any(
        any(
            keyword.arg is None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "l2c2_cfg"
            for keyword in call.keywords
        )
        for call in l2c2_calls
    )


def test_l2c2_package_does_not_reexport_normalization_helper() -> None:
    assert "normalize_l2c2_config" not in _L2C2_INIT_SOURCE.read_text()


def test_compute_losses_returns_plain_ppo_losses() -> None:
    alg = _minimal_mjpgppo()
    batch = _loss_batch()
    actions_log_prob = torch.tensor([0.1, -0.2])
    values = torch.tensor([[0.7], [0.3]])
    entropy = torch.tensor([0.3, 0.7])
    context = ppo_mod.MjPgPpoLossContext(
        batch=batch,
        original_batch_size=2,
        actions_log_prob=actions_log_prob,
        values=values,
        entropy=entropy,
    )

    losses = alg.compute_losses(context)

    ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
    surrogate = -torch.squeeze(batch.advantages) * ratio
    surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio,
        1.0 - alg.clip_param,
        1.0 + alg.clip_param,
    )
    expected_surrogate = torch.max(surrogate, surrogate_clipped).mean()
    value_clipped = batch.values + (values - batch.values).clamp(
        -alg.clip_param,
        alg.clip_param,
    )
    value_losses = (values - batch.returns).pow(2)
    value_losses_clipped = (value_clipped - batch.returns).pow(2)
    expected_value = torch.max(value_losses, value_losses_clipped).mean()
    expected_entropy = entropy.mean()
    expected_ppo = (
        expected_surrogate
        + alg.value_loss_coef * expected_value
        - alg.entropy_coef * expected_entropy
    )

    torch.testing.assert_close(losses.surrogate_loss, expected_surrogate)
    torch.testing.assert_close(losses.value_loss, expected_value)
    torch.testing.assert_close(losses.entropy, expected_entropy)
    torch.testing.assert_close(losses.ppo_loss, expected_ppo)
    assert losses.rnd_loss is None
    assert losses.symmetry_loss is None
    assert losses.l2c2_loss is None


def test_compute_losses_keeps_rnd_loss_separate_from_ppo_loss() -> None:
    class DummyRnd:
        def compute_loss(self, observations: TensorDict) -> torch.Tensor:
            assert observations.batch_size == torch.Size([2])
            return torch.tensor(3.0)

    alg = _minimal_mjpgppo()
    alg.rnd = DummyRnd()
    batch = _loss_batch()
    context = ppo_mod.MjPgPpoLossContext(
        batch=batch,
        original_batch_size=2,
        actions_log_prob=torch.tensor([0.1, -0.2]),
        values=torch.tensor([[0.7], [0.3]]),
        entropy=torch.tensor([0.3, 0.7]),
    )

    without_rnd = _minimal_mjpgppo().compute_losses(context)
    losses = alg.compute_losses(context)

    torch.testing.assert_close(losses.ppo_loss, without_rnd.ppo_loss)
    torch.testing.assert_close(losses.rnd_loss, torch.tensor(3.0))


def test_compute_losses_adds_symmetry_only_when_mirror_loss_enabled() -> None:
    class DummySymmetry:
        def __init__(self, *, use_mirror_loss: bool) -> None:
            self.use_mirror_loss = use_mirror_loss
            self.mirror_loss_coeff = 0.25

        def compute_loss(
            self,
            actor: object,
            batch: RolloutStorage.Batch,
            original_batch_size: int,
        ) -> torch.Tensor:
            assert actor is sentinel_actor
            assert batch is context.batch
            assert original_batch_size == 2
            return torch.tensor(4.0)

    sentinel_actor = object()
    context = ppo_mod.MjPgPpoLossContext(
        batch=_loss_batch(),
        original_batch_size=2,
        actions_log_prob=torch.tensor([0.1, -0.2]),
        values=torch.tensor([[0.7], [0.3]]),
        entropy=torch.tensor([0.3, 0.7]),
    )
    base_loss = _minimal_mjpgppo().compute_losses(context).ppo_loss

    logging_alg = _minimal_mjpgppo()
    logging_alg.actor = sentinel_actor
    logging_alg.symmetry = DummySymmetry(use_mirror_loss=False)
    logging_losses = logging_alg.compute_losses(context)

    learning_alg = _minimal_mjpgppo()
    learning_alg.actor = sentinel_actor
    learning_alg.symmetry = DummySymmetry(use_mirror_loss=True)
    learning_losses = learning_alg.compute_losses(context)

    torch.testing.assert_close(logging_losses.symmetry_loss, torch.tensor(4.0))
    torch.testing.assert_close(logging_losses.ppo_loss, base_loss)
    torch.testing.assert_close(learning_losses.symmetry_loss, torch.tensor(4.0))
    torch.testing.assert_close(learning_losses.ppo_loss, base_loss + torch.tensor(1.0))


def test_compute_losses_adds_weighted_l2c2_loss_when_enabled() -> None:
    class DummyL2C2:
        def compute_loss(
            self,
            actor: object,
            batch: RolloutStorage.Batch,
            original_batch_size: int,
        ) -> object:
            assert actor is sentinel_actor
            assert batch is context.batch
            assert original_batch_size == 2
            return type(
                "DummyL2C2Loss",
                (),
                {
                    "weighted": torch.tensor(0.75),
                    "l2c2_loss": torch.tensor(7.5),
                },
            )()

    sentinel_actor = object()
    context = ppo_mod.MjPgPpoLossContext(
        batch=_loss_batch(),
        original_batch_size=2,
        actions_log_prob=torch.tensor([0.1, -0.2]),
        values=torch.tensor([[0.7], [0.3]]),
        entropy=torch.tensor([0.3, 0.7]),
    )
    disabled_losses = _minimal_mjpgppo().compute_losses(context)
    enabled_alg = _minimal_mjpgppo()
    enabled_alg.actor = sentinel_actor
    enabled_alg.l2c2 = DummyL2C2()

    enabled_losses = enabled_alg.compute_losses(context)

    assert disabled_losses.l2c2_loss is None
    torch.testing.assert_close(enabled_losses.l2c2_loss, torch.tensor(0.75))
    torch.testing.assert_close(
        enabled_losses.ppo_loss,
        disabled_losses.ppo_loss + torch.tensor(0.75),
    )


def test_update_delegates_loss_construction_to_compute_losses() -> None:
    tree = ast.parse(_PPO_SOURCE.read_text())
    update_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "update"
    )
    compute_losses_calls = [
        node
        for node in ast.walk(update_node)
        if isinstance(node, ast.Call) and _name(node.func) == "compute_losses"
    ]
    direct_l2c2_compute_loss_calls = [
        node
        for node in ast.walk(update_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute_loss"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "l2c2"
    ]

    assert len(compute_losses_calls) == 1
    assert direct_l2c2_compute_loss_calls == []


def test_default_runner_obs_groups_do_not_include_l2c2_clean_actor_group() -> None:
    cfg = MjPgOnPolicyRunnerCfg()

    assert cfg.obs_groups == {
        "actor": ("actor",),
        "critic": ("critic",),
    }


def test_l2c2_config_is_keyword_only_in_mjpgppo_constructor() -> None:
    signature = inspect.signature(MjPgPpo)

    assert signature.parameters["l2c2_cfg"].kind is inspect.Parameter.KEYWORD_ONLY


def test_l2c2_config_does_not_expose_legacy_clean_obs_group() -> None:
    assert "clean_obs_group" not in L2C2Cfg.__dataclass_fields__


def test_l2c2_constructor_rejects_legacy_clean_obs_group() -> None:
    with pytest.raises(TypeError, match="clean_obs_group"):
        L2C2(clean_obs_group="legacy_actor_clean")


def test_resolve_l2c2_config_derives_single_actor_clean_mapping() -> None:
    obs = TensorDict(
        {
            "actor": torch.zeros(2, 3),
            "actor_clean": torch.ones(2, 3),
        },
        batch_size=[2],
    )
    alg_cfg = {"l2c2_cfg": {"enable": True, "clean_obs_suffix": "_clean"}}

    resolved = resolve_l2c2_config(alg_cfg, obs, {"actor": ["actor"]})

    assert resolved["l2c2_cfg"]["clean_obs_groups"] == {"actor": "actor_clean"}


def test_resolve_l2c2_config_derives_multiple_actor_clean_mappings() -> None:
    obs = TensorDict(
        {
            "actor": torch.zeros(2, 3),
            "actor_clean": torch.ones(2, 3),
            "actor_history": torch.zeros(2, 6),
            "actor_history_clean": torch.ones(2, 6),
        },
        batch_size=[2],
    )
    alg_cfg = {"l2c2_cfg": {"enable": True, "clean_obs_suffix": "_clean"}}

    resolved = resolve_l2c2_config(
        alg_cfg,
        obs,
        {"actor": ["actor", "actor_history"]},
    )

    assert resolved["l2c2_cfg"]["clean_obs_groups"] == {
        "actor": "actor_clean",
        "actor_history": "actor_history_clean",
    }


def test_resolve_l2c2_config_rejects_missing_clean_observation() -> None:
    obs = TensorDict({"actor": torch.zeros(2, 3)}, batch_size=[2])
    alg_cfg = {"l2c2_cfg": {"enable": True, "clean_obs_suffix": "_clean"}}

    with pytest.raises(KeyError, match="actor_clean"):
        resolve_l2c2_config(alg_cfg, obs, {"actor": ["actor"]})


def test_resolve_l2c2_config_rejects_shape_mismatch() -> None:
    obs = TensorDict(
        {
            "actor": torch.zeros(2, 3),
            "actor_clean": torch.ones(2, 4),
        },
        batch_size=[2],
    )
    alg_cfg = {"l2c2_cfg": {"enable": True, "clean_obs_suffix": "_clean"}}

    with pytest.raises(ValueError, match="same shape"):
        resolve_l2c2_config(alg_cfg, obs, {"actor": ["actor"]})


def test_resolve_l2c2_config_rejects_legacy_clean_obs_group_config() -> None:
    obs = TensorDict(
        {
            "actor": torch.zeros(2, 3),
            "legacy_actor_clean": torch.ones(2, 3),
        },
        batch_size=[2],
    )
    alg_cfg = {
        "l2c2_cfg": {
            "enable": True,
            "clean_obs_group": "legacy_actor_clean",
            "clean_obs_suffix": "_clean",
        }
    }

    with pytest.raises(ValueError, match="clean_obs_group.*deprecated"):
        resolve_l2c2_config(alg_cfg, obs, {"actor": ["actor"]})


def test_l2c2_build_clean_observations_uses_resolved_mapping() -> None:
    obs = TensorDict(
        {
            "actor": torch.zeros(2, 3),
            "reference_actor": torch.ones(2, 3),
        },
        batch_size=[2],
    )
    l2c2 = L2C2(clean_obs_groups={"actor": "reference_actor"})

    clean_obs = l2c2.build_clean_observations(obs, ("actor",))

    torch.testing.assert_close(clean_obs["actor"], obs["reference_actor"])
