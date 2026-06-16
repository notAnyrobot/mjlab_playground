"""Tests for RL extension wiring."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch
from mjlab_playground.rl_extensions import MjpOnPolicyRunnerCfg, MjpPpo
from mjlab_playground.rl_extensions.l2c2 import L2C2, L2C2Cfg, resolve_l2c2_config
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


def test_default_runner_obs_groups_do_not_include_l2c2_clean_actor_group() -> None:
    cfg = MjpOnPolicyRunnerCfg()

    assert cfg.obs_groups == {
        "actor": ("actor",),
        "critic": ("critic",),
    }


def test_l2c2_config_is_keyword_only_in_mjpppo_constructor() -> None:
    signature = inspect.signature(MjpPpo)

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
