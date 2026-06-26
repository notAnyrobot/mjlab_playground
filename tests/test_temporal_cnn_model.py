"""Tests for temporal CNN model modules."""

from __future__ import annotations

from dataclasses import fields
from typing import get_args, get_type_hints

import pytest
import torch
import torch.nn as nn
from mjlab_playground.rl_extensions.config import MjPgModelCfg
from mjlab_playground.rl_extensions.config_factory import create_mjpg_ppo_runner_cfg
from mjlab_playground.rl_extensions.temporal_cnn import (
    Conv1dEncoder,
    TemporalCNNModel,
)
from tensordict import TensorDict


def test_conv1d_encoder_is_a_fundamental_sequential_module() -> None:
    encoder = Conv1dEncoder(input_channels=4, output_channels=(8, 6))

    assert isinstance(encoder, nn.Sequential)
    assert not hasattr(encoder, "net")
    assert [type(module) for module in encoder] == [
        nn.Conv1d,
        nn.ELU,
        nn.Conv1d,
        nn.ELU,
        nn.AdaptiveAvgPool1d,
        nn.Flatten,
    ]
    assert encoder.output_dim == 6


def test_conv1d_encoder_forward_flattens_global_pool_output() -> None:
    encoder = Conv1dEncoder(input_channels=4, output_channels=(8, 6))

    y = encoder(torch.zeros(3, 4, 11))

    assert y.shape == (3, 6)


def test_temporal_cnn_model_builds_from_single_cnn_cfg_like_cnn_model() -> None:
    obs = TensorDict(
        {
            "actor_proprio": torch.zeros(2, 3),
            "actor_history": torch.zeros(2, 4, 5),
        },
        batch_size=[2],
    )
    model = TemporalCNNModel(
        obs=obs,
        obs_groups={"actor": ["actor_proprio", "actor_history"]},
        obs_set="actor",
        output_dim=7,
        hidden_dims=(8,),
        cnn_cfg={"output_channels": (6,), "kernel_size": 3},
    )

    assert model.obs_groups == ["actor_proprio"]
    assert model.obs_groups_3d == ["actor_history"]
    assert model.obs_dims_3d == [5]
    assert model.history_lengths == [4]
    assert model.cnn_latent_dim == 6
    assert model(obs).shape == (2, 7)


def test_temporal_cnn_model_requires_temporal_cnn_config_when_not_sharing_encoders() -> None:
    obs = TensorDict(
        {"actor_proprio": torch.zeros(2, 3), "actor_history": torch.zeros(2, 4, 5)},
        batch_size=[2],
    )

    with pytest.raises(ValueError, match="CNN configurations must be provided"):
        TemporalCNNModel(
            obs=obs,
            obs_groups={"actor": ["actor_proprio", "actor_history"]},
            obs_set="actor",
            output_dim=7,
        )


def test_temporal_cnn_model_supports_temporal_only_observations() -> None:
    obs = TensorDict({"actor_history": torch.zeros(2, 4, 5)}, batch_size=[2])
    model = TemporalCNNModel(
        obs=obs,
        obs_groups={"actor": ["actor_history"]},
        obs_set="actor",
        output_dim=7,
        hidden_dims=(8,),
        cnn_cfg={"output_channels": (6,), "kernel_size": 3},
    )

    assert model.obs_groups == []
    assert model(obs).shape == (2, 7)


def test_temporal_cnn_configs_use_upstream_cnn_cfg_field() -> None:
    model_field_names = {field.name for field in fields(MjPgModelCfg)}
    cfg = create_mjpg_ppo_runner_cfg(
        model_type="temporal_cnn",
        cnn_cfg={"output_channels": (6,), "kernel_size": 3},
    )

    assert "cnn_cfg" in model_field_names
    assert "cnn_cfg_map" not in model_field_names
    assert cfg.actor.class_name == "mjlab_playground.rl_extensions.temporal_cnn:TemporalCNNModel"
    assert set(cfg.actor.cnn_cfg) == {"actor_history", "actor_future"}
    assert set(cfg.critic.cnn_cfg) == {"critic_history", "critic_future"}


def test_mjpg_model_cfg_advertises_temporal_cnn_model_class_option() -> None:
    class_name_options = get_args(get_type_hints(MjPgModelCfg)["class_name"])

    assert "MLPModel" in class_name_options
    assert "CNNModel" in class_name_options
    assert "RNNModel" in class_name_options
    assert "mjlab_playground.rl_extensions.temporal_cnn:TemporalCNNModel" in class_name_options
