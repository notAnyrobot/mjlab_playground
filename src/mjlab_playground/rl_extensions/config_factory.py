"""Config factory for creating RL configurations.

This module provides a set of functions for creating RL configurations.
"""

from copy import deepcopy
from typing import Any, Literal

from mjlab_playground.rl_extensions.config import (
    MjPgModelCfg,
    MjPgOnPolicyRunnerCfg,
    MjPgPpoAlgorithmCfg,
)
from mjlab_playground.rl_extensions.l2c2 import L2C2Cfg

MODEL_CLASS = {
    "mlp": "MLPModel",
    "cnn": "CNNModel",
    "temporal_cnn": "mjlab_playground.rl_extensions.temporal_cnn:TemporalCNNModel",
}

def _build_temporal_cnn_cfg(
    cnn_cfg: dict[str, Any],
    *,
    prefix: Literal["actor", "critic"],
) -> dict[str, dict[str, Any]]:
    return {
        f"{prefix}_history": deepcopy(cnn_cfg),
        f"{prefix}_future": {**deepcopy(cnn_cfg), "kernel_size": 5},
    }


def create_mjpg_ppo_runner_cfg(
    num_critics: int | None = None,
    model_type: Literal["mlp", "temporal_cnn"] = "mlp",
    cnn_cfg: dict[str, Any] | None = None,
    l2c2_cfg: L2C2Cfg | None = None,
) -> MjPgOnPolicyRunnerCfg:
    """Create a PPO runner configuration.

    Args:
        num_critics: Number of critics to use. If None, defaults to 1.
        model_type: Type of model to use. Can be either "mlp" or "temporal_cnn".
        cnn_cfg: Optional CNN configuration for temporal CNN models.

    Returns:
        An instance of MjPgOnPolicyRunnerCfg with the specified settings.
    """
    if model_type == "temporal_cnn" and cnn_cfg is not None:
        actor_cnn_cfg = _build_temporal_cnn_cfg(cnn_cfg, prefix="actor")
        critic_cnn_cfg = _build_temporal_cnn_cfg(cnn_cfg, prefix="critic")
    else:
        actor_cnn_cfg = None
        critic_cnn_cfg = None

    hidden_dims = (1024, 1024, 512, 256, 128) if model_type == "temporal_cnn" else (512, 256, 128)

    actor_model_cfg = MjPgModelCfg(
        class_name=MODEL_CLASS[model_type],
        hidden_dims=hidden_dims,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        cnn_cfg=actor_cnn_cfg,
    )

    critic_model_cfg = MjPgModelCfg(
        class_name=MODEL_CLASS[model_type],
        hidden_dims=hidden_dims,
        cnn_cfg=critic_cnn_cfg,
    )

    algorithm_cfg = MjPgPpoAlgorithmCfg(
        l2c2_cfg=l2c2_cfg
    )

    return MjPgOnPolicyRunnerCfg(
        actor=actor_model_cfg,
        critic=critic_model_cfg,
        algorithm=algorithm_cfg,
    )