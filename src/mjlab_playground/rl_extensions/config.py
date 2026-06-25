"""Extension configuration for the RL algorithms."""

from dataclasses import dataclass, field
from typing import Any, Literal

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from mjlab_playground.rl_extensions.l2c2 import L2C2Cfg


@dataclass
class MjPgModelCfg(RslRlModelCfg):
    pass


@dataclass
class MjPgPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Config of PPO algorithm."""

    l2c2_cfg: L2C2Cfg | None = field(default_factory=L2C2Cfg)
    """Optional L2C2 configuration"""

    rnd_cfg: dict[str, Any] | None = None
    """Optional RSL-RL random network distillation config."""

    symmetry_cfg: dict[str, Any] | None = None
    """Optional RSL-RL symmetry augmentation config."""

    class_name: str = "mjlab_playground.rl_extensions.MjPgPpo"
    """Algorithm class name resolved to MjPgPpo by RSL-RL."""


@dataclass
class MjPgOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):

    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "actor": ("actor",),
            "critic": ("critic",),
        },
    )

    save_interval: int = 500
    logger: Literal["wandb", "tensorboard"] = "tensorboard"
    wandb_project: str = "mjlab_playground"
    upload_model: bool = False
