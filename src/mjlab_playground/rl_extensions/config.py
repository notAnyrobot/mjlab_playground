"""Extension configuration for the RL algorithms."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from mjlab_playground.rl_extensions.l2c2 import L2C2Cfg


@dataclass
class MjpModelCfg(RslRlModelCfg):
    pass


@dataclass
class MjpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Config of PPO algorithm."""

    l2c2_cfg: L2C2Cfg | None = field(default_factory=L2C2Cfg)
    """Optional L2C2 configuration"""

    rnd_cfg: dict[str, Any] | None = None
    """Optional RSL-RL random network distillation config."""

    symmetry_cfg: dict[str, Any] | None = None
    """Optional RSL-RL symmetry augmentation config."""

    class_name: str = "mjlab_playground.rl_extensions.MjRlPpo"
    """Algorithm class name resolved to MjpRlPpo by RSL-RL."""


@dataclass
class MjpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    pass
