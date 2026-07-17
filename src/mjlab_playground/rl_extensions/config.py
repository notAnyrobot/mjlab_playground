"""Extension configuration for the RL algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from mjlab_playground.rl_extensions.l2c2 import L2C2Cfg

MjPgModelClassName = Literal[
    "MLPModel",
    "CNNModel",
    "RNNModel",
    "mjlab_playground.rl_extensions.temporal_cnn:TemporalCNNModel",
]


@dataclass
class RndCfg:
    """Configuration for RSL-RL random network distillation."""

    num_outputs: int = 1
    """Output width of the target and predictor networks."""

    predictor_hidden_dims: tuple[int, ...] = (-1, -1)
    """Predictor hidden widths; ``-1`` resolves to the RND input width."""

    target_hidden_dims: tuple[int, ...] = (-1,)
    """Target hidden widths; ``-1`` resolves to the RND input width."""

    activation: str = "elu"
    """Activation used by the target and predictor networks."""

    state_normalization: bool = True
    """Whether to normalize the RND input state online."""

    reward_normalization: bool = False
    """Whether to normalize intrinsic rewards online."""

    weight: float = 1.0
    """Intrinsic-reward weight before environment-step scaling."""

    weight_schedule: dict[str, Any] | None = None
    """Optional upstream RSL-RL weight-schedule configuration."""

    learning_rate: float = 0.001
    """Learning rate of the predictor optimizer."""


@dataclass
class MjPgModelCfg(RslRlModelCfg):
    """Model config with playground-specific class-name options."""

    class_name: MjPgModelClassName = "MLPModel"
    """Model class name resolved by RSL-RL."""


@dataclass
class MjPgPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Config of PPO algorithm."""

    l2c2_cfg: L2C2Cfg | None = field(default_factory=L2C2Cfg)
    """Optional L2C2 configuration"""

    rnd_cfg: RndCfg | None = None
    """Optional RSL-RL random network distillation config."""

    symmetry_cfg: dict[str, Any] | None = None
    """Optional RSL-RL symmetry augmentation config."""

    class_name: str = "mjlab_playground.rl_extensions.MjPgPpo"
    """Algorithm class name resolved to MjPgPpo by RSL-RL."""


@dataclass
class MjPgOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    actor: RslRlModelCfg = field(
        default_factory=lambda: MjPgModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
            cnn_cfg=None,
        )
    )
    """The actor model configuration."""

    critic: RslRlModelCfg = field(default_factory=MjPgModelCfg)
    """The critic model configuration."""

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
