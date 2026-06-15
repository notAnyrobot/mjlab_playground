"""Modular extensions of rsl_rl."""

from mjlab_playground.rl_extensions.config import (
    MjpModelCfg as MjpModelCfg,
)
from mjlab_playground.rl_extensions.config import (
    MjpOnPolicyRunnerCfg as MjpOnPolicyRunnerCfg,
)
from mjlab_playground.rl_extensions.config import (
    MjpPpoAlgorithmCfg as MjpPpoAlgorithmCfg,
)

__all__ = ["MjpModelCfg", "MjpOnPolicyRunnerCfg", "MjpPpoAlgorithmCfg"]
