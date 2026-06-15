# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from mjlab_playground.rl_extensions.l2c2 import L2C2


@dataclass
class MjpRlPpoLosses:
    """Container for the losses of the MjpRlPpo algorithm."""

    policy_loss: torch.Tensor
    """The policy loss."""

    value_loss: torch.Tensor
    """The value loss."""

    entropy_loss: torch.Tensor
    """The entropy loss."""

    l2c2_loss: torch.Tensor | None
    """The L2C2 loss."""

    rnd_loss: torch.Tensor | None
    """The RND loss."""

    symmetry_loss: torch.Tensor | None
    """The symmetry loss."""


class MjpRlPpo(PPO):
    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # L2C2 parameters
        l2c2_cfg: dict[str, Any] | None = None,
        *ppo_args: Any,
        **ppo_kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, storage, *ppo_args, **ppo_kwargs)

        # L2C2 regularization
        if l2c2_cfg is not None:
            self.l2c2 = L2C2(device=self.device, **l2c2_cfg)
