"""L2C2 actor regularization"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


@dataclass
class L2C2Loss:
    l2c2_loss: torch.Tensor
    weighted: torch.Tensor
    input_dist: torch.Tensor
    output_dist: torch.Tensor


class L2C2:
    def __init__(
        self,
        enable: bool = True,
        lambda_l2c2: float = 0.1,
        clean_obs_suffix: str = "_clean",
        clean_obs_groups: dict[str, str] | None = None,
        eps: float = 1e-8,
        device: str = "cpu",
    ) -> None:
        self.enable = enable
        self.lambda_l2c2 = lambda_l2c2
        self.clean_obs_suffix = clean_obs_suffix
        self.clean_obs_groups = clean_obs_groups
        self.eps = eps
        self.device = device

    def validate_models(self, actor: MLPModel, critic: MLPModel) -> None:
        if not self.enable:
            return
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError(
                "L2C2 regularization is not supported for recurrent policies. "
                "Disable l2c2_cfg.enable or switch to non-recurrent models."
            )

    def build_clean_observations(
        self,
        obs_td: TensorDict,
        actor_obs_groups: tuple[str, ...],
    ) -> TensorDict:
        clean_td = obs_td.clone(recurse=False)
        for key in actor_obs_groups:
            clean_key = self._resolve_clean_obs_group(obs_td, key, actor_obs_groups)
            clean_td[key] = obs_td[clean_key]
        return clean_td

    def _resolve_clean_obs_group(
        self,
        obs_td: TensorDict,
        actor_group: str,
        actor_obs_groups: tuple[str, ...],
    ) -> str:
        if self.clean_obs_groups is not None:
            if actor_group not in self.clean_obs_groups:
                raise KeyError(
                    f"L2C2 resolved clean observation groups do not include actor group '{actor_group}'."
                )
            clean_group = self.clean_obs_groups[actor_group]
            if clean_group not in obs_td.keys():
                raise KeyError(
                    f"L2C2 requires resolved clean observation group '{clean_group}' "
                    f"for actor group '{actor_group}'."
                )
            return clean_group

        clean_group = f"{actor_group}{self.clean_obs_suffix}"
        if clean_group in obs_td.keys():
            return clean_group
        raise KeyError(
            f"L2C2 requires '{clean_group}' in observations for actor group '{actor_group}'. "
            "Ensure the env config publishes it (see sync_actor_clean_observation_groups)."
        )

    def compute_actor_loss(
        self,
        actor: MLPModel,
        mu_noisy: torch.Tensor,
        noisy_obs_td: TensorDict,
        clean_obs_td: TensorDict,
        actor_obs_groups: tuple[str, ...],
    ) -> L2C2Loss:
        sq_sum = torch.zeros((), device=mu_noisy.device, dtype=mu_noisy.dtype)
        numel = 0
        for key in actor_obs_groups:
            diff = noisy_obs_td[key] - clean_obs_td[key]
            sq_sum = sq_sum + diff.pow(2).sum()
            numel += diff.numel()
        input_dist = (sq_sum / max(numel, 1)).detach()

        mu_clean = actor(clean_obs_td, stochastic_output=False)
        output_dist = (mu_noisy - mu_clean).pow(2).mean()
        l2c2_loss = output_dist / (input_dist + self.eps)
        weighted = self.lambda_l2c2 * l2c2_loss

        return L2C2Loss(
            l2c2_loss=l2c2_loss,
            weighted=weighted,
            input_dist=input_dist,
            output_dist=output_dist.detach(),
        )

    def compute_loss(
        self,
        actor: MLPModel,
        batch: RolloutStorage.Batch,
        original_batch_size: int,
    ) -> L2C2Loss:
        obs_slice = batch.observations[:original_batch_size]
        clean_slice = self.build_clean_observations(obs_slice, actor.obs_groups)
        mu_noisy = actor.output_mean[:original_batch_size]
        return self.compute_actor_loss(
            actor=actor,
            mu_noisy=mu_noisy,
            noisy_obs_td=obs_slice,
            clean_obs_td=clean_slice,
            actor_obs_groups=actor.obs_groups,
        )


def resolve_l2c2_config(
    alg_cfg: dict[str, Any],
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
) -> dict[str, Any]:
    cfg = alg_cfg.get("l2c2_cfg")
    if cfg is None or not cfg.get("enable", False):
        alg_cfg["l2c2_cfg"] = None
        return alg_cfg

    if "clean_obs_group" in cfg:
        raise ValueError(
            "L2C2 clean_obs_group is deprecated. Publish clean observations with "
            "the configured clean_obs_suffix and let resolve_l2c2_config derive clean_obs_groups."
        )

    suffix = cfg.get("clean_obs_suffix", "_clean")
    actor_groups = tuple(obs_groups["actor"])
    clean_obs_groups: dict[str, str] = {}
    for actor_group in actor_groups:
        clean_group = f"{actor_group}{suffix}"
        if clean_group not in obs.keys():
            raise KeyError(
                f"L2C2 requires '{clean_group}' in observations for actor group '{actor_group}'. "
                "Ensure the env config publishes it (see sync_actor_clean_observation_groups)."
            )
        if obs[actor_group].shape != obs[clean_group].shape:
            raise ValueError(
                f"L2C2 clean observation group '{clean_group}' must have the same shape as "
                f"actor observation group '{actor_group}'. Got {obs[clean_group].shape} and "
                f"{obs[actor_group].shape}."
            )
        clean_obs_groups[actor_group] = clean_group

    cfg["clean_obs_groups"] = clean_obs_groups
    alg_cfg["l2c2_cfg"] = cfg
    return alg_cfg
