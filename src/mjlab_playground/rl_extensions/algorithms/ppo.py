# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import (
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from mjlab_playground.rl_extensions.l2c2 import L2C2, resolve_l2c2_config


@dataclass
class MjPgPpoLosses:
    """Container for the losses of the MjPgPpo algorithm."""

    ppo_loss: torch.Tensor
    """The PPO loss used for the actor/critic optimizer."""

    surrogate_loss: torch.Tensor
    """The surrogate policy loss."""

    value_loss: torch.Tensor
    """The value loss."""

    entropy: torch.Tensor
    """The mean entropy."""

    rnd_loss: torch.Tensor | None = None
    """The RND loss."""

    symmetry_loss: torch.Tensor | None = None
    """The symmetry loss."""

    l2c2_loss: torch.Tensor | None = None
    """The L2C2 loss."""


@dataclass
class MjPgPpoLossContext:
    """Inputs needed to compute losses for one PPO mini-batch."""

    batch: RolloutStorage.Batch
    """The mini-batch from rollout storage."""

    original_batch_size: int
    """Number of non-augmented samples in the mini-batch."""

    actions_log_prob: torch.Tensor
    """Current action log probabilities."""

    values: torch.Tensor
    """Current critic values."""

    entropy: torch.Tensor
    """Current actor entropy for original samples."""


class MjPgPpo(PPO):
    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *ppo_args: Any,
        # L2C2 parameters
        l2c2_cfg: dict[str, Any] | None = None,
        **ppo_kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, storage, *ppo_args, **ppo_kwargs)

        # L2C2 regularization
        self.l2c2: L2C2 | None = None
        if l2c2_cfg is not None:
            self.l2c2 = L2C2(device=self.device, **l2c2_cfg)
            self.l2c2.validate_models(actor, critic)

    def compute_losses(self, context: MjPgPpoLossContext) -> MjPgPpoLosses:
        """Compute PPO loss components for one mini-batch."""
        batch = context.batch

        # Surrogate loss
        ratio = torch.exp(context.actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
        surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
        surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = batch.values + (context.values - batch.values).clamp(-self.clip_param, self.clip_param)
            value_losses = (context.values - batch.returns).pow(2)
            value_losses_clipped = (value_clipped - batch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (batch.returns - context.values).pow(2).mean()

        entropy = context.entropy.mean()
        ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        rnd_loss = (
            self.rnd.compute_loss(batch.observations[: context.original_batch_size])  # type: ignore
            if self.rnd
            else None
        )
        symmetry_loss = None
        if self.symmetry:
            symmetry_loss = self.symmetry.compute_loss(self.actor, batch, context.original_batch_size)
            if self.symmetry.use_mirror_loss:
                ppo_loss = ppo_loss + self.symmetry.mirror_loss_coeff * symmetry_loss
        l2c2_loss = None
        if self.l2c2:
            l2c2_result = self.l2c2.compute_loss(self.actor, batch, context.original_batch_size)
            l2c2_loss = l2c2_result.weighted
            ppo_loss = ppo_loss + l2c2_loss

        return MjPgPpoLosses(
            ppo_loss=ppo_loss,
            surrogate_loss=surrogate_loss,
            value_loss=value_loss,
            entropy=entropy,
            rnd_loss=rnd_loss,
            symmetry_loss=symmetry_loss,
            l2c2_loss=l2c2_loss,
        )


    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # L2C2 loss
        mean_l2c2_loss = 0 if self.l2c2 else None

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            losses = self.compute_losses(
                MjPgPpoLossContext(
                    batch=batch,
                    original_batch_size=original_batch_size,
                    actions_log_prob=actions_log_prob,
                    values=values,
                    entropy=entropy,
                )
            )

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            losses.ppo_loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                losses.rnd_loss.backward()  # type: ignore

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += losses.value_loss.item()
            mean_surrogate_loss += losses.surrogate_loss.item()
            mean_entropy += losses.entropy.item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += losses.rnd_loss.item()  # type: ignore
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += losses.symmetry_loss.item()  # type: ignore
            # L2C2 loss
            if mean_l2c2_loss is not None:
                mean_l2c2_loss += losses.l2c2_loss.item()  # type: ignore

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_l2c2_loss is not None:
            mean_l2c2_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.l2c2:
            loss_dict["l2c2"] = mean_l2c2_loss

        # Clear the storage
        self.storage.clear()

        return loss_dict


    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Resolve l2c2 config
        cfg["algorithm"] = resolve_l2c2_config(cfg["algorithm"], obs, cfg["obs_groups"])

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: PPO = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg
