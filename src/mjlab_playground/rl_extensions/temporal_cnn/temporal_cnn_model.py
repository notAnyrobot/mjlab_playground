"""Temporal-CNN policy model."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from tensordict import TensorDict

from mjlab_playground.rl_extensions.temporal_cnn.conv1d_encoder import Conv1dEncoder


def _export_input_name(group_name: str) -> str:
    """Map observation-group names to stable export input names."""
    if group_name in {"actor", "critic"}:
        return "obs"
    for prefix in ("actor_", "critic_"):
        if group_name.startswith(prefix):
            return "obs_" + group_name[len(prefix):]
    return group_name


def _is_per_group_cnn_cfg(cnn_cfg: dict[str, Any]) -> bool:
    """Return whether ``cnn_cfg`` is keyed by observation group name."""
    return bool(cnn_cfg) and all(isinstance(value, dict) for value in cnn_cfg.values())


class TemporalCNNModel(MLPModel):
    """MLP model extended with 1-D CNN encoders for temporal observations.

    This mirrors ``rsl_rl.models.CNNModel`` for observation groups shaped
    ``(B, T, D)``. 1-D groups ``(B, D)`` are handled by ``MLPModel`` and
    temporal groups are normalized per feature, permuted to ``(B, D, T)``,
    encoded by ``Conv1dEncoder``, and concatenated into the MLP latent.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        """Initialize the temporal CNN model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model.
            output_dim: Dimension of the model output.
            hidden_dims: Hidden dimensions of the MLP head.
            activation: Activation function of the MLP head and default temporal encoders.
            obs_normalization: Whether to normalize observations before the MLP/temporal encoders.
            distribution_cfg: Optional output distribution configuration.
            cnn_cfg: Configuration of the temporal encoder(s). A single encoder config is shared across all temporal
                groups; a dict keyed by temporal group name configures each group separately.
            cnns: Temporal encoder modules to use, e.g. for sharing encoders between actor and critic. If None,
                new encoders are created from ``cnn_cfg``.
        """
        # Resolve observation groups and dimensions before parent construction.
        self._get_obs_dim(obs, obs_groups, obs_set)

        # Create or validate temporal CNN encoders.
        if cnns is not None:
            if set(cnns.keys()) != set(self.obs_groups_3d):
                raise ValueError("The 3D observations must be identical for all models sharing CNN encoders.")
        else:
            if cnn_cfg is None:
                raise ValueError("CNN configurations must be provided if CNNs are not shared.")
            if not _is_per_group_cnn_cfg(cnn_cfg):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_3d}
            if set(cnn_cfg.keys()) != set(self.obs_groups_3d):
                raise ValueError("The CNN configuration keys must match the 3D observation groups.")

            cnns = {}
            for obs_group, obs_dim in zip(self.obs_groups_3d, self.obs_dims_3d, strict=True):
                cnns[obs_group] = Conv1dEncoder(input_channels=obs_dim, **cnn_cfg[obs_group])

        self.cnn_latent_dim = 0
        for cnn in cnns.values():
            if not hasattr(cnn, "output_dim"):
                raise ValueError("Temporal CNN encoders must expose an output_dim property.")
            self.cnn_latent_dim += int(cnn.output_dim)  # type: ignore[attr-defined]

        self._obs_normalization_3d = obs_normalization

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        if isinstance(cnns, nn.ModuleDict):
            self.cnns = cnns
        else:
            self.cnns = nn.ModuleDict(cnns)

        if obs_normalization:
            self.obs_normalizers_3d = nn.ModuleDict(
                {
                    group_name: EmpiricalNormalization(dim_3d)
                    for group_name, dim_3d in zip(self.obs_groups_3d, self.obs_dims_3d, strict=True)
                }
            )
        else:
            self.obs_normalizers_3d = nn.ModuleDict({g: nn.Identity() for g in self.obs_groups_3d})

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Build the latent by combining optional 1-D observations and encoded temporal groups."""
        latent_cnn = torch.cat([self._encode_temporal_obs(obs, group) for group in self.obs_groups_3d], dim=-1)
        if not self.obs_groups:
            return latent_cnn
        latent_1d = super().get_latent(obs, masks, hidden_state)
        return torch.cat([latent_1d, latent_cnn], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update normalization statistics for 1-D and temporal observation groups."""
        if self.obs_groups:
            super().update_normalization(obs)
        if self._obs_normalization_3d:
            for group_name in self.obs_groups_3d:
                h = obs[group_name]
                batch_size, history_length, obs_dim = h.shape
                self.obs_normalizers_3d[group_name].update(h.reshape(batch_size * history_length, obs_dim))  # type: ignore

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchTemporalCNNModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxTemporalCNNModel(self, verbose)

    def _encode_temporal_obs(self, obs: TensorDict, obs_group: str) -> torch.Tensor:
        h = self.obs_normalizers_3d[obs_group](obs[obs_group])
        h = h.permute(0, 2, 1)
        return self.cnns[obs_group](h)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute 1-D plus temporal dimensions."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d: list[str] = []
        obs_dims_3d: list[int] = []
        obs_groups_3d: list[str] = []
        history_lengths: list[int] = []

        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 3:  # B, T, D
                obs_groups_3d.append(obs_group)
                obs_dims_3d.append(obs[obs_group].shape[-1])
                history_lengths.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, D
                obs_groups_1d.append(obs_group)
                obs_dim_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        if not obs_groups_3d:
            raise ValueError("No 3D observations are provided. If this is intentional, use the MLP model instead.")

        self.obs_groups_3d = obs_groups_3d
        self.obs_dims_3d = obs_dims_3d
        self.history_lengths = history_lengths
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.cnn_latent_dim


class _TorchTemporalCNNModel(nn.Module):
    """Exportable TemporalCNNModel for JIT."""

    def __init__(self, model: TemporalCNNModel) -> None:
        """Create a TorchScript-friendly copy of a TemporalCNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.obs_normalizers_3d = nn.ModuleList(
            [copy.deepcopy(model.obs_normalizers_3d[group_name]) for group_name in model.obs_groups_3d]
        )
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[group_name]) for group_name in model.obs_groups_3d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.temporal_input_names = [_export_input_name(group_name) for group_name in model.obs_groups_3d]

    def _encode_temporal_inputs(self, obs_temporal: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if len(obs_temporal) != len(self.cnns):
            raise ValueError(f"Expected {len(self.cnns)} temporal inputs, got {len(obs_temporal)}.")

        latents = []
        for obs_input, normalizer, cnn in zip(obs_temporal, self.obs_normalizers_3d, self.cnns, strict=True):
            h = normalizer(obs_input)
            h = h.permute(0, 2, 1)
            latents.append(cnn(h))
        return latents

    def forward(self, obs_1d: torch.Tensor, *obs_temporal: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from separated 1-D and temporal inputs."""
        latent_1d = self.obs_normalizer(obs_1d)
        latent_parts = [latent_1d, *self._encode_temporal_inputs(obs_temporal)]
        out = self.mlp(torch.cat(latent_parts, dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for Temporal CNN exports)."""
        pass


class _OnnxTemporalCNNModel(nn.Module):
    """Exportable TemporalCNNModel for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: TemporalCNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a TemporalCNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.obs_normalizers_3d = nn.ModuleList(
            [copy.deepcopy(model.obs_normalizers_3d[group_name]) for group_name in model.obs_groups_3d]
        )
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[group_name]) for group_name in model.obs_groups_3d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self._obs_dim_1d = model.obs_dim
        self._obs_dims_3d = list(model.obs_dims_3d)
        self._history_lengths = list(model.history_lengths)
        self._temporal_input_names = [_export_input_name(group_name) for group_name in model.obs_groups_3d]

    def _encode_temporal_inputs(self, obs_temporal: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if len(obs_temporal) != len(self.cnns):
            raise ValueError(f"Expected {len(self.cnns)} temporal inputs, got {len(obs_temporal)}.")

        latents = []
        for obs_input, normalizer, cnn in zip(obs_temporal, self.obs_normalizers_3d, self.cnns, strict=True):
            h = normalizer(obs_input)
            h = h.permute(0, 2, 1)
            latents.append(cnn(h))
        return latents

    def forward(self, obs_1d: torch.Tensor, *obs_temporal: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_1d = self.obs_normalizer(obs_1d)
        latent_parts = [latent_1d, *self._encode_temporal_inputs(obs_temporal)]
        out = self.mlp(torch.cat(latent_parts, dim=-1))
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        temporal_inputs = [
            torch.zeros(1, history_length, obs_dim)
            for history_length, obs_dim in zip(self._history_lengths, self._obs_dims_3d, strict=True)
        ]
        return (torch.zeros(1, self._obs_dim_1d), *temporal_inputs)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self._temporal_input_names]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
