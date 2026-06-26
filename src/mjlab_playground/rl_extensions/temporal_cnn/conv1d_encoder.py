"""Conv1d temporal encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import get_param, resolve_nn_activation

_POOL_MODULES = {"avg": nn.AdaptiveAvgPool1d, "max": nn.AdaptiveMaxPool1d}


class Conv1dEncoder(nn.Sequential):
    """Encode a ``(B, input_channels, seq_len)`` sequence into ``(B, C)``.

    This module follows the ``rsl_rl.modules`` building-block pattern: it is a
    sequence of registered layers, exposes its output dimension, and keeps
    initialization behavior local to the module.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: tuple[int, ...] | list[int] = (64, 32),
        kernel_size: int | tuple[int, ...] | list[int] = 3,
        activation: str = "elu",
        global_pool: str = "avg",
    ) -> None:
        super().__init__()

        if len(output_channels) == 0:
            raise ValueError("Conv1dEncoder requires at least one output channel.")
        if global_pool not in _POOL_MODULES:
            raise ValueError(
                f"Unsupported global pooling type: {global_pool}. "
                "Supported types are 'avg' and 'max'."
            )

        activation_function = resolve_nn_activation(activation)
        layers: list[nn.Module] = []
        last_channels = input_channels
        for idx, output_channel in enumerate(output_channels):
            k = get_param(kernel_size, idx)
            layers.append(
                nn.Conv1d(
                    in_channels=last_channels,
                    out_channels=output_channel,
                    kernel_size=k,
                    padding=k // 2,
                )
            )
            layers.append(activation_function)
            last_channels = output_channel

        layers.append(_POOL_MODULES[global_pool](1))
        layers.append(nn.Flatten(start_dim=1))
        self._output_dim = output_channels[-1]

        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    @property
    def output_dim(self) -> int:
        """Get the flattened output dimension."""
        return self._output_dim

    def init_weights(self) -> None:
        """Initialize Conv1d weights with Kaiming initialization."""
        for module in self:
            if isinstance(module, nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, input_channels, seq_len)``

        Returns:
            ``(B, output_dim)``
        """
        for layer in self:
            x = layer(x)
        return x
