"""Stable construction surface for reference-motion viewer adapters."""

from __future__ import annotations

from typing import Any, Literal

__all__ = [
    "VIEWER_MODE_CHOICES",
    "ViewerMode",
    "create_viewer_adapter",
    "resolve_viewer_mode",
]

ViewerMode = Literal["auto", "native", "viser"]
VIEWER_MODE_CHOICES = ("auto", "native", "viser")


def resolve_viewer_mode(
    mode: str,
    *,
    platform_name: str,
    display_available: bool,
) -> Literal["native", "viser"]:
    """Resolve a user-facing viewer choice from injected runtime facts."""
    if mode == "native" or mode == "viser":
        return mode
    if mode == "auto" and platform_name == "darwin":
        return "viser"
    if mode == "auto" and platform_name == "linux":
        return "native" if display_available else "viser"
    raise ValueError(
        f"cannot automatically select a viewer on platform {platform_name!r}"
    )


def create_viewer_adapter(mode: str, *, scene: Any, **kwargs: Any) -> Any:
    """Construct a presentation adapter without exposing its concrete class."""
    if mode == "native":
        from .native import NativeMujocoViewerAdapter

        return NativeMujocoViewerAdapter(scene=scene, **kwargs)
    if mode == "viser":
        from .viser import ViserViewerAdapter

        return ViserViewerAdapter(scene=scene, **kwargs)
    raise ValueError(f"unsupported viewer mode: {mode!r}")
