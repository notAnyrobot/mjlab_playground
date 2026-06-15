"""Tests for RL extension wiring."""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PPO_SOURCE = _ROOT / "src/mjlab_playground/rl_extensions/algorithms/ppo.py"
_L2C2_INIT_SOURCE = _ROOT / "src/mjlab_playground/rl_extensions/l2c2/__init__.py"


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_l2c2_config_is_consumed_as_resolved_dict() -> None:
    tree = ast.parse(_PPO_SOURCE.read_text())

    normalize_refs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "normalize_l2c2_config"
    ]
    assert normalize_refs == []

    l2c2_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _name(node.func) == "L2C2"
    ]
    assert any(
        any(
            keyword.arg is None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "l2c2_cfg"
            for keyword in call.keywords
        )
        for call in l2c2_calls
    )


def test_l2c2_package_does_not_reexport_normalization_helper() -> None:
    assert "normalize_l2c2_config" not in _L2C2_INIT_SOURCE.read_text()
