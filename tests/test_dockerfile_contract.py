"""Static checks for the Docker development image contract."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"


def read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def dockerignore_entries() -> set[str]:
  return {
    line.strip()
    for line in read(DOCKERIGNORE).splitlines()
    if line.strip() and not line.strip().startswith("#")
  }


def test_dockerfile_uses_cuda128_runtime_base() -> None:
  text = read(DOCKERFILE)
  assert "FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04" in text


def test_dockerfile_bakes_locked_non_dev_runtime() -> None:
  text = read(DOCKERFILE)
  assert "uv python install 3.13" in text
  assert "uv sync --locked --no-install-project --no-editable --no-dev" in text
  assert "uv sync --locked --no-editable --no-dev" in text
  assert "UV_NO_SYNC=1" in text
  assert "UV_PROJECT_ENVIRONMENT=/app/.venv" in text
  assert "VIRTUAL_ENV=/app/.venv" in text
  assert "MUJOCO_GL=egl" in text
  assert "pytest" not in text.lower()


def test_dockerfile_installs_runtime_support_packages() -> None:
  text = read(DOCKERFILE)
  for package in ("git", "curl", "libegl-dev"):
    assert package in text


def test_dockerfile_uses_pinned_uv_image_tag() -> None:
  text = read(DOCKERFILE)
  assert "ghcr.io/astral-sh/uv:latest" not in text
  assert "COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/" in text


def test_dockerignore_excludes_local_runtime_artifacts() -> None:
  entries = dockerignore_entries()
  expected = {
    ".git",
    ".venv",
    ".container_home",
    ".container_cache",
    ".ruff_cache",
    "logs",
    "wandb",
    "artifacts",
    "*.pt",
    "*.onnx",
    "*.tar",
  }
  assert expected <= entries


def test_dockerignore_excludes_pytest_cache() -> None:
  assert ".pytest_cache" in dockerignore_entries()
