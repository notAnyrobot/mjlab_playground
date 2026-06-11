"""Static checks for the Docker development image contract."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
UV_SYNC_SCRIPT = ROOT / "scripts" / "docker_uv_sync_locked.sh"


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
  assert "rm -rf /root/.local/share/uv/python/.temp" in text
  assert "uv-sync-locked --no-install-workspace --no-editable --no-dev" in text
  assert "uv-sync-locked --no-editable --no-dev" in text
  assert "UV_NO_SYNC=1" in text
  assert "UV_PROJECT_ENVIRONMENT=/app/.venv" in text
  assert "VIRTUAL_ENV=/app/.venv" in text
  assert "MUJOCO_GL=egl" in text
  assert "pytest" not in text.lower()


def test_dockerfile_keeps_retry_logic_out_of_build_steps() -> None:
  text = read(DOCKERFILE)
  assert "for attempt" not in text
  assert "retry-command" not in text
  assert "APT_INSTALL_TIMEOUT" not in text
  assert "UV_PYTHON_INSTALL_TIMEOUT" not in text
  assert "UV_SYNC_TIMEOUT" not in text
  assert "COPY scripts/docker_uv_sync_locked.sh /usr/local/bin/uv-sync-locked" in text


def test_uv_sync_helper_sets_index_without_retrying() -> None:
  uv_sync_text = read(UV_SYNC_SCRIPT)

  assert 'if [ -n "${PIP_INDEX_URL:-}" ]; then' in uv_sync_text
  assert "uv lock --check" in uv_sync_text
  assert 'export UV_DEFAULT_INDEX="$PIP_INDEX_URL"' in uv_sync_text
  assert 'exec uv sync --frozen "$@"' in uv_sync_text
  assert 'exec uv sync --locked "$@"' in uv_sync_text
  assert "retry-command" not in uv_sync_text


def test_dockerfile_uses_local_mjlab_build_context() -> None:
  text = read(DOCKERFILE)
  assert "WORKDIR /workspace/mujocolab/mjlab_playground" in text
  assert "from=mjlab" in text
  assert "COPY --from=mjlab . /workspace/mujocolab/mjlab" in text
  assert "github.com/mujocolab/mjlab" not in text


def test_dockerfile_installs_runtime_support_packages() -> None:
  text = read(DOCKERFILE)
  for package in ("git", "curl", "libegl-dev"):
    assert package in text


def test_dockerfile_uses_pinned_uv_image_tag() -> None:
  text = read(DOCKERFILE)
  assert "ghcr.io/astral-sh/uv:latest" not in text
  assert "COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/" in text


def test_dockerfile_hardens_git_fetches_for_slow_networks() -> None:
  text = read(DOCKERFILE)
  assert "GIT_HTTP_LOW_SPEED_LIMIT=1" in text
  assert "GIT_HTTP_LOW_SPEED_TIME=600" in text
  assert "git config --global http.version HTTP/1.1" in text


def test_dockerfile_accepts_apt_mirror_build_arg() -> None:
  text = read(DOCKERFILE)
  assert "ARG APT_MIRROR=" in text
  assert "archive.ubuntu.com/ubuntu" in text
  assert "security.ubuntu.com/ubuntu" in text
  assert "--no-install-recommends" in text


def test_dockerfile_accepts_network_mirror_build_args() -> None:
  text = read(DOCKERFILE)
  uv_sync_text = read(UV_SYNC_SCRIPT)
  assert "ARG PIP_INDEX_URL=" in text
  assert "ARG UV_PYTHON_INSTALL_MIRROR=" in text
  assert "ARG UV_HTTP_TIMEOUT=3600" in text
  assert "ARG UV_CONCURRENT_DOWNLOADS=1" in text
  assert "ENV UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}" in text
  assert "ENV UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS}" in text
  assert 'if [ -n "${PIP_INDEX_URL:-}" ]; then' in uv_sync_text
  assert "uv lock --check" in uv_sync_text
  assert 'export UV_DEFAULT_INDEX="$PIP_INDEX_URL"' in uv_sync_text
  assert 'exec uv sync --frozen "$@"' in uv_sync_text


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
