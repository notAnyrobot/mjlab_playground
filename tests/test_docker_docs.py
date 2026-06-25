"""Documentation checks for the CUDA 12.8 Docker workflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "docker_cuda128_workflow.md"
README = ROOT / "README.md"
SCRIPTS_README = ROOT / "scripts" / "README.md"


def test_readme_links_docker_workflow() -> None:
  text = README.read_text(encoding="utf-8")
  assert "docs/docker_cuda128_workflow.md" in text
  assert "mjlab-playground:cuda128-dev" in text


def test_docker_workflow_documents_core_commands() -> None:
  text = DOC.read_text(encoding="utf-8")
  required = [
    "--build-context mjlab=/home/android/Code/mujocolab/mjlab",
    "--build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/",
    "--build-arg UV_PYTHON_INSTALL_MIRROR=https://python-standalone.org/mirror/astral-sh/python-build-standalone/",
    "--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
    "--build-arg UV_HTTP_TIMEOUT=3600",
    "--build-arg UV_CONCURRENT_DOWNLOADS=1",
    "./scripts/launch_docker_ws.sh nvidia-smi",
    "./scripts/launch_docker_ws.sh run list-envs --keyword Getup",
    "docker save mjlab-playground:cuda128-dev",
    "docker load -i /data/atom7/Data/docker_images/mjlab_playground_cuda128_dev.tar",
    "./scripts/launch_docker_hpc.sh print-config",
    "UV_NO_SYNC=1",
    "UV_PROJECT_ENVIRONMENT=/app/.venv",
    "Do not rebuild",
    "Rebuild",
    "pytest",
    "not installed",
    "MJLAB_REPO_ROOT",
    "UV_PYTHON_INSTALL_MIRROR",
    "APT_MIRROR",
    "Host-level",
    "not inherited",
    "sync helper first checks",
    "default `uv` package index",
    "large CUDA wheel downloads",
    "limits simultaneous package",
    "MJLAB_PLAYGROUND_GPU_MODE=manual",
    "MJLAB_PLAYGROUND_HPC_NVIDIA_LIBS",
    "MJLAB_PLAYGROUND_GPU_MODE=gpus",
    "rootless Docker",
    "ldconfig",
    "/usr/local/cuda/compat",
    "does not mount the whole host library directory",
    "train Mjlab-Getup-Flat-Unitree-Go1",
    "--gpu-ids '[0,1]'",
    "--agent.logger tensorboard",
    "Workers exited without errors.",
    "CUDA peer access: Not supported",
    "OpenGL.platform.ctypesloader: Failed to load libOpenGL.so.0",
    "inside the container, use `train` directly",
  ]
  for needle in required:
    assert needle in text


def test_docker_workflow_keeps_launcher_generic() -> None:
  text = DOC.read_text(encoding="utf-8")
  assert "getup-smoke" not in text
  assert "train-smoke" not in text
  assert "./scripts/launch_docker_ws.sh run train" in text


def test_scripts_readme_documents_container_scripts() -> None:
  text = SCRIPTS_README.read_text(encoding="utf-8")
  required = [
    "launch_docker_ws.sh",
    "launch_docker_hpc.sh",
    "_docker_launcher_common.sh",
    "sync_codebase.sh",
    "batch_convert_pyroki_to_gmr.sh",
    "docker_uv_sync_locked.sh",
    "MJPG_GPU_MODE=manual",
    "MJPG_GPU_MODE=gpus",
    "MJPG_GPUS=all",
    "--gpu-ids '[0,1]'",
    "--gpu-ids all",
    "inside the container, use `train` directly",
    "scripts/sync_codebase.sh push",
    "scripts/sync_codebase.sh pull-artifacts",
    "mjlab-playground:cuda128-dev",
    "/media/android/data/motion_datasets",
    "/data/share/motion_datasets",
    "MJPG_DATASET_READONLY=0",
    "MJPG_DATASET_ROOT",
    "MJLAB_PLAYGROUND_*",
    "gmr-astro",
    "pyroki-retargeted-astro",
  ]
  for needle in required:
    assert needle in text
