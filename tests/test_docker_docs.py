"""Documentation checks for the CUDA 12.8 Docker workflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "docker_cuda128_workflow.md"
README = ROOT / "README.md"


def test_readme_links_docker_workflow() -> None:
  text = README.read_text(encoding="utf-8")
  assert "docs/docker_cuda128_workflow.md" in text
  assert "mjlab-playground:cuda128-dev" in text


def test_docker_workflow_documents_core_commands() -> None:
  text = DOC.read_text(encoding="utf-8")
  required = [
    "docker build --pull=false -t mjlab-playground:cuda128-dev .",
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
  ]
  for needle in required:
    assert needle in text


def test_docker_workflow_keeps_launcher_generic() -> None:
  text = DOC.read_text(encoding="utf-8")
  assert "getup-smoke" not in text
  assert "train-smoke" not in text
  assert "./scripts/launch_docker_ws.sh run train" in text
