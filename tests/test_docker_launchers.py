"""Tests for neutral Docker launcher argument construction."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS_LAUNCHER = ROOT / "scripts" / "launch_docker_ws.sh"
HPC_LAUNCHER = ROOT / "scripts" / "launch_docker_hpc.sh"
COMMON_LAUNCHER = ROOT / "scripts" / "_docker_launcher_common.sh"


def make_fake_docker(tmp_path: Path) -> tuple[Path, Path]:
  log_path = tmp_path / "docker-args.txt"
  fake_docker = tmp_path / "docker"
  fake_docker.write_text(
    """#!/usr/bin/env sh
set -eu
: "${DOCKER_CALL_LOG:?DOCKER_CALL_LOG is required}"
printf '%s\\n' "$@" > "$DOCKER_CALL_LOG"
exit 0
""",
    encoding="utf-8",
  )
  fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR)
  return fake_docker, log_path


def run_launcher(
  launcher: Path,
  args: list[str],
  tmp_path: Path,
  extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
  fake_docker, log_path = make_fake_docker(tmp_path)
  env = os.environ.copy()
  env.update(
    {
      "PATH": f"{fake_docker.parent}{os.pathsep}{env['PATH']}",
      "DOCKER_CALL_LOG": str(log_path),
    }
  )
  if extra_env:
    env.update(extra_env)
  return subprocess.run(
    [str(launcher), *args],
    cwd=ROOT,
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
  )


def docker_call_text(tmp_path: Path) -> str:
  return (tmp_path / "docker-args.txt").read_text(encoding="utf-8")


def test_bash_mode_forwards_arguments_after_wrapper_separator(tmp_path: Path) -> None:
  result = run_launcher(
    WS_LAUNCHER,
    ["bash", "-lc", "echo ok"],
    tmp_path,
  )
  assert result.returncode == 0, result.stderr
  assert "\n--\nbash\n-lc\necho ok\n" in docker_call_text(tmp_path)


def test_ws_print_config_uses_playground_defaults() -> None:
  result = subprocess.run(
    [str(WS_LAUNCHER), "print-config"],
    cwd=ROOT,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
  )
  assert result.returncode == 0, result.stderr
  assert "image:              mjlab-playground:cuda128-dev" in result.stdout
  assert f"repo_root:          {ROOT}" in result.stdout
  assert "container_workdir:  /workspace/mjlab_playground" in result.stdout
  assert "gpus:               all" in result.stdout
  assert "published_port:     8080 -> 8080" in result.stdout
  assert "runtime:            workstation" in result.stdout


def test_print_config_does_not_create_home_or_cache_override_dirs(
  tmp_path: Path,
) -> None:
  host_home = tmp_path / "host-home"
  host_cache = tmp_path / "host-cache"
  env = os.environ.copy()
  env.update(
    {
      "MJLAB_PLAYGROUND_CONTAINER_HOME": str(host_home),
      "MJLAB_PLAYGROUND_CONTAINER_CACHE": str(host_cache),
    }
  )

  result = subprocess.run(
    [str(WS_LAUNCHER), "print-config"],
    cwd=ROOT,
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
  )

  assert result.returncode == 0, result.stderr
  assert not host_home.exists()
  assert not host_cache.exists()


def test_hpc_print_config_uses_hpc_runtime_label() -> None:
  result = subprocess.run(
    [str(HPC_LAUNCHER), "print-config"],
    cwd=ROOT,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
  )
  assert result.returncode == 0, result.stderr
  assert "image:              mjlab-playground:cuda128-dev" in result.stdout
  assert "container_workdir:  /workspace/mjlab_playground" in result.stdout
  assert "runtime:            hpc" in result.stdout


def test_nvidia_smi_invokes_docker_with_mount_env_and_gpu_args(tmp_path: Path) -> None:
  result = run_launcher(
    WS_LAUNCHER,
    ["nvidia-smi"],
    tmp_path,
    {
      "MJLAB_PLAYGROUND_GPUS": "device=0",
      "MJLAB_PLAYGROUND_PORT": "18080",
    },
  )
  assert result.returncode == 0, result.stderr
  call = docker_call_text(tmp_path)
  assert "run\n" in call
  assert "--rm\n" in call
  assert "--gpus\ndevice=0\n" in call
  assert "--workdir\n/workspace/mjlab_playground\n" in call
  assert "--env\nMUJOCO_GL=egl\n" in call
  assert "--env\nUV_NO_SYNC=1\n" in call
  assert "--env\nUV_PROJECT_ENVIRONMENT=/app/.venv\n" in call
  assert "--env\nVIRTUAL_ENV=/app/.venv\n" in call
  assert "--env\nPYTHONPATH=/workspace/mjlab_playground/src\n" in call
  assert (
    f"--mount\ntype=bind,source={ROOT},target=/workspace/mjlab_playground\n" in call
  )
  assert "--publish\n18080:8080\n" in call
  assert "mjlab-playground:cuda128-dev\n" in call
  assert "nvidia-smi" in call


def test_home_and_cache_overrides_mount_to_container_runtime_dirs(
  tmp_path: Path,
) -> None:
  host_home = tmp_path / "host-home"
  host_cache = tmp_path / "host-cache"
  result = run_launcher(
    WS_LAUNCHER,
    ["run", "python", "-c", "print('ok')"],
    tmp_path,
    {
      "MJLAB_PLAYGROUND_CONTAINER_HOME": str(host_home),
      "MJLAB_PLAYGROUND_CONTAINER_CACHE": str(host_cache),
    },
  )
  assert result.returncode == 0, result.stderr
  call = docker_call_text(tmp_path)
  assert (
    f"--mount\ntype=bind,source={host_home},target=/workspace/mjlab_playground/.container_home\n"
    in call
  )
  assert (
    f"--mount\ntype=bind,source={host_cache},target=/workspace/mjlab_playground/.container_cache\n"
    in call
  )
  assert "--env\nHOME=/workspace/mjlab_playground/.container_home\n" in call
  assert "--env\nXDG_CACHE_HOME=/workspace/mjlab_playground/.container_cache\n" in call
  assert (
    "--env\nMPLCONFIGDIR=/workspace/mjlab_playground/.container_cache/matplotlib\n"
    in call
  )


def test_run_passes_arbitrary_command_without_task_specific_mode(tmp_path: Path) -> None:
  result = run_launcher(
    WS_LAUNCHER,
    ["run", "list-envs", "--keyword", "Getup"],
    tmp_path,
  )
  assert result.returncode == 0, result.stderr
  call = docker_call_text(tmp_path)
  assert "list-envs" in call
  assert "--keyword" in call
  assert "Getup" in call
  assert "getup-smoke" not in call
  assert "train-smoke" not in call


def test_gpu_none_omits_docker_gpus_argument(tmp_path: Path) -> None:
  result = run_launcher(
    WS_LAUNCHER,
    ["run", "python", "-c", "print('ok')"],
    tmp_path,
    {"MJLAB_PLAYGROUND_GPUS": "none"},
  )
  assert result.returncode == 0, result.stderr
  call = docker_call_text(tmp_path)
  assert "--gpus" not in call
  assert "python" in call


def test_launchers_do_not_define_task_specific_modes() -> None:
  combined = (
    WS_LAUNCHER.read_text(encoding="utf-8")
    + "\n"
    + HPC_LAUNCHER.read_text(encoding="utf-8")
    + "\n"
    + COMMON_LAUNCHER.read_text(encoding="utf-8")
  )
  assert "getup-smoke" not in combined
  assert "train-smoke" not in combined
  assert "pytest)" not in combined


def test_gitignore_does_not_ignore_scripts_directory() -> None:
  ignored = {
    line.strip()
    for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
  }
  assert "scripts/" not in ignored
  assert "scripts" not in ignored
