"""Tests for neutral Docker launcher argument construction."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MJLAB_ROOT = ROOT.parent / "mjlab"
CONTAINER_MJLAB_ROOT = "/workspace/mujocolab/mjlab"
CONTAINER_PLAYGROUND_ROOT = "/workspace/mujocolab/mjlab_playground"
PYTHONPATH_VALUE = f"{CONTAINER_MJLAB_ROOT}/src:{CONTAINER_PLAYGROUND_ROOT}/src"
WS_DATASET_ROOT = "/media/android/data/motion_datasets"
HPC_DATASET_ROOT = "/data/share/motion_datasets"
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


def make_manual_gpu_env(tmp_path: Path, *, gpus: str = "0") -> dict[str, str]:
  dev_dir = tmp_path / "dev"
  for name in [
    "nvidia0",
    "nvidia1",
    "nvidiactl",
    "nvidia-uvm",
    "nvidia-uvm-tools",
  ]:
    (dev_dir / name).parent.mkdir(parents=True, exist_ok=True)
    (dev_dir / name).touch()

  libs_dir = tmp_path / "nvidia-libs"
  for name in [
    "libcuda.so.1",
    "libnvidia-ml.so.1",
    "libnvidia-ptxjitcompiler.so.1",
    "libc.so.6",
  ]:
    (libs_dir / name).parent.mkdir(parents=True, exist_ok=True)
    (libs_dir / name).touch()

  nvidia_smi = tmp_path / "nvidia-smi"
  nvidia_smi.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
  nvidia_smi.chmod(nvidia_smi.stat().st_mode | stat.S_IXUSR)

  return {
    "MJLAB_PLAYGROUND_GPUS": gpus,
    "MJLAB_PLAYGROUND_HPC_DEV_DIR": str(dev_dir),
    "MJLAB_PLAYGROUND_HPC_NVIDIA_LIBS": str(libs_dir),
    "MJLAB_PLAYGROUND_HPC_NVIDIA_SMI": str(nvidia_smi),
  }


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
  assert f"mjlab_root:         {MJLAB_ROOT}" in result.stdout
  assert f"container_workdir:  {CONTAINER_PLAYGROUND_ROOT}" in result.stdout
  assert f"mjlab_container:    {CONTAINER_MJLAB_ROOT}" in result.stdout
  assert f"dataset_root:       {WS_DATASET_ROOT}" in result.stdout
  assert "dataset_readonly:   1" in result.stdout
  assert f"pythonpath:         {PYTHONPATH_VALUE}" in result.stdout
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
  assert f"mjlab_root:         {MJLAB_ROOT}" in result.stdout
  assert f"container_workdir:  {CONTAINER_PLAYGROUND_ROOT}" in result.stdout
  assert f"mjlab_container:    {CONTAINER_MJLAB_ROOT}" in result.stdout
  assert f"dataset_root:       {HPC_DATASET_ROOT}" in result.stdout
  assert "dataset_readonly:   1" in result.stdout
  assert "runtime:            hpc" in result.stdout
  assert "gpu_mode:           manual" in result.stdout


def test_print_config_accepts_mjlab_repo_root_override(tmp_path: Path) -> None:
  mjlab_root = tmp_path / "custom-mjlab"
  mjlab_root.mkdir()
  (mjlab_root / "pyproject.toml").write_text(
    "[project]\nname = \"mjlab\"\nversion = \"0.0.0\"\n",
    encoding="utf-8",
  )
  env = os.environ.copy()
  env["MJLAB_REPO_ROOT"] = str(mjlab_root)

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
  assert f"mjlab_root:         {mjlab_root}" in result.stdout


def test_print_config_accepts_short_mjpg_aliases(tmp_path: Path) -> None:
  mjlab_root = tmp_path / "custom-mjlab"
  mjlab_root.mkdir()
  (mjlab_root / "pyproject.toml").write_text(
    "[project]\nname = \"mjlab\"\nversion = \"0.0.0\"\n",
    encoding="utf-8",
  )
  dataset_root = tmp_path / "motion_datasets"
  env = os.environ.copy()
  env.update(
    {
      "MJPG_IMAGE": "custom-image:latest",
      "MJPG_MJLAB_REPO_ROOT": str(mjlab_root),
      "MJPG_PORT": "18080",
      "MJPG_GPU_MODE": "none",
      "MJPG_GPUS": "none",
      "MJPG_DATASET_ROOT": str(dataset_root),
      "MJPG_DATASET_READONLY": "0",
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
  assert "image:              custom-image:latest" in result.stdout
  assert f"mjlab_root:         {mjlab_root}" in result.stdout
  assert f"dataset_root:       {dataset_root}" in result.stdout
  assert "dataset_readonly:   0" in result.stdout
  assert "gpu_mode:           none" in result.stdout
  assert "gpus:               none" in result.stdout
  assert "published_port:     18080 -> 8080" in result.stdout


def test_short_mjpg_aliases_take_precedence_over_long_names(tmp_path: Path) -> None:
  result = run_launcher(
    WS_LAUNCHER,
    ["run", "python", "-c", "print('ok')"],
    tmp_path,
    {
      "MJPG_GPUS": "none",
      "MJLAB_PLAYGROUND_GPUS": "device=0",
      "MJPG_PORT": "18080",
      "MJLAB_PLAYGROUND_PORT": "28080",
    },
  )
  assert result.returncode == 0, result.stderr
  call = docker_call_text(tmp_path)
  assert "--gpus" not in call
  assert "--publish\n18080:8080\n" in call
  assert "--publish\n28080:8080\n" not in call


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
  assert f"--workdir\n{CONTAINER_PLAYGROUND_ROOT}\n" in call
  assert "--env\nMUJOCO_GL=egl\n" in call
  assert "--env\nUV_NO_SYNC=1\n" in call
  assert "--env\nUV_PROJECT_ENVIRONMENT=/app/.venv\n" in call
  assert "--env\nVIRTUAL_ENV=/app/.venv\n" in call
  assert f"--env\nPYTHONPATH={PYTHONPATH_VALUE}\n" in call
  assert f"--mount\ntype=bind,source={MJLAB_ROOT},target={CONTAINER_MJLAB_ROOT}\n" in call
  assert (
    f"--mount\ntype=bind,source={ROOT},target={CONTAINER_PLAYGROUND_ROOT}\n" in call
  )
  assert (
    f"--mount\ntype=bind,source={WS_DATASET_ROOT},target={WS_DATASET_ROOT},readonly\n"
    in call
  )
  assert "--publish\n18080:8080\n" in call
  assert "mjlab-playground:cuda128-dev\n" in call
  assert "nvidia-smi" in call


def test_hpc_launcher_defaults_to_manual_gpu_passthrough(tmp_path: Path) -> None:
  result = run_launcher(
    HPC_LAUNCHER,
    ["nvidia-smi"],
    tmp_path,
    make_manual_gpu_env(tmp_path, gpus="0"),
  )
  assert result.returncode == 0, result.stderr
  call = docker_call_text(tmp_path)
  assert "--gpus" not in call
  assert f"--device={tmp_path}/dev/nvidia0\n" in call
  assert f"--device={tmp_path}/dev/nvidiactl\n" in call
  assert f"--device={tmp_path}/dev/nvidia-uvm\n" in call
  assert f"--device={tmp_path}/dev/nvidia-uvm-tools\n" in call
  for lib in ["libcuda.so.1", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1"]:
    assert (
      f"--mount\ntype=bind,source={tmp_path}/nvidia-libs/{lib},target=/usr/local/cuda/compat/{lib},readonly\n"
      in call
    )
  assert f"source={tmp_path}/nvidia-libs,target=/host-nvidia-libs" not in call
  assert "libc.so.6" not in call
  assert f"--mount\ntype=bind,source={tmp_path}/nvidia-smi,target=/usr/bin/nvidia-smi,readonly\n" in call
  assert "--env\nLD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64\n" in call
  assert "--env\nCUDA_VISIBLE_DEVICES=0\n" in call


def test_hpc_launcher_can_use_docker_gpus_when_requested(tmp_path: Path) -> None:
  result = run_launcher(
    HPC_LAUNCHER,
    ["nvidia-smi"],
    tmp_path,
    {
      "MJLAB_PLAYGROUND_GPU_MODE": "gpus",
      "MJLAB_PLAYGROUND_GPUS": "device=0",
    },
  )
  assert result.returncode == 0, result.stderr
  call = docker_call_text(tmp_path)
  assert "--gpus\ndevice=0\n" in call
  assert "--device=" not in call


def test_dataset_mount_can_be_overridden_and_made_writable(tmp_path: Path) -> None:
  dataset_root = tmp_path / "motion_datasets"
  result = run_launcher(
    WS_LAUNCHER,
    ["run", "python", "-c", "print('ok')"],
    tmp_path,
    {
      "MJLAB_PLAYGROUND_DATASET_ROOT": str(dataset_root),
      "MJLAB_PLAYGROUND_DATASET_READONLY": "0",
      "MJLAB_PLAYGROUND_GPUS": "none",
    },
  )
  assert result.returncode == 0, result.stderr
  call = docker_call_text(tmp_path)
  assert (
    f"--mount\ntype=bind,source={dataset_root},target={dataset_root}\n"
    in call
  )
  assert f"source={dataset_root},target={dataset_root},readonly" not in call


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
    f"--mount\ntype=bind,source={host_home},target={CONTAINER_PLAYGROUND_ROOT}/.container_home\n"
    in call
  )
  assert (
    f"--mount\ntype=bind,source={host_cache},target={CONTAINER_PLAYGROUND_ROOT}/.container_cache\n"
    in call
  )
  assert f"--env\nHOME={CONTAINER_PLAYGROUND_ROOT}/.container_home\n" in call
  assert f"--env\nXDG_CACHE_HOME={CONTAINER_PLAYGROUND_ROOT}/.container_cache\n" in call
  assert (
    f"--env\nMPLCONFIGDIR={CONTAINER_PLAYGROUND_ROOT}/.container_cache/matplotlib\n"
    in call
  )


def test_run_passes_arbitrary_command_without_task_specific_mode(
  tmp_path: Path,
) -> None:
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


def test_help_does_not_start_docker(tmp_path: Path) -> None:
  result = run_launcher(HPC_LAUNCHER, ["help"], tmp_path)
  assert result.returncode == 0, result.stderr
  assert "Usage:" in result.stdout
  assert "MJPG_GPU_MODE" in result.stdout
  assert "MJLAB_PLAYGROUND_* names remain supported" in result.stdout
  assert not (tmp_path / "docker-args.txt").exists()


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
