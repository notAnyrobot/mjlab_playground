#!/usr/bin/env sh

set -eu

: "${MJLAB_PLAYGROUND_REPO_ROOT:?MJLAB_PLAYGROUND_REPO_ROOT is required}"
: "${MJLAB_PLAYGROUND_RUNTIME:?MJLAB_PLAYGROUND_RUNTIME is required}"

IMAGE="${MJPG_IMAGE:-${MJLAB_PLAYGROUND_IMAGE:-mjlab-playground:cuda128-dev}}"
CONTAINER_MJLAB_ROOT="${MJLAB_CONTAINER_ROOT:-/workspace/mujocolab/mjlab}"
CONTAINER_WORKDIR="${MJPG_CONTAINER_WORKDIR:-${MJLAB_PLAYGROUND_CONTAINER_WORKDIR:-/workspace/mujocolab/mjlab_playground}}"
GPUS="${MJPG_GPUS:-${MJLAB_PLAYGROUND_GPUS:-all}}"
if [ -n "${MJPG_GPU_MODE:-}" ]; then
  GPU_MODE="$MJPG_GPU_MODE"
elif [ -n "${MJLAB_PLAYGROUND_GPU_MODE:-}" ]; then
  GPU_MODE="$MJLAB_PLAYGROUND_GPU_MODE"
elif [ "$MJLAB_PLAYGROUND_RUNTIME" = "hpc" ]; then
  GPU_MODE="manual"
else
  GPU_MODE="gpus"
fi
PORT="${MJPG_PORT:-${MJLAB_PLAYGROUND_PORT:-8080}}"
HOST_HOME="${MJPG_CONTAINER_HOME:-${MJLAB_PLAYGROUND_CONTAINER_HOME:-$MJLAB_PLAYGROUND_REPO_ROOT/.container_home}}"
HOST_CACHE="${MJPG_CONTAINER_CACHE:-${MJLAB_PLAYGROUND_CONTAINER_CACHE:-$MJLAB_PLAYGROUND_REPO_ROOT/.container_cache}}"
HOST_USER="${USER:-mjlab}"
PYTHONPATH_VALUE="$CONTAINER_MJLAB_ROOT/src:$CONTAINER_WORKDIR/src"
if [ -n "${MJPG_DATASET_ROOT+x}" ]; then
  DATASET_ROOT="$MJPG_DATASET_ROOT"
elif [ -n "${MJLAB_PLAYGROUND_DATASET_ROOT+x}" ]; then
  DATASET_ROOT="$MJLAB_PLAYGROUND_DATASET_ROOT"
elif [ "$MJLAB_PLAYGROUND_RUNTIME" = "hpc" ]; then
  DATASET_ROOT="/data/share/motion_datasets"
else
  DATASET_ROOT="/media/android/data/motion_datasets"
fi
DATASET_READONLY="${MJPG_DATASET_READONLY:-${MJLAB_PLAYGROUND_DATASET_READONLY:-1}}"
HPC_DEV_DIR="${MJPG_HPC_DEV_DIR:-${MJLAB_PLAYGROUND_HPC_DEV_DIR:-/dev}}"
HPC_NVIDIA_LIBS="${MJPG_HPC_NVIDIA_LIBS:-${MJLAB_PLAYGROUND_HPC_NVIDIA_LIBS:-}}"
HPC_NVIDIA_SMI="${MJPG_HPC_NVIDIA_SMI:-${MJLAB_PLAYGROUND_HPC_NVIDIA_SMI:-/usr/bin/nvidia-smi}}"

if [ -n "${MJPG_MJLAB_REPO_ROOT:-}" ]; then
  MJLAB_REPO_ROOT=$(CDPATH= cd "$MJPG_MJLAB_REPO_ROOT" && pwd -P)
elif [ -z "${MJLAB_REPO_ROOT:-}" ]; then
  MJLAB_REPO_ROOT=$(CDPATH= cd "$MJLAB_PLAYGROUND_REPO_ROOT/../mjlab" && pwd -P)
else
  MJLAB_REPO_ROOT=$(CDPATH= cd "$MJLAB_REPO_ROOT" && pwd -P)
fi

if [ ! -f "$MJLAB_PLAYGROUND_REPO_ROOT/pyproject.toml" ]; then
  printf '%s\n' "error: expected pyproject.toml under $MJLAB_PLAYGROUND_REPO_ROOT" >&2
  exit 2
fi

if [ ! -f "$MJLAB_REPO_ROOT/pyproject.toml" ]; then
  printf '%s\n' "error: expected pyproject.toml under $MJLAB_REPO_ROOT" >&2
  exit 2
fi

if [ -t 0 ] && [ -t 1 ]; then
  INTERACTIVE="-it"
else
  INTERACTIVE=""
fi

GPU_ARGS=""
GPU_ENV_ARGS=""
GPU_MOUNT_ARGS=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [shell|bash|nvidia-smi|python|run|print-config|help] [ARGS...]

Commands:
  shell         Open an interactive container shell. This is the default.
  bash          Run bash with forwarded arguments.
  nvidia-smi    Run nvidia-smi inside the container.
  python        Run python inside the container.
  run           Run an arbitrary command inside the container.
  print-config  Print resolved launcher configuration without starting Docker.
  help          Show this message.

Common overrides:
  MJPG_IMAGE                       Container image tag.
  MJPG_MJLAB_REPO_ROOT             Host sibling mjlab checkout.
  MJPG_PORT                        Host port published to container port 8080.
  MJPG_GPU_MODE                    gpus, manual, or none.
  MJPG_GPUS                        all, none, device=N, or comma-separated GPU IDs.
  MJPG_HPC_DEV_DIR                 Device directory for manual HPC GPU mode.
  MJPG_HPC_NVIDIA_LIBS             Optional host directory for NVIDIA driver libraries.
  MJPG_HPC_NVIDIA_SMI              Host nvidia-smi path for manual mode.
  MJPG_DATASET_ROOT                Host dataset root mounted at the same absolute path.
  MJPG_DATASET_READONLY            1 for read-only dataset mount, 0 for writable.

Long MJLAB_PLAYGROUND_* names remain supported for compatibility. MJPG_* wins
when both forms are set.
EOF
}

fail() {
  printf '%s\n' "error: $*" >&2
  exit 2
}

normalize_gpu_selection() {
  case "$GPUS" in
    device=*)
      printf '%s\n' "${GPUS#device=}"
      ;;
    *)
      printf '%s\n' "$GPUS"
      ;;
  esac
}

validate_manual_gpu_file() {
  if [ ! -e "$1" ]; then
    fail "required GPU file does not exist: $1"
  fi
}

resolve_manual_nvidia_lib() {
  lib="$1"

  if [ -n "$HPC_NVIDIA_LIBS" ]; then
    if [ ! -d "$HPC_NVIDIA_LIBS" ]; then
      fail "NVIDIA library dir does not exist: $HPC_NVIDIA_LIBS"
    fi

    if [ ! -e "$HPC_NVIDIA_LIBS/$lib" ]; then
      fail "missing NVIDIA library: $HPC_NVIDIA_LIBS/$lib"
    fi

    printf '%s\n' "$HPC_NVIDIA_LIBS/$lib"
    return 0
  fi

  path=$(ldconfig -p | awk -v lib="$lib" '$1 == lib {print $NF; exit}')
  if [ -z "$path" ] || [ ! -e "$path" ]; then
    fail "could not find host NVIDIA library with ldconfig: $lib"
  fi

  printf '%s\n' "$path"
}

validate_manual_nvidia_smi() {
  if [ ! -e "$HPC_NVIDIA_SMI" ]; then
    fail "nvidia-smi binary does not exist: $HPC_NVIDIA_SMI"
  fi
}

manual_gpu_ids() {
  selection=$(normalize_gpu_selection)

  case "$selection" in
    none)
      return 0
      ;;
    all)
      for path in "$HPC_DEV_DIR"/nvidia[0-9]*; do
        [ -e "$path" ] || continue
        name="${path##*/}"
        id="${name#nvidia}"
        case "$id" in
          ''|*[!0-9]*)
            ;;
          *)
            printf '%s\n' "$id"
            ;;
        esac
      done | sort -n
      ;;
    ''|*[!0-9,]*)
      fail "MJLAB_PLAYGROUND_GPUS must be all, none, device=N, or comma-separated GPU IDs in manual mode"
      ;;
    *)
      printf '%s\n' "$selection" | tr ',' '\n'
      ;;
  esac
}

manual_cuda_visible_devices() {
  selection=$(normalize_gpu_selection)

  case "$selection" in
    all|none)
      return 0
      ;;
  esac

  count=$(printf '%s\n' "$selection" | tr ',' '\n' | wc -l | tr -d ' ')
  index=0
  visible=""
  while [ "$index" -lt "$count" ]; do
    if [ -n "$visible" ]; then
      visible="$visible,$index"
    else
      visible="$index"
    fi
    index=$((index + 1))
  done
  printf '%s\n' "$visible"
}

build_gpu_args() {
  case "$GPU_MODE" in
    none)
      return 0
      ;;
    gpus)
      if [ "$GPUS" != "none" ]; then
        GPU_ARGS="--gpus $GPUS"
      fi
      ;;
    manual)
      selection=$(normalize_gpu_selection)
      if [ "$selection" = "none" ]; then
        return 0
      fi

      validate_manual_nvidia_smi
      ids=$(manual_gpu_ids)
      if [ -z "$ids" ]; then
        fail "no GPU devices found under $HPC_DEV_DIR"
      fi

      for id in $ids; do
        validate_manual_gpu_file "$HPC_DEV_DIR/nvidia$id"
        GPU_ARGS="$GPU_ARGS --device=$HPC_DEV_DIR/nvidia$id"
      done

      for device in nvidiactl nvidia-uvm nvidia-uvm-tools; do
        validate_manual_gpu_file "$HPC_DEV_DIR/$device"
        GPU_ARGS="$GPU_ARGS --device=$HPC_DEV_DIR/$device"
      done

      for lib in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
        lib_path=$(resolve_manual_nvidia_lib "$lib")
        GPU_MOUNT_ARGS="$GPU_MOUNT_ARGS --mount type=bind,source=$lib_path,target=/usr/local/cuda/compat/$lib,readonly"
      done

      GPU_MOUNT_ARGS="$GPU_MOUNT_ARGS --mount type=bind,source=$HPC_NVIDIA_SMI,target=/usr/bin/nvidia-smi,readonly"
      GPU_ENV_ARGS="--env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64"
      cuda_visible=$(manual_cuda_visible_devices)
      if [ -n "$cuda_visible" ]; then
        GPU_ENV_ARGS="$GPU_ENV_ARGS --env CUDA_VISIBLE_DEVICES=$cuda_visible"
      fi
      ;;
    *)
      fail "MJLAB_PLAYGROUND_GPU_MODE must be one of: gpus, manual, none"
      ;;
  esac
}

dataset_mount_arg() {
  if [ -z "$DATASET_ROOT" ]; then
    return 0
  fi

  mount_arg="type=bind,source=$DATASET_ROOT,target=$DATASET_ROOT"
  if [ "$DATASET_READONLY" != "0" ]; then
    mount_arg="$mount_arg,readonly"
  fi
  printf '%s\n' "$mount_arg"
}

print_config() {
  cat <<EOF
image:              $IMAGE
repo_root:          $MJLAB_PLAYGROUND_REPO_ROOT
mjlab_root:         $MJLAB_REPO_ROOT
container_workdir:  $CONTAINER_WORKDIR
mjlab_container:    $CONTAINER_MJLAB_ROOT
dataset_root:       ${DATASET_ROOT:-not mounted}
dataset_readonly:   $DATASET_READONLY
runtime:            $MJLAB_PLAYGROUND_RUNTIME
gpu_mode:           $GPU_MODE
gpus:               $GPUS
published_port:     $PORT -> 8080
host_logs:          $MJLAB_PLAYGROUND_REPO_ROOT/logs
host_wandb:         $MJLAB_PLAYGROUND_REPO_ROOT/wandb
host_artifacts:     $MJLAB_PLAYGROUND_REPO_ROOT/artifacts
host_home:          $HOST_HOME
host_cache:         $HOST_CACHE
container_user:     root
post_exit_chown:    $(id -u):$(id -g)
pythonpath:         $PYTHONPATH_VALUE
uv_environment:     /app/.venv
EOF
}

ensure_runtime_dirs() {
  mkdir -p \
    "$MJLAB_PLAYGROUND_REPO_ROOT/logs" \
    "$MJLAB_PLAYGROUND_REPO_ROOT/wandb" \
    "$MJLAB_PLAYGROUND_REPO_ROOT/artifacts" \
    "$HOST_HOME" \
    "$HOST_CACHE"
}

run_container() {
  ensure_runtime_dirs
  build_gpu_args

  docker run \
    --rm \
    $INTERACTIVE \
    $GPU_ARGS \
    --workdir "$CONTAINER_WORKDIR" \
    --env HOME="$CONTAINER_WORKDIR/.container_home" \
    --env XDG_CACHE_HOME="$CONTAINER_WORKDIR/.container_cache" \
    --env MPLCONFIGDIR="$CONTAINER_WORKDIR/.container_cache/matplotlib" \
    --env MUJOCO_GL=egl \
    --env PYTHONPATH="$PYTHONPATH_VALUE" \
    --env UV_NO_SYNC=1 \
    --env UV_PROJECT_ENVIRONMENT=/app/.venv \
    --env VIRTUAL_ENV=/app/.venv \
    $GPU_ENV_ARGS \
    --env WANDB_DIR="$CONTAINER_WORKDIR/wandb" \
    --env USER="$HOST_USER" \
    --env MJLAB_PLAYGROUND_HOST_UID="$(id -u)" \
    --env MJLAB_PLAYGROUND_HOST_GID="$(id -g)" \
    --mount type=bind,source="$MJLAB_REPO_ROOT",target="$CONTAINER_MJLAB_ROOT" \
    --mount type=bind,source="$MJLAB_PLAYGROUND_REPO_ROOT",target="$CONTAINER_WORKDIR" \
    --mount type=bind,source="$HOST_HOME",target="$CONTAINER_WORKDIR/.container_home" \
    --mount type=bind,source="$HOST_CACHE",target="$CONTAINER_WORKDIR/.container_cache" \
    ${DATASET_ROOT:+--mount "$(dataset_mount_arg)"} \
    $GPU_MOUNT_ARGS \
    --publish "$PORT:8080" \
    "$IMAGE" \
    "$@"
}

run_wrapped() {
  run_container /bin/bash -lc \
    'export PATH=/app/.venv/bin:$PATH
"$@"
status=$?
for path in logs wandb artifacts .container_home .container_cache; do
  if [ -e "$path" ]; then
    chown -R "$MJLAB_PLAYGROUND_HOST_UID:$MJLAB_PLAYGROUND_HOST_GID" "$path"
  fi
done
exit "$status"' -- "$@"
}

mjlab_playground_main() {
  cmd="${1:-shell}"
  if [ $# -gt 0 ]; then
    shift
  fi

  case "$cmd" in
    help|-h|--help)
      usage
      ;;
    shell)
      run_wrapped bash
      ;;
    bash)
      run_wrapped bash "$@"
      ;;
    nvidia-smi)
      run_wrapped nvidia-smi "$@"
      ;;
    python)
      run_wrapped python "$@"
      ;;
    run)
      run_wrapped "$@"
      ;;
    print-config)
      print_config
      ;;
    *)
      run_wrapped "$cmd" "$@"
      ;;
  esac
}
