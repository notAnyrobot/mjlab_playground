#!/usr/bin/env sh

set -eu

: "${MJLAB_PLAYGROUND_REPO_ROOT:?MJLAB_PLAYGROUND_REPO_ROOT is required}"
: "${MJLAB_PLAYGROUND_RUNTIME:?MJLAB_PLAYGROUND_RUNTIME is required}"

IMAGE="${MJLAB_PLAYGROUND_IMAGE:-mjlab-playground:cuda128-dev}"
CONTAINER_WORKDIR="${MJLAB_PLAYGROUND_CONTAINER_WORKDIR:-/workspace/mjlab_playground}"
GPUS="${MJLAB_PLAYGROUND_GPUS:-all}"
PORT="${MJLAB_PLAYGROUND_PORT:-8080}"
HOST_HOME="${MJLAB_PLAYGROUND_CONTAINER_HOME:-$MJLAB_PLAYGROUND_REPO_ROOT/.container_home}"
HOST_CACHE="${MJLAB_PLAYGROUND_CONTAINER_CACHE:-$MJLAB_PLAYGROUND_REPO_ROOT/.container_cache}"
HOST_USER="${USER:-mjlab}"

if [ ! -f "$MJLAB_PLAYGROUND_REPO_ROOT/pyproject.toml" ]; then
  printf '%s\n' "error: expected pyproject.toml under $MJLAB_PLAYGROUND_REPO_ROOT" >&2
  exit 2
fi

if [ -t 0 ] && [ -t 1 ]; then
  INTERACTIVE="-it"
else
  INTERACTIVE=""
fi

GPU_ARGS=""
if [ "$GPUS" != "none" ]; then
  GPU_ARGS="--gpus $GPUS"
fi

print_config() {
  cat <<EOF
image:              $IMAGE
repo_root:          $MJLAB_PLAYGROUND_REPO_ROOT
container_workdir:  $CONTAINER_WORKDIR
runtime:            $MJLAB_PLAYGROUND_RUNTIME
gpus:               $GPUS
published_port:     $PORT -> 8080
host_logs:          $MJLAB_PLAYGROUND_REPO_ROOT/logs
host_wandb:         $MJLAB_PLAYGROUND_REPO_ROOT/wandb
host_artifacts:     $MJLAB_PLAYGROUND_REPO_ROOT/artifacts
host_home:          $HOST_HOME
host_cache:         $HOST_CACHE
container_user:     root
post_exit_chown:    $(id -u):$(id -g)
pythonpath:         $CONTAINER_WORKDIR/src
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

  docker run \
    --rm \
    $INTERACTIVE \
    $GPU_ARGS \
    --workdir "$CONTAINER_WORKDIR" \
    --env HOME="$CONTAINER_WORKDIR/.container_home" \
    --env XDG_CACHE_HOME="$CONTAINER_WORKDIR/.container_cache" \
    --env MPLCONFIGDIR="$CONTAINER_WORKDIR/.container_cache/matplotlib" \
    --env MUJOCO_GL=egl \
    --env PYTHONPATH="$CONTAINER_WORKDIR/src" \
    --env UV_NO_SYNC=1 \
    --env UV_PROJECT_ENVIRONMENT=/app/.venv \
    --env VIRTUAL_ENV=/app/.venv \
    --env WANDB_DIR="$CONTAINER_WORKDIR/wandb" \
    --env USER="$HOST_USER" \
    --env MJLAB_PLAYGROUND_HOST_UID="$(id -u)" \
    --env MJLAB_PLAYGROUND_HOST_GID="$(id -g)" \
    --mount type=bind,source="$MJLAB_PLAYGROUND_REPO_ROOT",target="$CONTAINER_WORKDIR" \
    --mount type=bind,source="$HOST_HOME",target="$CONTAINER_WORKDIR/.container_home" \
    --mount type=bind,source="$HOST_CACHE",target="$CONTAINER_WORKDIR/.container_cache" \
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
