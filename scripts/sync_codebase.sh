#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="atom7@172.16.9.7"
REMOTE_ROOT="/data/atom7/Code/mujocolab"
REMOTE_REPO="mjlab_playground"
REMOTE="${REMOTE_HOST}:${REMOTE_ROOT}/${REMOTE_REPO}"

# Always run from the repo root so rsync paths are correct.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but not installed" >&2
  exit 1
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [COMMAND]

Commands:
  push            Sync mjlab_playground source to remote HPC  [default]
  push-data       Sync data/ directory to remote HPC (bypasses .gitignore)
  pull-artifacts  Download logs/, wandb/, and artifacts/ from remote HPC
  pull-logs       Alias for pull-artifacts
  help            Show this message

Notes:
  This script only syncs mjlab_playground. Manage the sibling mjlab checkout
  separately.

  Remote workspace root: ${REMOTE_ROOT}
  Remote repo path:       ${REMOTE_ROOT}/${REMOTE_REPO}

  The push command uses rsync --delete so files deleted locally are also removed
  from the remote source tree. Excluded runtime outputs are protected because
  this script does not use --delete-excluded.

Examples:
  $(basename "$0")
  $(basename "$0") push
  $(basename "$0") push-data
  $(basename "$0") pull-artifacts
EOF
}

# Default push: respects .gitignore and deletes stale remote source files.
cmd_push() {
  echo "==> Syncing codebase to ${REMOTE} (respecting .gitignore) ..."
  rsync -avz --progress \
    --delete \
    --filter=':- .gitignore' \
    --exclude 'data/' \
    --exclude 'logs/' \
    --exclude 'wandb/' \
    --exclude 'artifacts/' \
    --exclude '.git/' \
    "$ROOT_DIR/" "$REMOTE/"
  echo "==> Push complete."
}

# Push data only, bypassing .gitignore
cmd_push_data() {
  echo "==> Syncing data/ to ${REMOTE}/data/ ..."
  rsync -avz --progress \
    "$ROOT_DIR/data/" "$REMOTE/data/"
  echo "==> Push data complete."
}

# Pull runtime outputs from HPC to local.
cmd_pull_artifacts() {
  for dirname in logs wandb artifacts; do
    LOCAL_DIR="${ROOT_DIR}/${dirname}"
    mkdir -p "$LOCAL_DIR"
    echo "==> Downloading ${dirname}/ from ${REMOTE}/${dirname}/ to ${LOCAL_DIR}/ ..."
    rsync -avz --progress --ignore-missing-args \
      "${REMOTE}/${dirname}/" "$LOCAL_DIR/"
  done
  echo "==> Pull artifacts complete."
}

COMMAND="${1:-push}"

case "$COMMAND" in
  push)           cmd_push ;;
  push-data)      cmd_push_data ;;
  pull-artifacts|pull-logs)
                  cmd_pull_artifacts ;;
  help|--help|-h) usage ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage >&2
    exit 1
    ;;
esac
