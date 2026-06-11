#!/usr/bin/env sh
#
# Launch mjlab_playground in Docker on an HPC node.
#
# The script assumes mjlab-playground:cuda128-dev has already been loaded on the
# HPC host. It mounts sibling mjlab and mjlab_playground checkouts under
# /workspace/mujocolab so training logs and artifacts stay on the HPC
# filesystem.

set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd -P)
MJLAB_PLAYGROUND_REPO_ROOT=$(CDPATH= cd "$SCRIPT_DIR/.." && pwd -P)
MJLAB_PLAYGROUND_RUNTIME=hpc

export MJLAB_PLAYGROUND_REPO_ROOT
export MJLAB_PLAYGROUND_RUNTIME

. "$SCRIPT_DIR/_docker_launcher_common.sh"

mjlab_playground_main "$@"
