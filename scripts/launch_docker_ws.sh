#!/usr/bin/env sh
#
# Launch mjlab_playground in Docker on a local workstation.
#
# The launcher mounts the live checkout at /workspace/mjlab_playground while
# using the baked virtualenv from /app/.venv inside mjlab-playground:cuda128-dev.

set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd -P)
MJLAB_PLAYGROUND_REPO_ROOT=$(CDPATH= cd "$SCRIPT_DIR/.." && pwd -P)
MJLAB_PLAYGROUND_RUNTIME=workstation

export MJLAB_PLAYGROUND_REPO_ROOT
export MJLAB_PLAYGROUND_RUNTIME

. "$SCRIPT_DIR/_docker_launcher_common.sh"

mjlab_playground_main "$@"
