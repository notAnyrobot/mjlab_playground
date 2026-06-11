#!/usr/bin/env sh

set -eu

if [ -n "${PIP_INDEX_URL:-}" ]; then
  uv lock --check
  export UV_DEFAULT_INDEX="$PIP_INDEX_URL"
  exec uv sync --frozen "$@"
fi

exec uv sync --locked "$@"
