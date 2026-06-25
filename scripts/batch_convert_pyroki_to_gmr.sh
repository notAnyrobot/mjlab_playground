#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
CONVERTER="${REPO_ROOT}/src/mjlab_playground/motion_lib/scripts/convert_pyroki_to_gmr.py"

DEFAULT_HPC_ASTRO_ROOT="/data/share/motion_datasets/protomotions/astro"
DEFAULT_WS_ASTRO_ROOT="/media/android/data/motion_datasets/protomotions/astro"
SOURCE_DIR_NAME="pyroki-retargeted-astro"

ASTRO_ROOT=""
FPS=""
FORCE_REMAKE=0
SPLITS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/batch_convert_pyroki_to_gmr.sh --fps FPS [options]

Convert Astro PyRoki retargeted .npz motions for all direct AMASS split
directories into sibling GMR directories:

  <astro-root>/<split>/pyroki-retargeted-astro -> <astro-root>/<split>/gmr-astro

Options:
  --fps FPS              Required. FPS to write into every GMR pickle.
  --astro-root PATH      Astro dataset root containing split directories.
                         Defaults to the first existing root:
                           /data/share/motion_datasets/protomotions/astro
                           /media/android/data/motion_datasets/protomotions/astro
  --split NAME           Convert one split. Repeat to convert multiple explicit
                         splits. If omitted, converts all direct child splits
                         containing pyroki-retargeted-astro.
  --force-remake         Overwrite existing GMR .pkl files.
  -h, --help             Show this help.

Environment:
  PYTHON                 Optional Python interpreter override.
EOF
}

choose_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "${PYTHON}"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "error: neither python nor python3 is available on PATH" >&2
  exit 1
}

resolve_astro_root() {
  if [[ -n "${ASTRO_ROOT}" ]]; then
    echo "${ASTRO_ROOT}"
    return
  fi
  if [[ -d "${DEFAULT_HPC_ASTRO_ROOT}" ]]; then
    echo "${DEFAULT_HPC_ASTRO_ROOT}"
    return
  fi
  if [[ -d "${DEFAULT_WS_ASTRO_ROOT}" ]]; then
    echo "${DEFAULT_WS_ASTRO_ROOT}"
    return
  fi
  echo "error: could not find an Astro dataset root; pass --astro-root" >&2
  echo "checked: ${DEFAULT_HPC_ASTRO_ROOT}" >&2
  echo "checked: ${DEFAULT_WS_ASTRO_ROOT}" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fps)
      [[ $# -ge 2 ]] || { echo "error: --fps requires a value" >&2; exit 1; }
      FPS="$2"
      shift 2
      ;;
    --astro-root)
      [[ $# -ge 2 ]] || { echo "error: --astro-root requires a value" >&2; exit 1; }
      ASTRO_ROOT="$2"
      shift 2
      ;;
    --split)
      [[ $# -ge 2 ]] || { echo "error: --split requires a value" >&2; exit 1; }
      SPLITS+=("$2")
      shift 2
      ;;
    --force-remake)
      FORCE_REMAKE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${FPS}" ]]; then
  echo "error: --fps is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${CONVERTER}" ]]; then
  echo "error: converter not found: ${CONVERTER}" >&2
  exit 1
fi

PYTHON_BIN=$(choose_python)
RESOLVED_ASTRO_ROOT=$(resolve_astro_root)

if [[ ! -d "${RESOLVED_ASTRO_ROOT}" ]]; then
  echo "error: Astro dataset root does not exist: ${RESOLVED_ASTRO_ROOT}" >&2
  exit 1
fi

INPUT_DIRS=()
if [[ ${#SPLITS[@]} -gt 0 ]]; then
  for split_name in "${SPLITS[@]}"; do
    input_dir="${RESOLVED_ASTRO_ROOT}/${split_name}/${SOURCE_DIR_NAME}"
    if [[ ! -d "${input_dir}" ]]; then
      echo "error: missing PyRoki split input directory: ${input_dir}" >&2
      exit 1
    fi
    INPUT_DIRS+=("${input_dir}")
  done
else
  for split_dir in "${RESOLVED_ASTRO_ROOT}"/*; do
    [[ -d "${split_dir}" ]] || continue
    input_dir="${split_dir}/${SOURCE_DIR_NAME}"
    [[ -d "${input_dir}" ]] || continue
    INPUT_DIRS+=("${input_dir}")
  done
fi

if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
  echo "error: no direct child split directories containing ${SOURCE_DIR_NAME} found under ${RESOLVED_ASTRO_ROOT}" >&2
  exit 1
fi

common_args=(--fps "${FPS}")
if [[ ${FORCE_REMAKE} -eq 1 ]]; then
  common_args+=(--force-remake)
fi

converted_splits=0
for input_dir in "${INPUT_DIRS[@]}"; do
  split_name=$(basename -- "$(dirname -- "${input_dir}")")
  echo "==> ${split_name}: ${input_dir}"
  "${PYTHON_BIN}" "${CONVERTER}" --input "${input_dir}" "${common_args[@]}"
  converted_splits=$((converted_splits + 1))
done

echo "Converted ${converted_splits} split(s) under ${RESOLVED_ASTRO_ROOT}."
