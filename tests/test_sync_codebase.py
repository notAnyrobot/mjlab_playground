"""Static and fake-rsync checks for the HPC sync helper."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = ROOT / "scripts" / "sync_codebase.sh"
GITIGNORE = ROOT / ".gitignore"


def _make_fake_rsync(tmp_path: Path) -> Path:
  log_path = tmp_path / "rsync-args.txt"
  fake_rsync = tmp_path / "rsync"
  fake_rsync.write_text(
    """#!/usr/bin/env sh
set -eu
: "${RSYNC_CALL_LOG:?RSYNC_CALL_LOG is required}"
printf '%s\\n' "$@" >> "$RSYNC_CALL_LOG"
exit 0
""",
    encoding="utf-8",
  )
  fake_rsync.chmod(fake_rsync.stat().st_mode | stat.S_IXUSR)
  return log_path


def _run_sync(command: str, tmp_path: Path) -> str:
  log_path = _make_fake_rsync(tmp_path)
  env = os.environ.copy()
  env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
  env["RSYNC_CALL_LOG"] = str(log_path)

  result = subprocess.run(
    ["bash", str(SYNC_SCRIPT), command],
    cwd=ROOT,
    env=env,
    check=False,
    text=True,
    capture_output=True,
  )

  assert result.returncode == 0, result.stderr
  return log_path.read_text(encoding="utf-8")


def _gitignore_entries() -> set[str]:
  return {
    line.strip()
    for line in GITIGNORE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
  }


def test_push_deletes_stale_remote_source_without_deleting_artifacts(
  tmp_path: Path,
) -> None:
  args = _run_sync("push", tmp_path)

  assert "atom7@172.16.9.7:/data/atom7/Code/mujocolab/mjlab_playground/" in args
  assert "--delete" in args
  assert "--delete-excluded" not in args
  for excluded in ("data/", "logs/", "wandb/", "artifacts/"):
    assert excluded in args


def test_remote_workspace_root_is_the_shared_mujocolab_directory() -> None:
  text = SYNC_SCRIPT.read_text(encoding="utf-8")

  assert 'REMOTE_ROOT="/data/atom7/Code/mujocolab"' in text
  assert 'REMOTE_REPO="mjlab_playground"' in text


def test_pull_artifacts_fetches_logs_wandb_and_artifacts(tmp_path: Path) -> None:
  args = _run_sync("pull-artifacts", tmp_path)

  assert "mjlab_playground/logs/" in args
  assert "mjlab_playground/wandb/" in args
  assert "mjlab_playground/artifacts/" in args


def test_gitignore_covers_local_runtime_and_sync_payloads() -> None:
  entries = _gitignore_entries()
  expected = {
    ".agents/",
    ".codex/",
    ".pytest_cache/",
    ".container_home/",
    ".container_cache/",
    "logs/",
    "wandb/",
    "artifacts/",
    "data/",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.onnx",
    "*.tar",
  }
  assert expected <= entries
