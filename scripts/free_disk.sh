#!/usr/bin/env bash
# Pre-flight disk cleanup helper.
#
# Default mode: dry-run — list candidates + sizes only.
# Pass --apply to actually delete.
#
# Candidates (highest-yield first):
#   1. attacks-repo runs/main/*/raw_logits/   (can free 20-30 GB)
#   2. adversarial-reasoning-attacks/.hf_cache if symlink-able to ~/.cache/huggingface
#   3. pip cache (~/.cache/pip)
#   4. old runs/*/ckpt/ (keeps best + latest by registry; manual cleanup for stale dirs)
#   5. gzip uncompressed trajectory logs runs/*/trajectory_*.jsonl

set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ATTACKS_ROOT="$(cd "${REPO_ROOT}/../adversarial-reasoning-attacks" 2>/dev/null && pwd || echo "")"

run() {
  if [[ "$APPLY" -eq 1 ]]; then
    echo "[apply] $*"
    eval "$*"
  else
    echo "[dry-run] $*"
  fi
}

section() {
  echo
  echo "=== $* ==="
}

section "disk status"
df -h /

if [[ -n "$ATTACKS_ROOT" ]]; then
  section "attacks-repo raw_logits (candidate 1)"
  LOGITS_DIRS=$(find "$ATTACKS_ROOT/runs" -type d -name "raw_logits" 2>/dev/null || true)
  if [[ -n "$LOGITS_DIRS" ]]; then
    du -sh $LOGITS_DIRS 2>/dev/null || true
    for d in $LOGITS_DIRS; do
      run "rm -rf '$d'"
    done
  else
    echo "no raw_logits/ directories found"
  fi
fi

section "pip cache (candidate 3)"
if [[ -d "$HOME/.cache/pip" ]]; then
  du -sh "$HOME/.cache/pip" 2>/dev/null || true
  run "pip cache purge"
fi

section "training-repo trajectory logs (candidate 5)"
TRAJ_LOGS=$(find "$REPO_ROOT/runs" -type f -name "trajectory_*.jsonl" 2>/dev/null || true)
if [[ -n "$TRAJ_LOGS" ]]; then
  for f in $TRAJ_LOGS; do
    run "gzip -f '$f'"
  done
else
  echo "no uncompressed trajectory logs"
fi

section "final disk status"
df -h /

if [[ "$APPLY" -eq 0 ]]; then
  echo
  echo "Dry-run complete. Re-run with --apply to execute."
fi
