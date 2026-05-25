#!/usr/bin/env bash
# Install Scribe's git pre-commit hook (symlinks to scripts/pre-commit.sh).
# Idempotent.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

HOOK=".git/hooks/pre-commit"
TARGET="../../scripts/pre-commit.sh"

mkdir -p .git/hooks
chmod +x scripts/pre-commit.sh

if [ -L "$HOOK" ]; then
  # Already a symlink — relink (cheap; idempotent).
  ln -sf "$TARGET" "$HOOK"
elif [ -e "$HOOK" ]; then
  echo "Refusing to overwrite an existing non-symlink hook at $HOOK." >&2
  echo "Move it aside and rerun, or symlink manually:" >&2
  echo "  ln -sf ../../scripts/pre-commit.sh .git/hooks/pre-commit" >&2
  exit 1
else
  ln -s "$TARGET" "$HOOK"
fi

echo "✓ pre-commit hook installed at $HOOK"
echo "  Tests will run before each commit. Bypass with --no-verify."
