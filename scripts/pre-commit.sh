#!/usr/bin/env bash
# Scribe pre-commit hook: tests must pass.
#
# Runs the *fast* pytest suite (slow/gpu markers excluded by pytest.ini)
# and the Vitest suite. Pass --no-verify to git commit to bypass.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Don't fight the user about test infrastructure that isn't installed yet.
if [ ! -d .venv ]; then
  echo "[pre-commit] no .venv found — skipping Python tests."
  PYTEST_OK=1
else
  echo "[pre-commit] running pytest (fast suite)…"
  if .venv/bin/pytest -q; then
    PYTEST_OK=1
  else
    PYTEST_OK=0
  fi
fi

if [ ! -d node_modules ]; then
  echo "[pre-commit] no node_modules — skipping JS tests."
  VITEST_OK=1
else
  echo "[pre-commit] running vitest…"
  if npm test --silent; then
    VITEST_OK=1
  else
    VITEST_OK=0
  fi
fi

if [ "$PYTEST_OK" = "1" ] && [ "$VITEST_OK" = "1" ]; then
  exit 0
fi

echo
echo "[pre-commit] BLOCKED: tests failed."
echo "  - Fix the failures, then commit again, OR"
echo "  - re-run with: git commit --no-verify"
echo "    (only if you know what you're doing)"
exit 1
