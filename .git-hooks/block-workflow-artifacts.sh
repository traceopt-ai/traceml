#!/usr/bin/env bash
# Refuses to commit machine-local AI / workflow artifacts.
# Receives staged file paths as arguments.
set -euo pipefail

forbidden_regex='(^|/)(CLAUDE\.md|CLAUDE\.local\.md|AGENTS\.md|GEMINI\.md|\.claude(/|$)|\.claude\.json|\.codex/|\.cursor/|\.cursor-server|\.windsurf-server|\.vscode-server|\.planning/|\.gsd/|specs/)'

if [ "$#" -eq 0 ]; then
  exit 0
fi

bad=$(printf '%s\n' "$@" | grep -E "$forbidden_regex" || true)

if [ -n "$bad" ]; then
  echo "ERROR: refusing to commit workflow artifacts:" >&2
  printf '%s\n' "$bad" | sed 's/^/  - /' >&2
  echo "" >&2
  echo "These are local-only. If one slipped in, run: git rm --cached <path>" >&2
  exit 1
fi
