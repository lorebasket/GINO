#!/usr/bin/env bash
# Wait for substitution suite, then run stock SharPy UVLM contributions.
set -euo pipefail
SUBST_PID="${1:-}"
if [[ -n "$SUBST_PID" ]] && kill -0 "$SUBST_PID" 2>/dev/null; then
  echo "Waiting for PID $SUBST_PID to finish..."
  while kill -0 "$SUBST_PID" 2>/dev/null; do sleep 30; done
fi
exec "$(dirname "$0")/run_sharpy_only_contributions.sh"
