#!/usr/bin/env bash
# The formal M0→M6 sequence, with every gate in place.
#
# Stops at the first refusal: a red gate must be fixed, not skipped.
set -euo pipefail
CAMPAIGN="${1:-scope-core}"
BACKEND="${2:-cuda}"
WALL="${3:-28800}"

run() { echo; echo "=== $* ==="; python -m SCOPE.exp_app "$@"; }

run prepare        --campaign "$CAMPAIGN" --blocks E1 E2 --backend "$BACKEND"
run selfcheck      --campaign "$CAMPAIGN"
run controls       --campaign "$CAMPAIGN" --backend "$BACKEND"
run ablation-audit --campaign "$CAMPAIGN"
run freeze         --campaign "$CAMPAIGN" --blocks E1 E2 --backend "$BACKEND" \
                   --per-task-wall-seconds "$WALL"
run pilot          --campaign "$CAMPAIGN"
run status         --campaign "$CAMPAIGN"

cat <<'EOF'

Gates are green. Review analysis/pilot_report.json, then decide the formal budget:

  python -m SCOPE.exp_app run       --campaign <id>
  python -m SCOPE.exp_app branches  --campaign <id>
  python -m SCOPE.exp_app summarize --campaign <id> --figures
  python -m SCOPE.exp_app audit     --campaign <id> --deep
EOF