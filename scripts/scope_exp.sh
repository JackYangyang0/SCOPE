#!/usr/bin/env bash
# Bash wrapper: python -m SCOPE.exp_app <args> using the repository virtualenv.
#
#   ./scripts/scope_exp.sh status --campaign rehearsal
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$root/.venv/bin/python"
[ -x "$python" ] || python="python3"
cd "$root" || exit 2
exec "$python" -m SCOPE.exp_app "$@"