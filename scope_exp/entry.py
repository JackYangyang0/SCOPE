
from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    backend = os.environ.get("SCOPE_EXP_BACKEND", "cuda").strip().lower()
    if backend == "cpu":
        from SCOPE import app as scope_app
        args = scope_app.parse_args(arguments)
        scope_app.run_pipeline(args, backend="cpu")
        return 0
    from SCOPE.app import main as scope_main
    scope_main(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
