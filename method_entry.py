from __future__ import annotations

import importlib.util
import os
import pathlib
import sys


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "app.py").exists() and (candidate / "__init__.py").exists():
            return candidate
    return here.parents[1]


def _scope_app():
    try:
        from SCOPE import app as module
        return module
    except ModuleNotFoundError:
        pass
    root = _repo_root()
    if "SCOPE" not in sys.modules and (root / "__init__.py").exists():
        spec = importlib.util.spec_from_file_location(
            "SCOPE", root / "__init__.py", submodule_search_paths=None,
            submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules["SCOPE"] = package
        spec.loader.exec_module(package)
    from SCOPE import app as module
    return module


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    backend = os.environ.get("SCOPE_EXP_BACKEND", "cuda").strip().lower()
    module = _scope_app()
    if backend == "cpu":
        args = module.parse_args(arguments)
        module.run_pipeline(args, backend="cpu")
        return 0
    module.main(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
