#!/usr/bin/env python
"""Pre-flight the CPU pipeline without spending an API token.

    python scripts/preflight_cpu.py --campaign scope-cpu --task-id PILOT-full-256-r01

What it proves, in an isolated copy of the checkout and with the campaign's real
resolved configuration:

1. the launcher used by the runner (``SCOPE.method_entry``) imports ``SCOPE`` from the
   copy and accepts every flag the runner passes;
2. ``run_pipeline(..., backend="cpu")`` switches the IR target to CPU, picks the CPU
   code root and the CPU strategy documents;
3. the initial CPU skeleton compiles and runs through the frozen build verifier;
4. execution reaches the *first* LLM call, where a stand-in client raises
   ``PREFLIGHT_STOP`` so nothing is sent and nothing is charged.

It never writes into the campaign and never touches the shared checkout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.harness import config as config_module, isolation
from experiments.harness import protocol as protocol_module  # noqa: E402
from experiments.harness.paths import campaign_paths  # noqa: E402
from experiments.harness.util import run_command, utc_now, write_json  # noqa: E402

DRIVER = '''
import json
import sys
from SCOPE import app as scope_app

args = scope_app.parse_args(sys.argv[1:])


class _Stub:
    def close(self):
        pass


class FakeClient:
    """Stands in for the LLM client: records the call, never sends it."""

    def __init__(self, config):
        self._config = config
        self._client = _Stub()
        self.calls = []

    def _raise(self, name):
        def call(*a, **k):
            print("PREFLIGHT_STOP " + name, flush=True)
            raise RuntimeError("PREFLIGHT_STOP " + name)
        return call

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._raise(name)

    def usage_snapshot(self):
        return {}


scope_app.OpenAICompatibleClient = FakeClient
try:
    scope_app.run_pipeline(args, backend="cpu")
except RuntimeError as exc:
    if "PREFLIGHT_STOP" not in str(exc):
        raise
print("PREFLIGHT_DONE", flush=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="scope-cpu")
    parser.add_argument("--task-id", default="PILOT-full-256-r01")
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()

    paths = campaign_paths(args.campaign)
    protocol = protocol_module.load(paths, frozen=True, allow_missing_credentials=True)
    if str(protocol.get("backend")) != "cpu":
        print(f"campaign backend is {protocol.get('backend')!r}; this pre-flight is "
              "for a CPU campaign", file=sys.stderr)
        return 2
    records = {record["task_id"]: record for record in
               (protocol.get("tasks") or {}).get("task_records") or []}
    if args.task_id not in records:
        print(f"unknown task: {args.task_id}", file=sys.stderr)
        return 2
    task = records[args.task_id]

    base_config = config_module.load_yaml(
        Path((protocol.get("source") or {})["base_config_path"]))
    model = None
    if task.get("model_role"):
        model = config_module.resolve_model(base_config, task["model_role"])
    effective = config_module.resolve_task_config(
        base_config, task, model=model, backend="cpu",
        base_config_path=Path((protocol.get("source") or {})["base_config_path"]))

    workspace = isolation.create_workspace(paths, f"preflight-{args.task_id}")
    try:
        import yaml
        workspace_config = json.loads(json.dumps(effective["redacted"]))
        for value in (workspace_config.get("llm") or {}).values():
            if isinstance(value, dict):
                value["api_key"] = ""
        (workspace.package_dir / "conf.yaml").write_text(
            yaml.safe_dump(workspace_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8")

        argv = config_module.cli_arguments(
            task, config_path=workspace.package_dir / "conf.yaml",
            build_platform=str((workspace_config.get("build") or {}).get("platform")
                               or "windows"), backend="cpu")
        (Path(workspace.root) / "preflight_driver.py").write_text(DRIVER, encoding="utf-8")
        env = workspace.env()
        env["SCOPE_EXP_BACKEND"] = "cpu"
        # The argv from cli_arguments starts with "-m SCOPE.method_entry"; the driver
        # calls parse_args directly, so drop the interpreter/module prefix.
        forwarded = list(argv)
        if forwarded[:2] == ["-m", "SCOPE.method_entry"]:
            forwarded = forwarded[2:]
        # Delete any IR left over from the repository copy so the observed target
        # must come from this run.
        stale_ir = (workspace.package_dir / "data" / "IRs" / "ir_patch"
                    / "optir.extracted.json")
        if stale_ir.exists():
            stale_ir.unlink()
        result = run_command([sys.executable, "preflight_driver.py", *forwarded],
                             cwd=workspace.root, env=env, timeout=1800)
        output = (result["stdout"] or "") + (result["stderr"] or "")
        reached = "PREFLIGHT_STOP" in output
        ir_path = workspace.package_dir / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
        ir_target = None
        if ir_path.exists():
            ir_target = (json.loads(ir_path.read_text(encoding="utf-8"))
                         .get("target"))
        summary = {
            "created_at": utc_now(),
            "campaign": args.campaign,
            "task_id": args.task_id,
            "backend": "cpu",
            "argv": forwarded,
            "config_hash": effective["config_hash"],
            "returncode": result["returncode"],
            "timed_out": result["timed_out"],
            "reached_first_llm_call": reached,
            "ir_target": ir_target,
            "cpu_target_confirmed": bool(ir_target
                                         and str(ir_target.get("backend")) == "cpu"),
            "stdout_tail": (result["stdout"] or "")[-3000:],
            "stderr_tail": (result["stderr"] or "")[-3000:],
        }
        write_json(paths.audit / "preflight_cpu.json", summary)
        print(json.dumps({key: summary[key] for key in
                          ("returncode", "reached_first_llm_call",
                           "cpu_target_confirmed", "ir_target")}, indent=2))
        if not reached:
            print("\n--- output tail ---")
            print((result["stdout"] or "")[-2500:])
            print((result["stderr"] or "")[-2500:])
            return 4
        print(f"\npre-flight passed; details in {paths.audit / 'preflight_cpu.json'}")
        return 0
    finally:
        isolation.cleanup_workspace(workspace, keep=args.keep_workspace)


if __name__ == "__main__":
    raise SystemExit(main())
