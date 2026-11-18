"""Effective configuration (REQ-01), redaction (REQ-07) and the ablation audit."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from SCOPE.experiments import ablation_audit as ablation_audit
from SCOPE.utils import harness_config as config_module
from SCOPE.experiments import isolation as isolation
from SCOPE.utils import api_ledger as ledger
from SCOPE.utils import redaction as redact

import support


class EffectiveConfigTest(unittest.TestCase):
    def test_shape_and_tuning_targets_are_pinned(self):
        base = config_module.load_yaml(support.base_config())
        effective = config_module.resolve_task_config(
            base, support.sample_task(shape_M=2048, shape_N=2048, shape_K=2048),
            model=None, base_config_path=support.base_config())
        derived = effective["derived"]
        self.assertEqual(derived["matrix_size_MNK"], [2048, 2048, 2048])
        self.assertEqual(effective["redacted"]["final_top1_tuning"]["gpu_shapes"],
                         [2048])
        self.assertFalse(effective["redacted"]["shape_specialization"]["enabled"])
        self.assertEqual(derived["shape_reuse"], "off")

    def test_cpu_backend_pins_the_cpu_tuning_target(self):
        base = config_module.load_yaml(support.base_config())
        effective = config_module.resolve_task_config(
            base, support.sample_task(shape_M=512, shape_N=512, shape_K=512),
            model=None, backend="cpu", base_config_path=support.base_config())
        tuning = effective["redacted"]["final_top1_tuning"]
        self.assertEqual(tuning["cpu_shapes"], [512])
        self.assertEqual(tuning["gpu_shapes"], [])
        self.assertEqual(effective["derived"]["backend"], "cpu")
        names = {check["check"] for check in effective["checks"]}
        self.assertIn("final_top1_tuning.cpu_shapes", names)
        self.assertIn("final_top1_tuning.gpu_shapes", names)

    def test_launcher_argv_names_the_shared_entry_point(self):
        argv = config_module.cli_arguments(support.sample_task(),
                                           config_path=Path("c.yaml"),
                                           build_platform="windows",
                                           backend="cpu")
        self.assertEqual(argv[:2], ["-m", "SCOPE.method_entry"])
        self.assertIn("--ablation", argv)

    def test_ablation_block_matches_the_method_table(self):
        for variant, switches in config_module.ABLATION_SWITCHES.items():
            block = config_module.ablation_block(variant, "run")
            for switch, value in switches.items():
                self.assertEqual(block[switch], value, f"{variant}.{switch}")
            self.assertEqual(block["enabled"], variant != "full")
            self.assertTrue(block["stage_constraints"])
            self.assertTrue(block["shadow_evaluation"])

    def test_non_cube_task_is_refused(self):
        base = config_module.load_yaml(support.base_config())
        with self.assertRaises(Exception):
            config_module.resolve_task_config(
                base, support.sample_task(shape_M=512, shape_N=512, shape_K=768),
                model=None, base_config_path=support.base_config())

    def test_compact_profile_is_refused(self):
        base = config_module.load_yaml(support.base_config())
        with self.assertRaises(Exception):
            config_module.resolve_task_config(
                base, support.sample_task(), model=None,
                base_config_path=support.base_config(), budget_profile="compact")

    def test_compact_looking_values_are_detected(self):
        base = config_module.load_yaml(support.base_config())
        compact_like = copy.deepcopy(base)
        compact_like["final_top1_tuning"]["max_candidates_per_shape"] = 6
        compact_like["final_top1_tuning"]["rounds"] = 2
        effective = config_module.resolve_task_config(
            compact_like, support.sample_task(), model=None,
            base_config_path=support.base_config())
        with self.assertRaises(Exception):
            config_module.assert_no_compact_leak(effective, compact_like)

    def test_model_resolution_rejects_placeholders(self):
        broken = {"llm": {"provider": "cmecloud",
                          "qwen": {"model": "REPLACE_ME", "base_url": "",
                                   "api_key_env": "X"}}}
        with self.assertRaises(Exception):
            config_module.resolve_model(broken, "qwen")
        broken["llm"]["qwen"]["model"] = "qwen3.5-397b-a17b"
        broken["llm"]["qwen"]["base_url"] = "https://example.invalid/v1"
        resolved = config_module.resolve_model(broken, "qwen3.5-397b-a17b")
        self.assertEqual(resolved["model"], "qwen3.5-397b-a17b")
        self.assertFalse(resolved["api_key_env_present"])

    def test_cli_argv_never_contains_a_secret(self):
        base = config_module.load_yaml(support.base_config())
        effective = config_module.resolve_task_config(
            base, support.sample_task(), model=None,
            base_config_path=support.base_config())
        argv = config_module.cli_arguments(support.sample_task(), config_path=Path("c.yaml"),
                                           build_platform="windows")
        self.assertIn("--ablation", argv)
        self.assertIn("--shape-reuse", argv)
        self.assertIn("off", argv)
        self.assertNotIn("compact", argv)
        self.assertNotIn("api_key", " ".join(argv))
        self.assertFalse(any("sk-" in part for part in argv))

    def test_every_assertion_is_recorded(self):
        base = config_module.load_yaml(support.base_config())
        effective = config_module.resolve_task_config(
            base, support.sample_task(), model=None,
            base_config_path=support.base_config())
        names = {check["check"] for check in effective["checks"]}
        for required in ("matrix_size", "final_top1_tuning.gpu_shapes",
                         "ablation.variant", "execution.gpu_verification_workers"):
            self.assertIn(required, names)
        self.assertTrue(all(check["ok"] for check in effective["checks"]))


class RedactionTest(unittest.TestCase):
    def test_api_key_is_replaced_and_reported(self):
        config = {"llm": {"provider": "x",
                          "qwen": {"api_key": "sk-abcdefghijklmnopqrstuvwx",
                                   "api_key_env": "QWEN_API_KEY",
                                   "model": "m"}}}
        redacted, paths = redact.redact_config(config)
        self.assertEqual(redacted["llm"]["qwen"]["api_key"], redact.REDACTED)
        self.assertEqual(redacted["llm"]["qwen"]["api_key_env"], "QWEN_API_KEY")
        self.assertIn("llm.qwen.api_key", paths)
        self.assertNotIn("api_key_env", " ".join(paths))

    def test_token_budgets_are_not_secrets(self):
        """max_tokens is a number the client parses; redacting it aborts the run."""
        config = {"llm": {"provider": "x", "cmecloud": {
            "api_key": "sk-abcdefghijklmnopqrstuvwx",
            "api_key_env": "K",
            "max_tokens": 8192,
            "selection_max_tokens": 4096,
            "codegen_max_tokens": 16384,
            "total_tokens": 10,
        }}}
        redacted, paths = redact.redact_config(config)
        entry = redacted["llm"]["cmecloud"]
        self.assertEqual(entry["max_tokens"], 8192)
        self.assertEqual(entry["selection_max_tokens"], 4096)
        self.assertEqual(entry["codegen_max_tokens"], 16384)
        self.assertEqual(entry["total_tokens"], 10)
        self.assertEqual(entry["api_key"], redact.REDACTED)
        self.assertEqual(paths, ["llm.cmecloud.api_key"])

    def test_numeric_field_is_never_a_secret_even_with_a_secret_name(self):
        self.assertFalse(redact.is_secret_field("token", 5))
        self.assertFalse(redact.is_secret_field("token", False))
        self.assertTrue(redact.is_secret_field("token", "abc12345"))
        self.assertTrue(redact.is_secret_field("access_token", "abc12345"))
        self.assertFalse(redact.is_secret_field("token_budget", 100))

    def test_credential_patterns_are_detected(self):
        self.assertTrue(redact.find_credentials("key = sk-" + "a" * 24))
        self.assertTrue(redact.find_credentials("Authorization: Bearer " + "b" * 30))
        self.assertFalse(redact.find_credentials("nothing secret here"))

    def test_live_environment_values_are_detected(self):
        hits = redact.find_credentials("token is hunter2hunter2",
                                       ["hunter2hunter2"])
        self.assertTrue(hits)


@unittest.skipIf(support.cpu_toolchain() is None, "needs app.py only, kept for symmetry")
class AblationAuditAgainstMethodTest(unittest.TestCase):
    def test_mapping_matches_app_py(self):
        result = ablation_audit.check_mapping()
        self.assertTrue(result["ok"], result["problems"])
        self.assertEqual(len(result["variants"]), 9)   # 7 variants + 2 aliases

    def test_fixture_shows_the_declared_change(self):
        result = ablation_audit.audit_fixture()
        self.assertTrue(result["ok"], result["problems"])
        rows = {row["variant"]: row for row in result["variants"]}
        self.assertGreater(len(rows["minus_state_in_prompt"]["removed_relative_to_full"]), 0)
        self.assertEqual(rows["minus_state_in_prompt"]["predicates_suppressed"], 0)
        self.assertGreater(rows["intent_only_control"]["predicates_suppressed"], 0)
        self.assertEqual(rows["minus_local_checks"]["removed_relative_to_full"], [])
        self.assertTrue(result["shadow_under_local_check_ablation"]["recorded"])

    def test_activation_is_not_claimed_for_runtime_factors(self):
        fixture = ablation_audit.audit_fixture()
        activation = {row["variant"]: row
                      for row in ablation_audit.activation_summary(fixture)["rows"]}
        self.assertIsNone(activation["minus_stage_ordering"]["activated_in_fixture"])
        self.assertIn("decided at run time", activation["minus_stage_ordering"]["evidence"])


class RuntimeConfigTest(unittest.TestCase):
    """The child process must receive real numbers and no credential."""

    def test_runtime_config_keeps_numbers_and_blanks_the_key(self):
        import yaml
        from SCOPE.experiments import runner as runner
        effective = support.resolved_sample_config()
        runtime = config_module.runtime_config(effective)
        self.assertIn("llm", runtime)
        # a real numeric budget survives into the file the client parses
        section = next(value for key, value in runtime["llm"].items()
                       if isinstance(value, dict))
        self.assertIsInstance(int(section["max_tokens"]), int)
        written = runner.workspace_config(runtime, task_id="T")
        section = next(value for key, value in written["llm"].items()
                       if isinstance(value, dict))
        self.assertEqual(section["api_key"], "")
        self.assertIsInstance(int(section["max_tokens"]), int)
        # and the serialised form must never carry the runtime copy
        from SCOPE.utils.harness_util import write_json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.json"
            write_json(path, {k: v for k, v in effective.items()
                              if k not in ("redacted", "_runtime_config")}
                       | {"redacted_config": effective["redacted"]})
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("_runtime_config", text)
            self.assertNotIn("api_key\": \"", text.replace('api_key": ""', ""))

    def test_frozen_client_accepts_the_written_configuration(self):
        """The exact failure that stopped the pilot: int('<redacted>').

        The frozen client lives inside the ``SCOPE`` package, which is only
        importable from an isolated copy, so the check runs there — the same
        context the runner uses.
        """
        import json
        from SCOPE.experiments import runner as runner
        from SCOPE.utils.harness_util import run_command
        campaign = support.temp_campaign("runtime-config")
        workspace = isolation.create_workspace(campaign, "config-check")
        try:
            effective = support.resolved_sample_config()
            written = runner.workspace_config(
                config_module.runtime_config(effective), task_id="config-check")
            llm = written["llm"]
            provider = llm.get("provider")
            if provider in llm and isinstance(llm[provider], dict):
                section = dict(llm[provider])
            else:
                # the sample configuration declares several roles without a
                # provider block; use the first role entry
                provider, block = next((key, value) for key, value in llm.items()
                                       if isinstance(value, dict))
                section = dict(block)
            section["provider"] = provider
            section["api_key"] = "dummy-key-for-construction"
            (workspace.root / "check_client.py").write_text(
                "import json, sys\n"
                "from SCOPE.llm.openai_client import OpenAICompatibleClient\n"
                "config = json.loads(sys.stdin.read())\n"
                "client = OpenAICompatibleClient(config)\n"
                "print('CONSTRUCTED', client._max_tokens, client._max_retries)\n"
                "client._client.close()\n", encoding="utf-8")
            result = run_command([__import__("sys").executable, "check_client.py"],
                                 cwd=workspace.root, env=workspace.env(),
                                 stdin_text=json.dumps(section), timeout=300)
            self.assertEqual(result["returncode"], 0,
                             (result["stdout"] or "") + (result["stderr"] or ""))
            self.assertIn("CONSTRUCTED", result["stdout"])
            self.assertNotIn("redacted", result["stdout"])
        finally:
            isolation.cleanup_workspace(workspace)


class LedgerTest(unittest.TestCase):
    def test_unknown_tokens_stay_unknown(self):
        row = ledger.record_request(request_id="r1", task_id="t", attempt_id="a",
                                    phase=None, call_type="codegen",
                                    started_at="now", duration_seconds=1.0,
                                    failed=False, tokens=None)
        self.assertIsNone(row["tokens"]["prompt_tokens"])
        self.assertIsNone(row["cost_usd"])
        summary = ledger.summarize([row])
        self.assertEqual(summary["tokens_unknown_rows"]["prompt_tokens"], 1)
        self.assertFalse(summary["cost_is_complete"])

    def test_priced_requests_sum(self):
        pricing = {"m": {"input": 1.0, "output": 2.0}}
        rows = [ledger.record_request(
            request_id="r1", task_id="t", attempt_id="a", phase=None,
            call_type="codegen", started_at="now", duration_seconds=1.0,
            failed=False, tokens={"prompt_tokens": 1_000_000,
                                  "completion_tokens": 500_000},
            returned_model="m")]
        summary = ledger.summarize(rows, pricing=pricing)
        self.assertAlmostEqual(summary["cost_usd_known"], 2.0)
        self.assertTrue(summary["cost_is_complete"])

    def test_failed_request_is_a_row(self):
        row = ledger.record_request(request_id=None, task_id="t", attempt_id="a",
                                    phase=None, call_type="codegen",
                                    started_at="now", duration_seconds=None,
                                    failed=True, error="timeout")
        summary = ledger.summarize([row])
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["requests"], 1)

    def test_empty_ledger_is_zero_cost_not_unknown(self):
        summary = ledger.summarize([])
        self.assertEqual(summary["requests"], 0)
        self.assertTrue(summary["cost_is_complete"])
        self.assertIn("known to be zero", summary["cost_note"])

    def test_ledger_appends_and_reads_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api_ledger.jsonl"
            api = ledger.ApiLedger(path, task_id="t", attempt_id="a")
            api.add(call_type="selection", duration_seconds=0.5, failed=False)
            api.add(call_type="codegen", duration_seconds=1.5, failed=True)
            rows = api.read()
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[1]["failed"])


if __name__ == "__main__":
    unittest.main()
