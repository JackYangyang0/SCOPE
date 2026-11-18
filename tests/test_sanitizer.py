"""Sanitizer applicability and honest states (no GPU required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from SCOPE.verification import sanitizer as sanitizer


class ApplicabilityTest(unittest.TestCase):
    def test_plain_kernel_needs_only_memcheck(self):
        features = sanitizer.source_features("__global__ void k() { float x = 0; }")
        self.assertFalse(features["uses_shared_memory"])
        self.assertFalse(features["uses_explicit_sync"])
        tools = {item.tool: item.applicable for item in
                 sanitizer.applicability("__global__ void k() { }")}
        self.assertTrue(tools["memcheck"])
        self.assertFalse(tools["racecheck"])
        self.assertFalse(tools["synccheck"])

    def test_shared_memory_requires_racecheck_even_without_sync(self):
        source = "__global__ void k() { __shared__ float t[16][16]; t[0][0] = 1.0f; }"
        tools = {item.tool: item.applicable for item in
                 sanitizer.applicability(source)}
        self.assertTrue(tools["racecheck"])
        self.assertTrue(tools["synccheck"])

    def test_comment_cannot_fake_a_synchronisation_site(self):
        source = ("__global__ void k() { __shared__ float t[2]; "
                  "/* deliberately missing __syncthreads() */ t[0] = 1.0f; }")
        features = sanitizer.source_features(source)
        self.assertEqual(features["sync_sites"], 0)
        self.assertFalse(features["uses_explicit_sync"])

    def test_reason_is_recorded_for_every_tool(self):
        for item in sanitizer.applicability("__shared__ float t[4];"):
            self.assertTrue(item.reason)
            self.assertIn("tool", item.as_record())


class StateTest(unittest.TestCase):
    def test_cpu_backend_is_unsupported_not_passing(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = Path(directory) / "driver"
            driver.write_bytes(b"")
            case = Path(directory) / "case.bin"
            case.write_bytes(b"")
            record = sanitizer.run_sanitizer(driver, case, tool="memcheck",
                                             timeout_seconds=5.0,
                                             out_dir=Path(directory),
                                             backend="cpu")
            self.assertEqual(record["status"], "unsupported")
            self.assertIn("CPU", record["reason"])

    def test_cpu_suite_is_scoped_not_a_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            kernel_dir = Path(directory) / "kernel"
            kernel_dir.mkdir()
            (kernel_dir / "cpu_kernel.c").write_text(
                "void cpu_gemm(void) { }", encoding="utf-8")
            record = sanitizer.run_safety_suite(
                kernel_dir, Path(directory) / "driver",
                Path(directory) / "case.bin", kernel_dir, backend="cpu",
                timeout_seconds=5.0)
            self.assertEqual(record["status"], "pass_scoped")
            self.assertEqual(record["scope"], "c_write_bounds_on_all_executed_cases")
            self.assertFalse(record["safety_is_proven"])
            self.assertTrue(record["limitations"])
            self.assertTrue(record["not_run_is_not_safe"])
            self.assertTrue(all(item["status"] == "not_applicable"
                                for item in record["per_tool"]))

    def test_probe_without_tool_is_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            result = sanitizer.probe_timeout(Path(directory) / "driver",
                                             Path(directory) / "case.bin")
            if sanitizer.sanitizer_available():
                self.assertIn(result["status"], ("ok", "inconclusive"))
            else:
                self.assertEqual(result["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
