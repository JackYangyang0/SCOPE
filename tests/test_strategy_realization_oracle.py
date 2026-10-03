from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from SCOPE.verification.strategy_realization_oracle import verify_strategy_realization


class StrategyRealizationOracleTests(unittest.TestCase):
    def verify_source(self, source: str, strategies: list[str]):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "cuda_kernel.cuh").write_text(source, encoding="utf-8")
            return verify_strategy_realization(path, strategies, {"verification": {}})

    def test_unused_float4_macro_does_not_realize_vector_load(self):
        report = self.verify_source(
            "#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])\n"
            "float value = A[index];\n",
            ["Vectorization.GlobalLoadA.float4"],
        )
        self.assertEqual(report["realization_status"], "not_realized")
        self.assertFalse(report["hard_gate_passed"])

    def test_ab_vector_load_requires_both_operands(self):
        only_a = self.verify_source(
            "float4 value = FLOAT4(A[index]);\n",
            ["Vectorization.GlobalLoadAB.float4"],
        )
        both = self.verify_source(
            "float4 av = FLOAT4(A[a_index]);\nfloat4 bv = FLOAT4(B[b_index]);\n",
            ["Vectorization.GlobalLoadAB.float4"],
        )
        self.assertFalse(only_a["hard_gate_passed"])
        self.assertTrue(both["hard_gate_passed"])

    def test_cp_async_strategy_is_a_critical_realization_gate(self):
        report = self.verify_source(
            "__shared__ float tile[128];\n",
            ["Pipeline.CpAsync.Multistage2.SharedAB"],
        )
        self.assertEqual(report["realization_status"], "not_realized")
        self.assertFalse(report["hard_gate_passed"])


if __name__ == "__main__":
    unittest.main()
