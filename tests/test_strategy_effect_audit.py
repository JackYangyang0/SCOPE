import unittest

from SCOPE.app import deterministic_code_regions, is_deterministic_strategy, store_lines
from SCOPE.generate_ir.stage_controller import synthesize_ir_updates


class StrategyEffectTests(unittest.TestCase):
    def test_store_mapping_does_not_overwrite_compute(self):
        self.assertEqual(deterministic_code_regions("Mapping.WarpStore.CoalescedC", {}), ["STORE"])

    def test_safety_changes_require_code_generation(self):
        for sid in ("Vectorization.AlignmentGuard", "Safety.AssumeDivisibleAligned",
                    "Safety.BoundaryPolicy.StaticDivisibleNoGuard"):
            self.assertFalse(is_deterministic_strategy(sid))

    def test_beta_specializations_have_runtime_fallback(self):
        general = store_lines({})
        zero = store_lines({"epilogue": {"mode": "beta_zero_fast_path"}})
        one = store_lines({"epilogue": {"mode": "beta_one_fast_path"}})
        self.assertNotEqual(general, zero)
        self.assertNotEqual(general, one)
        self.assertIn("beta == 0.0f ?", "\n".join(zero))
        self.assertIn("beta == 1.0f ?", "\n".join(one))
        self.assertIn("beta * C[c_index]", "\n".join(zero))

    def test_keep_current_does_not_reset_pipeline(self):
        updates = synthesize_ir_updates("Pipeline.NoAsyncCopy.V1")
        self.assertFalse(any(key.startswith("pipeline.") or key == "resource.shared_memory.pipeline_multiplier"
                             for key in updates))


if __name__ == "__main__":
    unittest.main()
