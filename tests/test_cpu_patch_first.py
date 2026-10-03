import tempfile
import unittest
from pathlib import Path

from SCOPE.app import try_apply_cpu_patch_first


class CpuPatchFirstTests(unittest.TestCase):
    def test_valid_local_diff_skips_full_source_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            kernel = code_dir / "cpu_kernel.c"
            kernel.write_text(
                "void cpu_gemm(void) {\n    int loop_order = 1;\n}\n",
                encoding="utf-8",
            )
            patch = {
                "modified_code_regions": [{"file": "cpu_kernel.c", "region": "CPU_KERNEL"}],
                "code_patch": {
                    "diff": (
                        "--- a/cpu_kernel.c\n"
                        "+++ b/cpu_kernel.c\n"
                        "@@ -1,3 +1,3 @@\n"
                        " void cpu_gemm(void) {\n"
                        "-    int loop_order = 1;\n"
                        "+    int loop_order = 2;\n"
                        " }\n"
                    )
                },
            }

            generated, apply_result = try_apply_cpu_patch_first(
                patch,
                code_dir,
                ["cpu_kernel.c"],
            )

            self.assertIsNotNone(generated)
            self.assertIsNotNone(apply_result)
            self.assertEqual("pass", apply_result["status"])
            self.assertEqual("cpu_local_patch_applied", generated["generation_method"])
            self.assertTrue(generated["full_source_codegen_skipped"])
            self.assertIn("loop_order = 2", kernel.read_text(encoding="utf-8"))

    def test_empty_diff_uses_existing_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated, apply_result = try_apply_cpu_patch_first(
                {"code_patch": {"diff": ""}},
                Path(tmp),
                ["cpu_kernel.c"],
            )

            self.assertIsNone(generated)
            self.assertIsNone(apply_result)


if __name__ == "__main__":
    unittest.main()
