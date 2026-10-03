import unittest
from pathlib import Path

from SCOPE import app
from SCOPE.verification.resume_terminal import terminal_source_dir


class ResumeTerminalCpuTest(unittest.TestCase):
    def test_uses_persisted_cpu_code_directory(self):
        expected = Path("cpu-code") / "chain.1-2-3"
        source = terminal_source_dir(
            app,
            {"chain_code": "1-2-3", "code_dir": str(expected)},
            {"target": {"backend": "cpu"}},
        )
        self.assertEqual(source, expected)

    def test_cpu_fallback_uses_cpu_generated_root(self):
        source = terminal_source_dir(
            app,
            {"chain_code": "1-2-3"},
            {"target": {"backend": "cpu"}},
        )
        self.assertEqual(source, Path(app.DEFAULT_GENERATED_CPU_CODE_ROOT) / "chain.1-2-3")


if __name__ == "__main__":
    unittest.main()
