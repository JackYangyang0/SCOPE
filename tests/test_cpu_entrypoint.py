import unittest
from unittest.mock import patch

from SCOPE import app, cpu_app


class CPUEntrypointTest(unittest.TestCase):
    def test_cpu_default_shape_is_512(self) -> None:
        from SCOPE.cpu_app import DEFAULT_DESCRIPTION

        self.assertIn("M:512 N:512 K:512", DEFAULT_DESCRIPTION)

    def test_cuda_app_rejects_cpu_backend_option(self):
        with self.assertRaises(SystemExit):
            app.parse_args(["--backend", "cpu"])

    def test_cuda_app_fixes_pipeline_backend(self):
        with patch("SCOPE.app.run_pipeline") as pipeline:
            app.main(["--build-platform", "linux"])
        self.assertEqual(pipeline.call_args.kwargs["backend"], "cuda")

    def test_cpu_app_defaults_to_staged_strategy_search(self):
        with patch("SCOPE.cpu_app.load_config", return_value={"build": {"platform": "linux"}}), \
                patch("SCOPE.app.run_pipeline") as staged_pipeline:
            cpu_app.main(["--description", "CPU C GEMM M:64 N:64 K:64"])

        pipeline_args = staged_pipeline.call_args.args[0]
        self.assertEqual(staged_pipeline.call_args.kwargs["backend"], "cpu")
        self.assertEqual(pipeline_args.build_platform, "linux")
        self.assertEqual(pipeline_args.description, "CPU C GEMM M:64 N:64 K:64")

    def test_cpu_app_forwards_staged_runtime_options(self):
        with patch("SCOPE.cpu_app.load_config", return_value={"build": {"platform": "windows"}}), \
                patch("SCOPE.app.run_pipeline") as staged_pipeline:
            cpu_app.main([
                "--matrix-size", "128", "256", "64",
                "--benchmark-runs", "7",
                "--benchmark-warmup-runs", "2",
                "--keep-generated-cache",
            ])

        pipeline_args = staged_pipeline.call_args.args[0]
        self.assertEqual(staged_pipeline.call_args.kwargs["backend"], "cpu")
        self.assertEqual(pipeline_args.matrix_size, [128, 256, 64])
        self.assertEqual(pipeline_args.benchmark_runs, 7)
        self.assertEqual(pipeline_args.benchmark_warmup_runs, 2)
        self.assertTrue(pipeline_args.keep_generated_cache)

if __name__ == "__main__":
    unittest.main()
