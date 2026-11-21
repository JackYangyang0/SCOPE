# SCOPE

SCOPE is a structured, strategy-guided system for generating and optimizing
shape-specialized FP32 GEMM kernels for CUDA GPUs and multicore CPUs.

The repository and the importable Python package are both named `SCOPE`.

## Included artifact

- staged GPU and CPU generation entry points;
- typed IR extraction and validation;
- strategy libraries and dependency graphs;
- deterministic transformations and LLM patch generation;
- semantic checking, locked repair, compilation, correctness, and runtime gates;
- feedback optimization and constrained tile re-instantiation;
- CUDA/cuBLAS and CPU/OpenBLAS benchmark harnesses;
- RQ2/RQ3 ablation scripts, compact results, protocol audit, and tests;
- representative final CUDA and CPU kernel bundles.

Generated chains, API responses, build products, caches, and full raw experiment
logs are intentionally excluded from this source release.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate              # Linux
# .venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

Set the API key for the provider selected in `SCOPE/conf.yaml`, for example:

```bash
export CMECLOUD_API_KEY=...
```

CUDA execution additionally requires an NVIDIA driver and CUDA Toolkit with
`nvcc`. CPU baseline comparison requires OpenBLAS headers and libraries.

## Run

```bash
python -m SCOPE.app --build-platform linux --matrix-size 512 512 512
python -m SCOPE.cpu_app --build-platform linux --matrix-size 512 512 512
python -m SCOPE.benchmark --backend gpu --program scope=/path/to/gemm
python -m SCOPE.tune --backend cuda --source /path/to/generated/kernel
```

Use `--build-platform windows` on Windows. See `SCOPE/docs/` for the chain,
shape-reuse, architecture, and RQ2/RQ3 experiment descriptions.

### Generate a CUDA bundle for several matrix sizes

Set the dimensions in `conf.yaml` in **M, N, K** order. Each triple starts its
own generation from Phase 1. Phase 3 tunes only the shape of that run.

```yaml
generation:
  alias: gemm
  shapes: [[512, 512, 512], [1024, 1024, 1024], [2048, 2048, 2048]]
```

From the project directory on Windows, run:

```powershell
.\.venv\Scripts\python.exe app.py
```

The console prints the batch name and output path. For the first batch named
`gemm`, the checkpoint is `results/batches/gemm/batch.json`. Each shape has
a private directory, for example `results/batches/gemm/512x512x512/run/`,
containing `ir/`, `gemm_code/`, and `results/`. New runs using the same alias
get names such as `gemm-2` and `gemm-3`.

After interruption, supply the printed batch name and the original pipeline
options, for example `.\.venv\Scripts\python.exe app.py --resume-batch gemm`.
Completed shapes are verified against their saved Phase 3 measurements. An
unfinished shape restarts from Phase 1 in a new attempt directory.

When every configured shape has a verified winner, the batch publishes a single
build directory. Its `shape_manifest.json` maps dimensions to kernel files,
measured GFLOPS and source directories. Example for the first batch:

```powershell
cd results\batches\gemm\results
make
make run-512 run-1024 run-2048
```

Rectangular shapes get targets such as `make run-512x1024x256`. Make passes
the dimensions to the benchmark in its native M, K, N order. An explicit
`--matrix-size M N K` runs the original single-shape pipeline with its
usual output directories. `--resume-terminal` retains its single-run meaning.


## Credentials and data

No API credentials are included. Full raw generations can contain model output,
absolute paths, and substantial intermediate data, so publish them separately as
an archival dataset if required by the artifact evaluation policy.

## License

No license is selected in the research workspace. Add the license chosen by the
authors before making the GitHub repository public.
