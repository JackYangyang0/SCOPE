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

## Reproduce the compact RQ2 figure

```bash
python SCOPE/experiments/rq2_4060ti_512/figures/plot_rq2_figures.py
```

The exact input distributions, correctness thresholds, reference precision,
cuBLAS math mode, CPU threading, and timing boundaries are documented in
`SCOPE/experiments/rq2_4060ti_512/EXPERIMENT_PROTOCOL_AUDIT.md`.

## Credentials and data

No API credentials are included. Full raw generations can contain model output,
absolute paths, and substantial intermediate data, so publish them separately as
an archival dataset if required by the artifact evaluation policy.

## License

No license is selected in the research workspace. Add the license chosen by the
authors before making the GitHub repository public.
