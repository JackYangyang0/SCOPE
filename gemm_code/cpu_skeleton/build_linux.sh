#!/usr/bin/env bash
set -euo pipefail

mkdir -p build
cc="${CC:-gcc}"
"${cc}" -O2 -std=c11 main.c cpu_kernel.c -lm -o build/gemm_cpu
echo "Built build/gemm_cpu"
