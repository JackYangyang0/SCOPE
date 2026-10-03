#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

if pkg-config --exists openblas 2>/dev/null; then
    gcc -O3 -std=c11 openblas_sgemm_fp32.c \
        $(pkg-config --cflags --libs openblas) -lm \
        -o build/openblas_sgemm_fp32
elif [[ -n "${OPENBLAS_ROOT:-}" ]]; then
    gcc -O3 -std=c11 openblas_sgemm_fp32.c \
        -I"${OPENBLAS_ROOT}/include" -L"${OPENBLAS_ROOT}/lib" \
        -Wl,-rpath,"${OPENBLAS_ROOT}/lib" -lopenblas -lm \
        -o build/openblas_sgemm_fp32
else
    echo "OpenBLAS not found. Install its development package or set OPENBLAS_ROOT." >&2
    exit 2
fi

echo "Built build/openblas_sgemm_fp32"
