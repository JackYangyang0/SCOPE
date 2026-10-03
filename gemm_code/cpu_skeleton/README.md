# SCOPE CPU GEMM Skeleton

This directory is the CPU/C backend skeleton. It is intentionally separate from the CUDA skeleton.

Expected files:

- `main.c`: CPU correctness and timing harness.
- `kernel.h`: public CPU GEMM signature.
- `cpu_kernel.c`: generated CPU GEMM implementation target.

`main.c` and `kernel.h` are fixed harness/interface files. LLM generation and
AST extraction operate on `cpu_kernel.c` and the public declaration only;
`main.c` is copied for builds but is not an optimization target.

The checked-in kernel is intentionally scalar and unblocked. Every cache
tiling, packing, SIMD, threading, and compiler optimization must therefore be
introduced by an explicit strategy and remain visible in the strategy history.

Build the pristine skeleton with `build.bat` on Windows or
`bash build_linux.sh` on Linux.
