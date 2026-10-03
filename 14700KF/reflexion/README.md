# Reflexion CPU GEMM

The four kernels are dispatched for matrix extents 256, 512, 768, and 1024.
Only the selected final implementations are retained, under names such
as `gemm_512.c`.

```bash
make
./build/reflexion_cpu 512 512 512
make run-all
```

Compilation succeeds, but the supplied `gemm_256.c` does not pass the
shared numerical correctness test and must be treated as an invalid candidate.
