# Reflexion GEMM benchmark

```bash
make ARCH=sm_90
./build/reflexion_gemm 512 512 512
make run-all
```

This directory is self-contained and can be copied independently. Each of the
512, 1024, 2048, and 4096 shapes dispatches to its repaired kernel variant.
