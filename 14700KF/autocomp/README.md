# AutoComp CPU GEMM

The four kernels are dispatched for matrix extents 256, 512, 768, and 1024.
Only the selected final implementations are retained, under names such
as `gemm_512.c`.

```bash
make
./build/autocomp_cpu 512 512 512
make run-all
```

Override the thread count when needed:

```bash
make OMP_NUM_THREADS=20
OMP_NUM_THREADS=20 ./build/autocomp_cpu 1024 1024 1024
```
