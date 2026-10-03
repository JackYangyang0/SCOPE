# Combined AutoComp and Reflexion GEMM Results

This directory uses the same benchmark shape as the SCOPE `results` bundle:

- one shared `main.cpp` for timing, cuBLAS FP32 comparison, and correctness;
- one size dispatcher per method;
- one executable for AutoComp and one for Reflexion;
- matrix arguments are ordered as `M K N`.

## Build on Linux/H800

AutoComp and Reflexion are built independently with separate Makefiles:

```bash
make -C autocomp ARCH=sm_90
make -C reflexion ARCH=sm_90
```

Override `ARCH` when compiling for another GPU, for example:

```bash
make -C autocomp clean
make -C autocomp ARCH=sm_89
```

## Run

```bash
./autocomp/build/autocomp_gemm 512 512 512
./reflexion/build/reflexion_gemm 512 512 512
make -C autocomp run-all
make -C reflexion run-all
```

`reflexion_4096.cu` was not present in the supplied kernel directory. The
4096 Reflexion benchmark therefore dispatches to `reflexion_2048.cu`, the
largest available variant. Also note that `reflexion_2048.cu` converts FP32
inputs to FP16 and uses WMMA, whereas the cuBLAS reference is strict FP32.
