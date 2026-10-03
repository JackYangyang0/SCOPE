TARGET ?= gemm_cpu
BUILD_DIR ?= build
PYTHON ?= python
.DEFAULT_GOAL := all

ifeq ($(OS),Windows_NT)
PLATFORM := windows
EXEEXT := .exe
MKDIR_BUILD = $(PYTHON) -c "from pathlib import Path; Path(r'$(BUILD_DIR)').mkdir(parents=True, exist_ok=True)"
REMOVE_BUILD = $(PYTHON) -c "import shutil; shutil.rmtree(r'$(BUILD_DIR)', ignore_errors=True)"
OPENMP_CFLAGS := -fopenmp
OPENMP_LDLIBS := -fopenmp
else
UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),Linux)
PLATFORM := linux
else
$(error Unsupported operating system: OS='$(OS)' uname='$(UNAME_S)')
endif
EXEEXT :=
MKDIR_BUILD = mkdir -p "$(BUILD_DIR)"
REMOVE_BUILD = rm -rf "$(BUILD_DIR)"
OPENMP_CFLAGS := -fopenmp
OPENMP_LDLIBS := -fopenmp
endif

# GNU make defines CC=cc by default. Prefer GCC for the AVX2/FMA/OpenMP
# kernels unless the caller explicitly selects another compatible compiler.
ifeq ($(origin CC),default)
CC := gcc
endif

TARGET_FILE := $(BUILD_DIR)/$(TARGET)$(EXEEXT)

# Match the compilation protocol used while ranking generated CPU candidates.
# Override these variables explicitly for another machine.
CFLAGS ?= -O3 -std=c11 -march=native -ffast-math -funroll-loops
CFLAGS += -mavx2 -mfma $(OPENMP_CFLAGS)
LDFLAGS ?=
LDLIBS ?= -lm $(OPENMP_LDLIBS)
OMP_NUM_THREADS ?= 32
OMP_DYNAMIC ?= FALSE
export OMP_NUM_THREADS
export OMP_DYNAMIC

OBJECTS := \
	$(BUILD_DIR)/main.o \
	$(BUILD_DIR)/kernel_dispatch.o \
	$(BUILD_DIR)/cpu_kernel_256.o \
	$(BUILD_DIR)/cpu_kernel_512.o \
	$(BUILD_DIR)/cpu_kernel_768.o \
	$(BUILD_DIR)/cpu_kernel_1024.o

.PHONY: all clean info run-256 run-512 run-768 run-1024

all: $(TARGET_FILE)

info:
	$(PYTHON) -c "print('PLATFORM=$(PLATFORM)'); print('CC=$(CC)'); print('TARGET=$(TARGET_FILE)'); print('CFLAGS=$(CFLAGS)'); print('LDFLAGS=$(LDFLAGS)'); print('LDLIBS=$(LDLIBS)')"

$(BUILD_DIR):
	$(MKDIR_BUILD)

$(BUILD_DIR)/main.o: main.c kernel.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/kernel_dispatch.o: kernel_dispatch.c kernel.h kernel_variants.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/cpu_kernel_256.o: cpu_kernel_256.c kernel.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Dcpu_gemm=cpu_gemm_256 -c $< -o $@

$(BUILD_DIR)/cpu_kernel_512.o: cpu_kernel_512.c kernel.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Dcpu_gemm=cpu_gemm_512 -c $< -o $@

$(BUILD_DIR)/cpu_kernel_768.o: cpu_kernel_768.c kernel.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Dcpu_gemm=cpu_gemm_768 -c $< -o $@

$(BUILD_DIR)/cpu_kernel_1024.o: cpu_kernel_1024.c kernel.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Dcpu_gemm=cpu_gemm_1024 -c $< -o $@

$(TARGET_FILE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(LDFLAGS) -o $@ $(LDLIBS)

run-256: all
	./$(TARGET_FILE) 256 256 256

run-512: all
	./$(TARGET_FILE) 512 512 512

run-768: all
	./$(TARGET_FILE) 768 768 768

run-1024: all
	./$(TARGET_FILE) 1024 1024 1024

clean:
	$(REMOVE_BUILD)
