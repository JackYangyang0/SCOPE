TARGET ?= autocomp_cpu
BUILD_DIR ?= build
ifeq ($(origin CC),default)
CC := gcc
endif
PYTHON ?= python
CFLAGS ?= -O3 -std=gnu11 -march=native -ffast-math -funroll-loops -mavx2 -mfma -fopenmp
LDFLAGS ?=
LDLIBS ?= -lm -fopenmp
OMP_NUM_THREADS ?= 32
OMP_DYNAMIC ?= FALSE
export OMP_NUM_THREADS OMP_DYNAMIC

ifeq ($(OS),Windows_NT)
EXEEXT := .exe
MKDIR_BUILD = $(PYTHON) -c "from pathlib import Path; Path(r'$(BUILD_DIR)').mkdir(parents=True, exist_ok=True)"
REMOVE_BUILD = $(PYTHON) -c "import shutil; shutil.rmtree(r'$(BUILD_DIR)', ignore_errors=True)"
else
EXEEXT :=
MKDIR_BUILD = mkdir -p "$(BUILD_DIR)"
REMOVE_BUILD = rm -rf "$(BUILD_DIR)"
endif

TARGET_FILE := $(BUILD_DIR)/$(TARGET)$(EXEEXT)
SIZES := 256 512 768 1024
KERNEL_OBJECTS := $(addprefix $(BUILD_DIR)/gemm_,$(addsuffix .o,$(SIZES)))
OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/kernel_dispatch.o $(KERNEL_OBJECTS)
.DEFAULT_GOAL := all
.PHONY: all clean info run-256 run-512 run-768 run-1024 run-all

all: $(TARGET_FILE)
$(BUILD_DIR):
	$(MKDIR_BUILD)
$(BUILD_DIR)/main.o: main.c kernel.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD_DIR)/kernel_dispatch.o: kernel_dispatch.c kernel.h kernel_variants.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD_DIR)/gemm_%.o: gemm_%.c repaired_blocked_gemm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Dgemm_cpu=gemm_cpu_$*_raw -c $< -o $@
$(TARGET_FILE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(LDFLAGS) -o $@ $(LDLIBS)
info:
	@echo CC=$(CC) CFLAGS=$(CFLAGS) TARGET=$(TARGET_FILE) OMP_NUM_THREADS=$(OMP_NUM_THREADS)
run-256: all
	./$(TARGET_FILE) 256 256 256
run-512: all
	./$(TARGET_FILE) 512 512 512
run-768: all
	./$(TARGET_FILE) 768 768 768
run-1024: all
	./$(TARGET_FILE) 1024 1024 1024
run-all: run-256 run-512 run-768 run-1024
clean:
	$(REMOVE_BUILD)
