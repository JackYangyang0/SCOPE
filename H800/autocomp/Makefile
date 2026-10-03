NVCC ?= nvcc
BUILD_DIR ?= build
ARCH ?= sm_90
ARCH_NUM := $(patsubst sm_%,%,$(ARCH))
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
NVCCFLAGS += -gencode arch=compute_$(ARCH_NUM),code=sm_$(ARCH_NUM) -gencode arch=compute_$(ARCH_NUM),code=compute_$(ARCH_NUM)
LDLIBS ?= -lcublas
ROOT := .
NAMES := 512 1024 2048 4096
ADAPTER_OBJS := $(addprefix $(BUILD_DIR)/autocomp_,$(addsuffix _adapter.o,$(NAMES)))
TARGET := $(BUILD_DIR)/autocomp_gemm
.DEFAULT_GOAL := all
.PHONY: all clean info run-512 run-1024 run-2048 run-4096 run-all
all: $(TARGET)
$(BUILD_DIR):
	mkdir -p $@
$(BUILD_DIR)/main.o: main.cpp kernel.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(ROOT) -x cu -c $< -o $@
$(BUILD_DIR)/autocomp_dispatch.o: autocomp_dispatch.cu kernel.h kernel_variants.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(ROOT) -c $< -o $@
$(BUILD_DIR)/autocomp_%_adapter.o: autocomp_%_adapter.cu autocomp_%.cu kernel_variants.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(ROOT) -c $< -o $@
$(TARGET): $(BUILD_DIR)/main.o $(BUILD_DIR)/autocomp_dispatch.o $(ADAPTER_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)
info:
	@echo NVCC=$(NVCC) ARCH=$(ARCH) TARGET=$(TARGET)
run-512: $(TARGET)
	./$(TARGET) 512 512 512
run-1024: $(TARGET)
	./$(TARGET) 1024 1024 1024
run-2048: $(TARGET)
	./$(TARGET) 2048 2048 2048
run-4096: $(TARGET)
	./$(TARGET) 4096 4096 4096
run-all: run-512 run-1024 run-2048 run-4096
clean:
	rm -rf $(BUILD_DIR)
