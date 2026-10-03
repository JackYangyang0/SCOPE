NVCC ?= nvcc
TARGET ?= reflexion_gemm
BUILD_DIR ?= build
PYTHON ?= python
.DEFAULT_GOAL := all

AUTO_ARCH := $(shell $(PYTHON) -c "import re,shutil,subprocess; nv=shutil.which('$(NVCC)'); smi=shutil.which('nvidia-smi'); nvout=subprocess.run([nv,'--list-gpu-arch'],capture_output=True,text=True).stdout if nv else ''; gpuout=subprocess.run([smi,'--query-gpu=compute_cap','--format=csv,noheader'],capture_output=True,text=True).stdout if smi else ''; supported=sorted({int(x) for x in re.findall(r'compute_([0-9]+)',nvout)}); match=re.search(r'([0-9]+)\.([0-9]+)',gpuout); hw=int(match.group(1)+match.group(2)) if match else None; lower=[x for x in supported if hw is not None and x<=hw]; chosen=hw if hw in supported else (max(lower) if lower else (max(supported) if supported else 87)); print('sm_'+str(chosen))")
ARCH ?= $(AUTO_ARCH)
ARCH_NUM := $(patsubst sm_%,%,$(ARCH))

ifeq ($(OS),Windows_NT)
EXEEXT := .exe
REMOVE_BUILD = $(PYTHON) -c "import shutil; shutil.rmtree(r'$(BUILD_DIR)', ignore_errors=True)"
else
EXEEXT :=
REMOVE_BUILD = rm -rf "$(BUILD_DIR)"
endif

TARGET_FILE := $(BUILD_DIR)/$(TARGET)$(EXEEXT)
NVCCFLAGS ?= -O3 -std=c++17
MAXRREGCOUNT ?= 128
NVCCFLAGS += -gencode arch=compute_$(ARCH_NUM),code=sm_$(ARCH_NUM) -gencode arch=compute_$(ARCH_NUM),code=compute_$(ARCH_NUM) -lineinfo --maxrregcount=$(MAXRREGCOUNT)
LDLIBS ?= -lcublas
DEPS := kernel.h kernel_variants.h cuda_kernel_512.cuh cuda_kernel_1024.cuh cuda_kernel_2048.cuh

.PHONY: all clean info run-512 run-1024 run-2048

info:
	$(PYTHON) -c "print('NVCC=$(NVCC)'); print('ARCH=$(ARCH)'); print('TARGET=$(TARGET_FILE)')"

ifeq ($(OS),Windows_NT)
all: $(TARGET_FILE)
$(TARGET_FILE): main.cpp kernel_dispatch.cu $(DEPS) build_windows.bat
	cmd /d /c build_windows.bat $(ARCH_NUM) $(TARGET) $(MAXRREGCOUNT)
else
all: $(TARGET_FILE)
$(BUILD_DIR):
	mkdir -p "$(BUILD_DIR)"
$(BUILD_DIR)/main.o: main.cpp kernel.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@
$(BUILD_DIR)/kernel_dispatch.o: kernel_dispatch.cu $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
$(TARGET_FILE): $(BUILD_DIR)/main.o $(BUILD_DIR)/kernel_dispatch.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)
endif

run-512: all
	./$(TARGET_FILE) 512 512 512
run-1024: all
	./$(TARGET_FILE) 1024 1024 1024
run-2048: all
	./$(TARGET_FILE) 2048 2048 2048
clean:
	$(REMOVE_BUILD)
