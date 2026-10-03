NVCC ?= nvcc
TARGET ?= gemm
BUILD_DIR ?= build
PYTHON ?= python
.DEFAULT_GOAL := all

# Select an architecture supported by both the installed nvcc and the current
# GPU. An explicit `make ARCH=sm_XX` always takes precedence. If the GPU is
# newer than the toolkit, use nvcc's highest supported architecture and embed
# PTX so the driver can JIT it for the newer device.
AUTO_ARCH := $(shell $(PYTHON) -c "import re,shutil,subprocess; nv=shutil.which('$(NVCC)'); smi=shutil.which('nvidia-smi'); nvout=subprocess.run([nv,'--list-gpu-arch'],capture_output=True,text=True).stdout if nv else ''; gpuout=subprocess.run([smi,'--query-gpu=compute_cap','--format=csv,noheader'],capture_output=True,text=True).stdout if smi else ''; supported=sorted({int(x) for x in re.findall(r'compute_([0-9]+)',nvout)}); match=re.search(r'([0-9]+)\.([0-9]+)',gpuout); hw=int(match.group(1)+match.group(2)) if match else None; lower=[x for x in supported if hw is not None and x<=hw]; chosen=hw if hw in supported else (max(lower) if lower else (max(supported) if supported else 87)); print('sm_'+str(chosen))")
ARCH ?= $(AUTO_ARCH)
ARCH_NUM := $(patsubst sm_%,%,$(ARCH))

ifeq ($(OS),Windows_NT)
PLATFORM := windows
EXEEXT := .exe
MKDIR_BUILD = $(PYTHON) -c "from pathlib import Path; Path(r'$(BUILD_DIR)').mkdir(parents=True, exist_ok=True)"
REMOVE_BUILD = $(PYTHON) -c "import shutil; shutil.rmtree(r'$(BUILD_DIR)', ignore_errors=True)"
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
endif

TARGET_FILE := $(BUILD_DIR)/$(TARGET)$(EXEEXT)

NVCCFLAGS ?= -O3 -std=c++17
MAXRREGCOUNT ?= 128
GENCODE_FLAGS := -gencode arch=compute_$(ARCH_NUM),code=sm_$(ARCH_NUM) -gencode arch=compute_$(ARCH_NUM),code=compute_$(ARCH_NUM)
NVCCFLAGS += $(GENCODE_FLAGS) -lineinfo --maxrregcount=$(MAXRREGCOUNT)
LDLIBS ?= -lcublas

SOURCES := main.cpp kernel_dispatch.cu
OBJECTS := $(addprefix $(BUILD_DIR)/,$(SOURCES:.cpp=.o))
OBJECTS := $(OBJECTS:.cu=.o)
DEPS := kernel.h kernel_variants.h \
	cuda_kernel_512.cuh cuda_kernel_1024.cuh cuda_kernel_2048.cuh cuda_kernel_4096.cuh

.PHONY: all clean info list-arch run-512 run-1024 run-2048 run-4096

info:
	$(PYTHON) -c "print('PLATFORM=$(PLATFORM)'); print('NVCC=$(NVCC)'); print('AUTO_ARCH=$(AUTO_ARCH)'); print('ARCH=$(ARCH)'); print('TARGET=$(TARGET_FILE)'); print('NVCCFLAGS=$(NVCCFLAGS)'); print('LDLIBS=$(LDLIBS)')"

list-arch:
	$(NVCC) --list-gpu-arch

ifeq ($(OS),Windows_NT)

all: $(TARGET_FILE)

$(TARGET_FILE): $(SOURCES) $(DEPS) build_windows.bat
	cmd /d /c build_windows.bat $(ARCH_NUM) $(TARGET) $(MAXRREGCOUNT)

else

all: $(TARGET_FILE)

$(BUILD_DIR):
	$(MKDIR_BUILD)

$(BUILD_DIR)/main.o: main.cpp kernel.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

$(BUILD_DIR)/kernel_dispatch.o: kernel_dispatch.cu $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET_FILE): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o $@ $(LDLIBS)

endif

run-512: all
	./$(TARGET_FILE) 512 512 512

run-1024: all
	./$(TARGET_FILE) 1024 1024 1024

run-2048: all
	./$(TARGET_FILE) 2048 2048 2048

run-4096: all
	./$(TARGET_FILE) 4096 4096 4096

clean:
	$(REMOVE_BUILD)
