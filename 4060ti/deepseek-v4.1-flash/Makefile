NVCC ?= nvcc
ARCH ?= sm_89
TARGET ?= gemm
BUILD_DIR ?= build
PYTHON ?= python

ifeq ($(OS),Windows_NT)
EXEEXT := .exe
else
EXEEXT :=
endif

TARGET_FILE := $(BUILD_DIR)/$(TARGET)$(EXEEXT)
MKDIR_BUILD = $(PYTHON) -c "from pathlib import Path; Path(r'$(BUILD_DIR)').mkdir(parents=True, exist_ok=True)"
REMOVE_BUILD = $(PYTHON) -c "import shutil; shutil.rmtree(r'$(BUILD_DIR)', ignore_errors=True)"

NVCCFLAGS ?= -O3 -std=c++17
MAXRREGCOUNT ?= 128
NVCCFLAGS += -arch=$(ARCH) -lineinfo --maxrregcount=$(MAXRREGCOUNT)
LDLIBS ?= -lcublas

SOURCES := main.cpp kernel_dispatch.cu
OBJECTS := $(addprefix $(BUILD_DIR)/,$(SOURCES:.cpp=.o))
OBJECTS := $(OBJECTS:.cu=.o)
DEPS := kernel.h kernel_variants.h \
	cuda_kernel_512.cuh cuda_kernel_1024.cuh cuda_kernel_2048.cuh cuda_kernel_4096.cuh

.PHONY: all clean run-512 run-1024 run-2048 run-4096

ifeq ($(OS),Windows_NT)

all: $(TARGET_FILE)

$(TARGET_FILE): $(SOURCES) $(DEPS) build_windows.bat
	cmd /d /c build_windows.bat $(ARCH) $(TARGET)

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
