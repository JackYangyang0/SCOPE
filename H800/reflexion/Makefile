.DEFAULT_GOAL := help
ARCH ?= sm_90

.PHONY: help autocomp reflexion clean

help:
	@echo "Build each method independently:"
	@echo "  make autocomp ARCH=sm_90"
	@echo "  make reflexion ARCH=sm_90"
	@echo "or use make -C autocomp / make -C reflexion"

autocomp:
	$(MAKE) -C autocomp ARCH=$(ARCH)

reflexion:
	$(MAKE) -C reflexion ARCH=$(ARCH)

clean:
	$(MAKE) -C autocomp clean
	$(MAKE) -C reflexion clean
