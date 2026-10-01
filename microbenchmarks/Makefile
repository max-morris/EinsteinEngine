# Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Simple fallback build when cmake is unavailable. Prefers cmake if present.
CXX ?= g++
CXXFLAGS ?= -O3 -Wall -Wextra -fno-math-errno -std=c++17
NVCC ?= nvcc
CUDA_ARCH ?= 80
HIPCC ?= hipcc
# gfx908 = MI100. Override: make hip HIP_ARCH=gfx90a
HIP_ARCH ?= $(shell amdgpu-arch 2>/dev/null | head -1)
ifeq ($(strip $(HIP_ARCH)),)
HIP_ARCH := gfx908
endif
SRC = src/main.cpp src/bench_arith.cpp src/bench_pow.cpp src/bench_transcendental.cpp src/bench_memory.cpp src/bench_branch.cpp
HDR = common/ee_bench.hpp common/kernels.hpp common/ee_device.hpp

all: ee_microbench

ee_microbench: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -Icommon $(SRC) -o ee_microbench -lm

# Device kernels. Each target overwrites ./ee_microbench. sm_60+ is required
# (the reduction does not use double atomicAdd). No fast-math.
cuda: $(SRC) $(HDR)
	$(NVCC) -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) -DEE_HAVE_CUDA=1 -Icommon \
	  -Xcompiler -fno-math-errno -x cu $(SRC) -o ee_microbench

# ROCm. Same kernels as cuda, compiled with hipcc.
hip rocm: $(SRC) $(HDR)
	$(HIPCC) -O3 -std=c++17 --offload-arch=$(HIP_ARCH) -DEE_HAVE_HIP=1 -Icommon \
	  -Wall -Wextra -fno-math-errno $(SRC) -o ee_microbench

quick: ee_microbench
	./ee_microbench --quick

clean:
	rm -f ee_microbench

.PHONY: all cuda hip rocm quick clean
