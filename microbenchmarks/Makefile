# Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Simple fallback build when cmake is unavailable. Prefers cmake if present.
CXX ?= g++
CXXFLAGS ?= -O3 -Wall -Wextra -fno-math-errno -std=c++17
SRC = src/main.cpp src/bench_arith.cpp src/bench_pow.cpp src/bench_transcendental.cpp src/bench_memory.cpp src/bench_branch.cpp

all: ee_microbench

ee_microbench: $(SRC) common/ee_bench.hpp common/kernels.hpp
	$(CXX) $(CXXFLAGS) -Icommon $(SRC) -o ee_microbench -lm

quick: ee_microbench
	./ee_microbench --quick

clean:
	rm -f ee_microbench

.PHONY: all quick clean
