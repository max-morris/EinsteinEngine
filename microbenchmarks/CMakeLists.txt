# Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# CPU build (this machine):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
#   ./build/ee_microbench --quick
#
# CUDA build (GPU machine):
#   cmake -S . -B build-cuda -DEE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-cuda
#
# HIP build (GPU machine):
#   cmake -S . -B build-hip -DEE_ENABLE_HIP=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-hip

cmake_minimum_required(VERSION 3.21)
project(ee_microbenchmarks LANGUAGES CXX)

option(EE_ENABLE_CUDA "Build with CUDA support (GPU benchmarks)" OFF)
option(EE_ENABLE_HIP "Build with HIP support (GPU benchmarks)" OFF)
option(EE_VREAL_FLOAT "Use float instead of double for vreal" OFF)

if(EE_ENABLE_CUDA)
  enable_language(CUDA)
  add_compile_definitions(EE_HAVE_CUDA=1)
endif()

if(EE_ENABLE_HIP)
  enable_language(HIP)
  add_compile_definitions(EE_HAVE_HIP=1)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ee_microbench
  src/main.cpp
  src/bench_arith.cpp
  src/bench_pow.cpp
  src/bench_transcendental.cpp
  src/bench_memory.cpp
  src/bench_branch.cpp
)

target_include_directories(ee_microbench PRIVATE common)

if(EE_VREAL_FLOAT)
  target_compile_definitions(ee_microbench PRIVATE EE_VREAL_FLOAT=1)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(ee_microbench PRIVATE -O3 -Wall -Wextra -fno-math-errno)
endif()

if(EE_ENABLE_CUDA)
  set_target_properties(ee_microbench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()
