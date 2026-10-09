# Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# CPU build (this machine):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
#   ./build/ee_microbench --quick
#
# CUDA build (GPU machine; default arch is sm_80, override with
# -DCMAKE_CUDA_ARCHITECTURES=80):
#   cmake -S . -B build-cuda -DEE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-cuda
#
# ROCm/HIP build (GPU machine). Default arch is whatever amdgpu-arch reports,
# or gfx908 (MI100) when no GPU is visible. Override with
# -DCMAKE_HIP_ARCHITECTURES=gfx90a (MI200) or gfx942 (MI300):
#   cmake -S . -B build-hip -DEE_ENABLE_HIP=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-hip
#
# CUDA and ROCm are separate binaries. Do not enable both.

cmake_minimum_required(VERSION 3.21)
project(ee_microbenchmarks LANGUAGES CXX)

option(EE_ENABLE_CUDA "Build with CUDA support (GPU benchmarks)" OFF)
option(EE_ENABLE_HIP "Build with ROCm/HIP support (GPU benchmarks)" OFF)
option(EE_VREAL_FLOAT "Use float instead of double for vreal" OFF)

if(EE_ENABLE_CUDA AND EE_ENABLE_HIP)
  message(FATAL_ERROR "EE_ENABLE_CUDA and EE_ENABLE_HIP are separate builds; enable only one")
endif()

if(EE_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_compile_definitions(EE_HAVE_CUDA=1)
endif()

if(EE_ENABLE_HIP)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
      set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
      set(ROCM_PATH /opt/rocm)
    endif()
  endif()
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
  if(NOT CMAKE_HIP_ARCHITECTURES)
    execute_process(
      COMMAND "${ROCM_PATH}/bin/amdgpu-arch"
      OUTPUT_VARIABLE EE_HIP_ARCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE EE_HIP_ARCH_RC)
    if(EE_HIP_ARCH_RC EQUAL 0 AND NOT EE_HIP_ARCH STREQUAL "")
      string(REPLACE "\n" ";" EE_HIP_ARCH_LIST "${EE_HIP_ARCH}")
      list(GET EE_HIP_ARCH_LIST 0 EE_HIP_ARCH0)
      set(CMAKE_HIP_ARCHITECTURES "${EE_HIP_ARCH0}")
      message(STATUS "CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES} (amdgpu-arch)")
    else()
      set(CMAKE_HIP_ARCHITECTURES gfx908)
      message(STATUS
        "No AMD GPU visible; CMAKE_HIP_ARCHITECTURES=gfx908 (MI100). "
        "Override with -DCMAKE_HIP_ARCHITECTURES=<gfx>.")
    endif()
  endif()
  enable_language(HIP)
  # hip-config probes amdgpu-arch if this is unset, and warns on a CPU node.
  if(NOT GPU_TARGETS)
    set(GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "AMD GPU targets to compile for")
  endif()
  find_package(hip REQUIRED CONFIG)
  add_compile_definitions(EE_HAVE_HIP=1)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(EE_ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
if(EE_ENABLE_HIP)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()

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
  target_compile_options(ee_microbench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -fno-math-errno>)
endif()

if(EE_ENABLE_CUDA)
  set_source_files_properties(
    src/bench_arith.cpp
    src/bench_pow.cpp
    src/bench_transcendental.cpp
    src/bench_memory.cpp
    src/bench_branch.cpp
    PROPERTIES LANGUAGE CUDA)
  set_target_properties(ee_microbench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_link_libraries(ee_microbench PRIVATE CUDA::cudart)
  target_compile_options(ee_microbench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-fno-math-errno>)
endif()

if(EE_ENABLE_HIP)
  # Bench TUs carry the kernels. main.cpp stays host C++ and only queries the device.
  # hip::host supplies the runtime include and libamdhip64. Do not link hip::device:
  # it injects `-x hip` into every CXX file.
  set_source_files_properties(
    src/bench_arith.cpp
    src/bench_pow.cpp
    src/bench_transcendental.cpp
    src/bench_memory.cpp
    src/bench_branch.cpp
    PROPERTIES LANGUAGE HIP)
  set_target_properties(ee_microbench PROPERTIES LINKER_LANGUAGE HIP)
  target_link_libraries(ee_microbench PRIVATE hip::host)
  target_compile_options(ee_microbench PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-Wall -Wextra -fno-math-errno>)
endif()
