// Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
//
// This file is part of the Einstein Engine (EinsteinEngine).
//
// EinsteinEngine is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// EinsteinEngine is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// Device driver for the CUDA (nvcc) and ROCm (hipcc) builds. Included only
// from those TUs. The host g++ build never includes this header.
//
// Timed region is the kernel only (events). H2D copies, the checksum
// reduction, and D2H of that scalar are outside the timer. Checksums must be
// finite; a runtime error aborts.
//
// Kernels live in an anonymous namespace so each bench TU gets its own copy
// and identical template instantiations do not collide at link time.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "ee_bench.hpp"
#include "kernels.hpp"

namespace ee_device {

#if defined(__HIPCC__)
using rt_error_t = hipError_t;
using rt_event_t = hipEvent_t;
inline constexpr rt_error_t rt_success = hipSuccess;
inline const char *backend_name() { return "HIP"; }
inline const char *rt_strerror(rt_error_t e) { return hipGetErrorString(e); }
inline rt_error_t rt_malloc(void **p, std::size_t n) { return hipMalloc(p, n); }
inline rt_error_t rt_free(void *p) { return hipFree(p); }
inline rt_error_t rt_memcpy_h2d(void *dst, const void *src, std::size_t n) {
  return hipMemcpy(dst, src, n, hipMemcpyHostToDevice);
}
inline rt_error_t rt_memcpy_d2h(void *dst, const void *src, std::size_t n) {
  return hipMemcpy(dst, src, n, hipMemcpyDeviceToHost);
}
inline rt_error_t rt_memset(void *p, int v, std::size_t n) { return hipMemset(p, v, n); }
inline rt_error_t rt_last_error() { return hipGetLastError(); }
inline rt_error_t rt_sync() { return hipDeviceSynchronize(); }
inline rt_error_t rt_event_create(rt_event_t *e) { return hipEventCreate(e); }
inline rt_error_t rt_event_record(rt_event_t e) { return hipEventRecord(e); }
inline rt_error_t rt_event_sync(rt_event_t e) { return hipEventSynchronize(e); }
inline rt_error_t rt_event_elapsed(float *ms, rt_event_t a, rt_event_t b) {
  return hipEventElapsedTime(ms, a, b);
}
inline rt_error_t rt_event_destroy(rt_event_t e) { return hipEventDestroy(e); }
#else
using rt_error_t = cudaError_t;
using rt_event_t = cudaEvent_t;
inline constexpr rt_error_t rt_success = cudaSuccess;
inline const char *backend_name() { return "CUDA"; }
inline const char *rt_strerror(rt_error_t e) { return cudaGetErrorString(e); }
inline rt_error_t rt_malloc(void **p, std::size_t n) { return cudaMalloc(p, n); }
inline rt_error_t rt_free(void *p) { return cudaFree(p); }
inline rt_error_t rt_memcpy_h2d(void *dst, const void *src, std::size_t n) {
  return cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
}
inline rt_error_t rt_memcpy_d2h(void *dst, const void *src, std::size_t n) {
  return cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);
}
inline rt_error_t rt_memset(void *p, int v, std::size_t n) { return cudaMemset(p, v, n); }
inline rt_error_t rt_last_error() { return cudaGetLastError(); }
inline rt_error_t rt_sync() { return cudaDeviceSynchronize(); }
inline rt_error_t rt_event_create(rt_event_t *e) { return cudaEventCreate(e); }
inline rt_error_t rt_event_record(rt_event_t e) { return cudaEventRecord(e); }
inline rt_error_t rt_event_sync(rt_event_t e) { return cudaEventSynchronize(e); }
inline rt_error_t rt_event_elapsed(float *ms, rt_event_t a, rt_event_t b) {
  return cudaEventElapsedTime(ms, a, b);
}
inline rt_error_t rt_event_destroy(rt_event_t e) { return cudaEventDestroy(e); }
#endif

inline void check(rt_error_t err, const char *what) {
  if (err != rt_success) {
    std::fprintf(stderr, "FATAL: %s %s: %s\n", backend_name(), what, rt_strerror(err));
    std::fflush(stderr);
    std::abort();
  }
}

namespace {

constexpr int kBlock = 256;

struct Buf {
  void *p = nullptr;
  Buf() = default;
  Buf(const Buf &) = delete;
  Buf &operator=(const Buf &) = delete;
  ~Buf() {
    if (p)
      (void)rt_free(p);
  }
  void alloc(std::size_t bytes) { check(rt_malloc(&p, bytes), "malloc"); }
  template <typename T>
  T *get() const {
    return static_cast<T *>(p);
  }
};

template <typename T>
void copy_h2d(T *dst, const T *src, std::size_t n) {
  check(rt_memcpy_h2d(dst, src, n * sizeof(T)), "H2D");
}

// One partial sum per block. Host finishes the reduction so this does not
// depend on double atomicAdd (missing on older CUDA, and not required here).
template <typename T>
__global__ void partial_sum_kernel(const T *__restrict__ x, std::size_t n,
                                   double *__restrict__ partial) {
  double local = 0.0;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride)
    local += static_cast<double>(x[i]);

  __shared__ double smem[kBlock];
  smem[threadIdx.x] = local;
  __syncthreads();
  for (int offset = kBlock / 2; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset)
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partial[blockIdx.x] = smem[0];
}

template <typename T>
double device_sum(const T *x, std::size_t n) {
  constexpr int kGrid = 128;
  Buf partial;
  partial.alloc(static_cast<std::size_t>(kGrid) * sizeof(double));
  partial_sum_kernel<T><<<kGrid, kBlock>>>(x, n, partial.get<double>());
  check(rt_last_error(), "checksum launch");
  check(rt_sync(), "checksum sync");
  double host[kGrid];
  check(rt_memcpy_d2h(host, partial.get<double>(), sizeof(host)), "checksum copy");
  double sum = 0.0;
  for (double v : host)
    sum += v;
  return sum;
}

template <typename Launch>
std::vector<double> time_launches(int warmup, int repeats, Launch launch) {
  for (int w = 0; w < warmup; ++w)
    launch();
  check(rt_last_error(), "warmup launch");
  check(rt_sync(), "warmup sync");

  rt_event_t ev0{}, ev1{};
  check(rt_event_create(&ev0), "event create");
  check(rt_event_create(&ev1), "event create");
  std::vector<double> runs;
  runs.reserve(static_cast<std::size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    check(rt_event_record(ev0), "record start");
    launch();
    check(rt_last_error(), "kernel launch");
    check(rt_event_record(ev1), "record stop");
    check(rt_event_sync(ev1), "kernel sync");
    float ms = 0.f;
    check(rt_event_elapsed(&ms, ev0, ev1), "elapsed");
    if (!(ms > 0.f) || !std::isfinite(ms)) {
      std::fprintf(stderr, "FATAL: %s timer returned %g ms\n", backend_name(),
                   static_cast<double>(ms));
      std::fflush(stderr);
      std::abort();
    }
    runs.push_back(static_cast<double>(ms) * 1.0e6);
  }
  (void)rt_event_destroy(ev0);
  (void)rt_event_destroy(ev1);
  return runs;
}

// PTX constraints are nvcc-only. AMDGPU uses the VGPR constraint. The empty
// asm blocks strength reduction without adding a floating-point op.
// Not every translation unit launches the register kernel, and the float
// overloads are only used when vreal is float.
#if defined(__HIPCC__)
#define EE_DEVICE_MAYBE_UNUSED __attribute__((unused))
#else
#define EE_DEVICE_MAYBE_UNUSED
#endif
EE_DEVICE_MAYBE_UNUSED __device__ void keep_live(double &x) {
#if defined(__HIP_PLATFORM_AMD__)
  asm volatile("" : "+v"(x));
#else
  asm volatile("" : "+d"(x));
#endif
}
EE_DEVICE_MAYBE_UNUSED __device__ void keep_live(float &x) {
#if defined(__HIP_PLATFORM_AMD__)
  asm volatile("" : "+v"(x));
#else
  asm volatile("" : "+f"(x));
#endif
}

EE_DEVICE_MAYBE_UNUSED __device__ unsigned long long ee_bits(double x) {
  return __double_as_longlong(x);
}
EE_DEVICE_MAYBE_UNUSED __device__ unsigned long long ee_bits(float x) {
  return static_cast<unsigned long long>(__float_as_uint(x));
}
EE_DEVICE_MAYBE_UNUSED __device__ int ee_bad(double x) { return isnan(x) || isinf(x); }
EE_DEVICE_MAYBE_UNUSED __device__ int ee_bad(float x) { return isnan(x) || isinf(x); }
EE_DEVICE_MAYBE_UNUSED __device__ double ee_nan() {
  return __longlong_as_double(0x7ff8000000000000LL);
}

// One streamed element is one HBM load. On a GPU that hides every scalar op,
// so each thread repeats the functor `reps` times on a slightly perturbed
// input. Independent iterations (not a carried chain) so this is throughput.
// The xor sink keeps every result live without adding another FP op.
template <typename T, typename F>
__device__ T repeat_sink(int reps, F body) {
  unsigned long long s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int bad = 0;
  int k = 0;
#pragma unroll 1
  for (; k + 3 < reps; k += 4) {
    T z0 = body(k + 0);
    T z1 = body(k + 1);
    T z2 = body(k + 2);
    T z3 = body(k + 3);
    bad |= ee_bad(z0) | ee_bad(z1) | ee_bad(z2) | ee_bad(z3);
    s0 ^= ee_bits(z0);
    s1 ^= ee_bits(z1);
    s2 ^= ee_bits(z2);
    s3 ^= ee_bits(z3);
  }
#pragma unroll 1
  for (; k < reps; ++k) {
    T z = body(k);
    bad |= ee_bad(z);
    s0 ^= ee_bits(z);
  }
  if (bad)
    return static_cast<T>(ee_nan());
  return static_cast<T>(static_cast<double>(s0 ^ s1 ^ s2 ^ s3));
}

inline int stream_reps(const ee_bench::BenchConfig &cfg) { return cfg.quick ? 32 : 256; }

template <typename Op>
__global__ void binary_kernel(const ee_bench::vreal *__restrict__ a,
                              const ee_bench::vreal *__restrict__ b,
                              ee_bench::vreal *__restrict__ c, std::size_t n, int reps, Op op) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const ee_bench::vreal base = a[i];
  const ee_bench::vreal other = b[i];
  constexpr ee_bench::vreal eps = static_cast<ee_bench::vreal>(1.0e-8);
  c[i] = repeat_sink<ee_bench::vreal>(reps, [&](int k) {
    return op(base + static_cast<ee_bench::vreal>(k) * eps, other);
  });
}

template <typename Op>
__global__ void unary_kernel(const ee_bench::vreal *__restrict__ a, ee_bench::vreal *__restrict__ c,
                             std::size_t n, int reps, Op op) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const ee_bench::vreal base = a[i];
  constexpr ee_bench::vreal eps = static_cast<ee_bench::vreal>(1.0e-8);
  c[i] = repeat_sink<ee_bench::vreal>(reps, [&](int k) {
    return op(base + static_cast<ee_bench::vreal>(k) * eps);
  });
}

template <typename Op>
__global__ void ternary_kernel(const ee_bench::vreal *__restrict__ a,
                               const ee_bench::vreal *__restrict__ b,
                               const ee_bench::vreal *__restrict__ c,
                               ee_bench::vreal *__restrict__ d, std::size_t n, int reps, Op op) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const ee_bench::vreal aa = a[i];
  const ee_bench::vreal bb = b[i];
  const ee_bench::vreal cc = c[i];
  constexpr ee_bench::vreal eps = static_cast<ee_bench::vreal>(1.0e-8);
  d[i] = repeat_sink<ee_bench::vreal>(reps, [&](int k) {
    return op(aa + static_cast<ee_bench::vreal>(k) * eps, bb, cc);
  });
}

template <typename Op>
__global__ void compare_kernel(const ee_bench::vreal *__restrict__ a,
                               const ee_bench::vreal *__restrict__ b, char *__restrict__ out,
                               std::size_t n, int reps, Op op) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const ee_bench::vreal base = a[i];
  const ee_bench::vreal other = b[i];
  constexpr ee_bench::vreal eps = static_cast<ee_bench::vreal>(1.0e-8);
  int acc = 0;
#pragma unroll 1
  for (int k = 0; k < reps; ++k)
    acc += op(base + static_cast<ee_bench::vreal>(k) * eps, other);
  out[i] = static_cast<char>(acc & 0x7f);
}

// One thread per interior point. threadIdx.x runs along i so a warp's loads
// of gf[c], gf[c+1], gf[c+nx], and gf[c+nx*ny] are each coalesced.
template <typename Op>
__global__ void grid_kernel(const ee_bench::vreal *__restrict__ gf, ee_bench::vreal *__restrict__ out,
                            std::size_t nx, std::size_t ny, std::size_t nz, Op op) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
  const std::size_t j = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
  const std::size_t k = static_cast<std::size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
  if (i + 1 >= nx || j + 1 >= ny || k + 1 >= nz)
    return;
  const std::size_t c = ee_kernels::ee_lin(i, j, k, nx, ny);
  out[c] = op(gf, out, c, nx, ny);
}

// Many independent chains, no global memory in the inner loop. ns/elem is
// kernel time / (threads * iters): device throughput, not single-thread latency.
template <typename Op>
__global__ void reg_kernel(ee_bench::vreal *__restrict__ out, ee_bench::vreal x0,
                           ee_bench::vreal k, int iters, Op op) {
  using ee_bench::vreal;
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  vreal x = x0 + static_cast<vreal>(tid) * static_cast<vreal>(1.0e-10);
  constexpr int kRepeat = 16;
  const int outer = iters / kRepeat;
#pragma unroll 1
  for (int i = 0; i < outer; ++i) {
#pragma unroll
    for (int j = 0; j < kRepeat; ++j) {
      x = op(x, k);
      keep_live(x);
    }
  }
  out[tid] = x;
}

inline int grid_for(std::size_t n) {
  const std::size_t g = (n + static_cast<std::size_t>(kBlock) - 1) / static_cast<std::size_t>(kBlock);
  if (g == 0 || g > static_cast<std::size_t>(2147483647)) {
    std::fprintf(stderr, "FATAL: %s grid size %zu is not launchable\n", backend_name(), g);
    std::abort();
  }
  return static_cast<int>(g);
}

} // namespace

template <typename Op>
ee_bench::BenchResult time_binary(const char *group, const char *name, const char *node,
                                  const ee_bench::BenchConfig &cfg, const ee_bench::vreal *h_a,
                                  const ee_bench::vreal *h_b, std::size_t n, Op op,
                                  int ops_per_elem) {
  using ee_bench::vreal;
  Buf a, b, c;
  a.alloc(n * sizeof(vreal));
  b.alloc(n * sizeof(vreal));
  c.alloc(n * sizeof(vreal));
  copy_h2d(a.get<vreal>(), h_a, n);
  copy_h2d(b.get<vreal>(), h_b, n);
  check(rt_memset(c.get<vreal>(), 0, n * sizeof(vreal)), "memset");
  const int grid = grid_for(n);
  const int reps = stream_reps(cfg);
  auto runs = time_launches(cfg.warmup, cfg.repeats, [&]() {
    binary_kernel<Op>
        <<<grid, kBlock>>>(a.get<vreal>(), b.get<vreal>(), c.get<vreal>(), n, reps, op);
  });
  const double sum = device_sum(c.get<vreal>(), n);
  const std::size_t n_ops = n * static_cast<std::size_t>(reps);
  return ee_bench::summarize(group, node, name, runs, n_ops, ops_per_elem, sum);
}

template <typename Op>
ee_bench::BenchResult time_unary(const char *group, const char *name, const char *node,
                                 const ee_bench::BenchConfig &cfg, const ee_bench::vreal *h_a,
                                 std::size_t n, Op op) {
  using ee_bench::vreal;
  Buf a, c;
  a.alloc(n * sizeof(vreal));
  c.alloc(n * sizeof(vreal));
  copy_h2d(a.get<vreal>(), h_a, n);
  check(rt_memset(c.get<vreal>(), 0, n * sizeof(vreal)), "memset");
  const int grid = grid_for(n);
  const int reps = stream_reps(cfg);
  auto runs = time_launches(cfg.warmup, cfg.repeats, [&]() {
    unary_kernel<Op><<<grid, kBlock>>>(a.get<vreal>(), c.get<vreal>(), n, reps, op);
  });
  const double sum = device_sum(c.get<vreal>(), n);
  return ee_bench::summarize(group, node, name, runs, n * static_cast<std::size_t>(reps), 1, sum);
}

template <typename Op>
ee_bench::BenchResult time_ternary(const char *group, const char *name, const char *node,
                                   const ee_bench::BenchConfig &cfg, const ee_bench::vreal *h_a,
                                   const ee_bench::vreal *h_b, const ee_bench::vreal *h_c,
                                   std::size_t n, Op op) {
  using ee_bench::vreal;
  Buf a, b, c, d;
  a.alloc(n * sizeof(vreal));
  b.alloc(n * sizeof(vreal));
  c.alloc(n * sizeof(vreal));
  d.alloc(n * sizeof(vreal));
  copy_h2d(a.get<vreal>(), h_a, n);
  copy_h2d(b.get<vreal>(), h_b, n);
  copy_h2d(c.get<vreal>(), h_c, n);
  check(rt_memset(d.get<vreal>(), 0, n * sizeof(vreal)), "memset");
  const int grid = grid_for(n);
  const int reps = stream_reps(cfg);
  auto runs = time_launches(cfg.warmup, cfg.repeats, [&]() {
    ternary_kernel<Op><<<grid, kBlock>>>(a.get<vreal>(), b.get<vreal>(), c.get<vreal>(),
                                         d.get<vreal>(), n, reps, op);
  });
  const double sum = device_sum(d.get<vreal>(), n);
  return ee_bench::summarize(group, node, name, runs, n * static_cast<std::size_t>(reps), 1, sum);
}

template <typename Op>
ee_bench::BenchResult time_compare(const ee_bench::BenchConfig &cfg, const ee_bench::vreal *h_a,
                                   const ee_bench::vreal *h_b, std::size_t n, Op op) {
  using ee_bench::vreal;
  Buf a, b, out;
  a.alloc(n * sizeof(vreal));
  b.alloc(n * sizeof(vreal));
  out.alloc(n * sizeof(char));
  copy_h2d(a.get<vreal>(), h_a, n);
  copy_h2d(b.get<vreal>(), h_b, n);
  check(rt_memset(out.get<char>(), 0, n * sizeof(char)), "memset");
  const int grid = grid_for(n);
  const int reps = stream_reps(cfg);
  auto runs = time_launches(cfg.warmup, cfg.repeats, [&]() {
    compare_kernel<Op>
        <<<grid, kBlock>>>(a.get<vreal>(), b.get<vreal>(), out.get<char>(), n, reps, op);
  });
  const double sum = device_sum(out.get<char>(), n);
  return ee_bench::summarize("branch", "Relational", "cmp_lt_stream", runs,
                             n * static_cast<std::size_t>(reps), 1, sum);
}

template <typename Op>
ee_bench::BenchResult time_grid(const char *name, const char *node, const ee_bench::BenchConfig &cfg,
                                const ee_bench::vreal *h_gf, std::size_t nx, std::size_t ny,
                                std::size_t nz, Op op) {
  using ee_bench::vreal;
  if (nx < 3 || ny < 3 || nz < 3 || (nz - 2) > 65535u) {
    std::fprintf(stderr, "FATAL: grid %zu x %zu x %zu is not launchable\n", nx, ny, nz);
    std::abort();
  }
  const std::size_t n = nx * ny * nz;
  const std::size_t n_eval = (nx - 2) * (ny - 2) * (nz - 2);
  Buf gf, out;
  gf.alloc(n * sizeof(vreal));
  out.alloc(n * sizeof(vreal));
  copy_h2d(gf.get<vreal>(), h_gf, n);
  check(rt_memset(out.get<vreal>(), 0, n * sizeof(vreal)), "memset");
  const dim3 block(32, 8, 1);
  const dim3 grid(static_cast<unsigned>((nx - 2 + 31) / 32),
                  static_cast<unsigned>((ny - 2 + 7) / 8), static_cast<unsigned>(nz - 2));
  auto runs = time_launches(cfg.warmup, cfg.repeats, [&]() {
    grid_kernel<Op><<<grid, block>>>(gf.get<vreal>(), out.get<vreal>(), nx, ny, nz, op);
  });
  const double sum = device_sum(out.get<vreal>(), n);
  return ee_bench::summarize("memory", node, name, runs, n_eval, 1, sum);
}

template <typename Op>
ee_bench::BenchResult time_reg(const char *name, const char *node, const ee_bench::BenchConfig &cfg,
                               Op op) {
  using ee_bench::vreal;
  const int block = kBlock;
  const int blocks = cfg.quick ? 256 : 2048;
  const int iters = cfg.quick ? 1024 : 16384;
  const std::size_t nthreads = static_cast<std::size_t>(block) * static_cast<std::size_t>(blocks);
  const vreal x0 = static_cast<vreal>(1.000001);
  const vreal k = static_cast<vreal>(1.000002);
  Buf out;
  out.alloc(nthreads * sizeof(vreal));
  auto runs = time_launches(cfg.warmup, cfg.repeats, [&]() {
    reg_kernel<Op><<<blocks, block>>>(out.get<vreal>(), x0, k, iters, op);
  });
  const double sum = device_sum(out.get<vreal>(), nthreads);
  const std::size_t n_ops = nthreads * static_cast<std::size_t>(iters);
  return ee_bench::summarize("arith", node, name, runs, n_ops, 1, sum);
}

} // namespace ee_device
