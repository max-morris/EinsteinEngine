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

// Portable benchmark harness.
//
// GPU-portability contract:
//  * Everything in this header is HOST code (timing, allocation, statistics).
//    It must never be called from inside a __device__ kernel.
//  * Measured kernels live in kernels.hpp and are annotated EE_HD_INLINE so
//    the same source compiles under nvcc/hipcc and plain g++.
//  * No exceptions are swallowed: a corrupted checksum or non-finite result
//    is a fatal error (abort), never a warning.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Device-annotation macros (GPU-portable)
// ---------------------------------------------------------------------------
// EE_DEVICE_BUILD selects the kernel driver (ee_device.hpp). One binary is
// either host, CUDA, or ROCm — never more than one. hipcc is tested first
// because some versions also define __CUDACC__.
#if defined(__HIPCC__)
#define EE_HD_INLINE __host__ __device__ inline
#define EE_DEVICE_INLINE __device__ inline
#define EE_HAVE_GPU 1
#define EE_DEVICE_BUILD 1
#if !defined(EE_HAVE_HIP)
#define EE_HAVE_HIP 1
#endif
#elif defined(__CUDACC__)
#define EE_HD_INLINE __host__ __device__ inline
#define EE_DEVICE_INLINE __device__ inline
#define EE_HAVE_GPU 1
#define EE_DEVICE_BUILD 1
#if !defined(EE_HAVE_CUDA)
#define EE_HAVE_CUDA 1
#endif
#else
#define EE_HD_INLINE inline
#define EE_DEVICE_INLINE inline
#endif

// Prevent the compiler from optimizing a value away without serializing
// through memory. Host-only helper (never called from device code).
template <typename T>
inline void ee_escape(const T *p) {
#if defined(__GNUC__)
  asm volatile("" : : "g"(p) : "memory");
#else
  volatile const T *sink = p;
  (void)sink;
#endif
}

template <typename T>
inline void ee_clobber(T &x) {
#if defined(__GNUC__)
  asm volatile("" : "+g"(x) : : "memory");
#else
  volatile T *s = &x;
  (void)*s;
#endif
}

namespace ee_bench {

// Scalar type mirroring CarpetX codegen (`vreal` in emitted thorns).
#ifdef EE_VREAL_FLOAT
using vreal = float;
#else
using vreal = double;
#endif

struct BenchConfig {
  std::size_t n_stream = 1u << 20; // elements per streaming pass (~8 MiB for double)
  int repeats = 15;                // timed repeats per case (min is reported)
  int warmup = 2;                  // untimed warmup passes
  bool quick = false;              // shrink work for smoke tests
};

inline BenchConfig default_config() {
  BenchConfig c;
#if defined(EE_HAVE_CUDA) || defined(EE_HAVE_HIP)
  // 2^27 elements keeps a GPU kernel in the millisecond range so launch
  // overhead is not the measurement. Host builds keep the member default
  // stream length and take more repeats: a 15-sample minimum of the host
  // add loop moves around by a large fraction of the result.
  c.n_stream = std::size_t{1} << 27;
  c.warmup = 3;
#else
  c.repeats = 30;
#endif
  return c;
}

inline BenchConfig default_config_quick() {
  BenchConfig c;
#if defined(EE_HAVE_CUDA) || defined(EE_HAVE_HIP)
  c.n_stream = std::size_t{1} << 24;
#else
  c.n_stream = std::size_t{1} << 18;
#endif
  c.repeats = 5;
  c.warmup = 1;
  c.quick = true;
  return c;
}

struct BenchResult {
  std::string group;       // e.g. "arith", "pow", "transcendental", "memory", "branch"
  std::string name;        // e.g. "add_stream", "pow2_reg"
  std::string complexity_node; // sympy node this measures, e.g. "Add", "Pow[x**2]", "stencil[x]"
  double ns_per_elem = 0.0;    // min over repeats
  double ns_median = 0.0;
  double gops = 0.0;           // 1e9 elements*ops_per_elem / seconds (ops_per_elem given below)
  int ops_per_elem = 1;
  double checksum = 0.0;       // anti-DCE sink (must be finite)
};

class Timer {
public:
  void start() { t0_ = std::chrono::steady_clock::now(); }
  double stop_ns() const {
    auto t1 = std::chrono::steady_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_).count());
  }

private:
  std::chrono::steady_clock::time_point t0_;
};

// Fill with deterministic pseudo-random values in a safe domain for
// exp/log/pow/sqrt (positive, bounded, no overflow).
inline void fill_inputs(vreal *a, vreal *b, std::size_t n) {
  std::uint64_t s = 0x243F6A8885A308D3ull;
  for (std::size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    // a in [0.5, 1.5), b in [0.25, 1.25)
    a[i] = static_cast<vreal>(0.5 + (s % 1000000) / 1000000.0);
    b[i] = static_cast<vreal>(0.25 + ((s >> 20) % 1000000) / 1000000.0);
  }
}

inline void fill_small_angles(vreal *a, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    a[i] = static_cast<vreal>(-0.5 + 1.0 * (i % 1000) / 1000.0);
}

// Fatal validation: checksums must be finite. Never ignore.
inline void require_finite(double v, const char *what) {
  if (!std::isfinite(v)) {
    std::fprintf(stderr, "FATAL: non-finite checksum in %s: %f\n", what, v);
    std::fflush(stderr);
    std::abort();
  }
}

inline BenchResult summarize(std::string group, std::string node, std::string name,
                             const std::vector<double> &ns_runs, std::size_t n_elem,
                             int ops_per_elem, double checksum) {
  require_finite(checksum, name.c_str());
  std::vector<double> s = ns_runs;
  std::sort(s.begin(), s.end());
  double mn = s.front();
  double med = s[s.size() / 2];
  BenchResult r;
  r.group = std::move(group);
  r.name = std::move(name);
  r.complexity_node = std::move(node);
  r.ns_per_elem = mn / static_cast<double>(n_elem);
  r.ns_median = med / static_cast<double>(n_elem);
  r.ops_per_elem = ops_per_elem;
  // mn is total ns for n_elem*ops; GOPS = n*ops / seconds / 1e9
  r.gops = (static_cast<double>(n_elem) * ops_per_elem) / (mn / 1e9) / 1e9;
  r.checksum = checksum;
  return r;
}

inline void print_json(const std::vector<BenchResult> &rs, std::FILE *f = stdout) {
  std::fprintf(f, "{\"results\":[\n");
  for (std::size_t i = 0; i < rs.size(); ++i) {
    const auto &r = rs[i];
    std::fprintf(f,
                 "  {\"group\":\"%s\",\"name\":\"%s\",\"complexity_node\":\"%s\","
                 "\"ns_per_elem\":%.6f,\"ns_median\":%.6f,\"gops\":%.6f,"
                 "\"ops_per_elem\":%d,\"checksum\":%.6g}%s\n",
                 r.group.c_str(), r.name.c_str(), r.complexity_node.c_str(), r.ns_per_elem,
                 r.ns_median, r.gops, r.ops_per_elem, r.checksum,
                 i + 1 < rs.size() ? "," : "");
  }
  std::fprintf(f, "]}\n");
}

inline void print_table(const std::vector<BenchResult> &rs, std::FILE *f = stdout) {
  std::fprintf(f, "%-14s %-22s %-18s %12s %10s %12s\n", "group", "name", "complexity_node",
               "ns/elem", "GOPS", "checksum");
  for (const auto &r : rs)
    std::fprintf(f, "%-14s %-22s %-18s %12.4f %10.3f %12.5g\n", r.group.c_str(),
                 r.name.c_str(), r.complexity_node.c_str(), r.ns_per_elem, r.gops,
                 r.checksum);
}

} // namespace ee_bench
