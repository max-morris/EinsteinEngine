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
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
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

// The driver runs the whole suite `rounds` times, so that every case is timed in every round, close in time to add.
// A change in clock speed then affects a case and add alike, and the fitter divides within a round.
struct BenchConfig {
  std::size_t n_stream = 1u << 14; // elements per streaming pass; on the host, small enough to stay in cache
  int rounds = 10;                 // passes over the whole suite
  int repeats = 3;                 // timed repeats per case in each round (the min is the round's time)
  int warmup = 2;                  // untimed warmup passes per case in each round
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
  c.rounds = 5;
#endif
  return c;
}

inline BenchConfig default_config_quick() {
  BenchConfig c;
#if defined(EE_HAVE_CUDA) || defined(EE_HAVE_HIP)
  c.n_stream = std::size_t{1} << 24;
#else
  c.n_stream = std::size_t{1} << 12;
#endif
  c.rounds = 2;
  c.repeats = 2;
  c.warmup = 1;
  c.quick = true;
  return c;
}

struct BenchResult {
  std::string group;       // e.g. "arith", "pow", "transcendental", "memory", "branch"
  std::string name;        // e.g. "add_stream", "pow2_reg"
  std::string complexity_node; // sympy node this measures, e.g. "Add", "Pow[x**2]", "stencil[x]"
  double ns_per_elem = 0.0;    // min over repeats (and, once combined, over rounds)
  double ns_median = 0.0;      // median over repeats (once combined: median of the round minima)
  std::vector<double> ns_rounds; // each round's min, once combined
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

// The host counterpart of the device streamed kernels (ee_device.hpp). One op per loaded element would time the
// loads and the loop, not the op, so each element repeats the op `reps` times on a slightly perturbed input:
// independent iterations, so this is throughput, as in an emitted expression. The xor sink keeps every result live
// without another floating-point op. identity_stream times this harness alone, for the fitter to subtract.
inline int stream_reps(const BenchConfig &cfg) { return cfg.quick ? 32 : 256; }

inline std::uint64_t host_bits(double x) {
  std::uint64_t u;
  std::memcpy(&u, &x, sizeof u);
  return u;
}
inline std::uint64_t host_bits(float x) {
  std::uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return u;
}

template <typename F>
inline vreal host_repeat_sink(int reps, F body) {
  std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  bool bad = false;
  int k = 0;
  for (; k + 3 < reps; k += 4) {
    const vreal z0 = body(k + 0), z1 = body(k + 1), z2 = body(k + 2), z3 = body(k + 3);
    bad |= !std::isfinite(z0) | !std::isfinite(z1) | !std::isfinite(z2) | !std::isfinite(z3);
    s0 ^= host_bits(z0);
    s1 ^= host_bits(z1);
    s2 ^= host_bits(z2);
    s3 ^= host_bits(z3);
  }
  for (; k < reps; ++k) {
    const vreal z = body(k);
    bad |= !std::isfinite(z);
    s0 ^= host_bits(z);
  }
  if (bad)
    return std::numeric_limits<vreal>::quiet_NaN();
  // The low bits only: a finite value that still depends on every result.
  return static_cast<vreal>((s0 ^ s1 ^ s2 ^ s3) & 0xffffu);
}

// Time `op` on the host, as above. A two-input op reads its second input from `b`; a one-input op ignores it.
template <typename Op>
BenchResult host_time_stream(const char *group, const char *name, const char *node, const BenchConfig &cfg,
                             const vreal *a, const vreal *b, std::size_t n, Op op, int ops_per_elem = 1) {
  const int reps = stream_reps(cfg);
  constexpr vreal eps = static_cast<vreal>(1.0e-8);
  std::vector<vreal> c(n);
  auto pass = [&] {
    for (std::size_t i = 0; i < n; ++i) {
      const vreal base = a[i];
      if constexpr (std::is_invocable_v<Op, vreal, vreal>) {
        const vreal other = b[i];
        c[i] = host_repeat_sink(reps, [&](int k) { return op(base + static_cast<vreal>(k) * eps, other); });
      } else {
        c[i] = host_repeat_sink(reps, [&](int k) { return op(base + static_cast<vreal>(k) * eps); });
      }
    }
    ee_clobber(c);
  };
  for (int w = 0; w < cfg.warmup; ++w)
    pass();
  std::vector<double> runs;
  Timer t;
  for (int r = 0; r < cfg.repeats; ++r) {
    t.start();
    pass();
    runs.push_back(t.stop_ns());
  }
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += static_cast<double>(c[i]);
  ee_escape(&sum);
  return summarize(group, node, name, runs, n * static_cast<std::size_t>(reps), ops_per_elem, sum);
}

// Combine the suite's results from every round (the same cases in the same order each round) into one result per case.
inline std::vector<BenchResult> combine_rounds(const std::vector<std::vector<BenchResult>> &rounds) {
  std::vector<BenchResult> out = rounds.front();
  for (std::size_t i = 0; i < out.size(); ++i) {
    BenchResult &r = out[i];
    r.ns_rounds.clear();
    std::size_t best = 0;
    for (std::size_t k = 0; k < rounds.size(); ++k) {
      const BenchResult &rk = rounds[k][i];
      if (rk.group != r.group || rk.name != r.name) {
        std::fprintf(stderr, "FATAL: round %zu ran %s/%s where round 0 ran %s/%s\n", k,
                     rk.group.c_str(), rk.name.c_str(), r.group.c_str(), r.name.c_str());
        std::abort();
      }
      r.ns_rounds.push_back(rk.ns_per_elem);
      if (rk.ns_per_elem < rounds[best][i].ns_per_elem)
        best = k;
    }
    std::vector<double> sorted = r.ns_rounds;
    std::sort(sorted.begin(), sorted.end());
    r.ns_per_elem = rounds[best][i].ns_per_elem;
    r.gops = rounds[best][i].gops;
    r.ns_median = sorted[sorted.size() / 2];
    r.checksum = rounds.back()[i].checksum;
  }
  return out;
}

inline void print_json(const std::vector<BenchResult> &rs, std::FILE *f = stdout) {
  std::fprintf(f, "{\"results\":[\n");
  for (std::size_t i = 0; i < rs.size(); ++i) {
    const auto &r = rs[i];
    std::fprintf(f,
                 "  {\"group\":\"%s\",\"name\":\"%s\",\"complexity_node\":\"%s\","
                 "\"ns_per_elem\":%.6g,\"ns_median\":%.6g,\"ns_rounds\":[",
                 r.group.c_str(), r.name.c_str(), r.complexity_node.c_str(), r.ns_per_elem,
                 r.ns_median);
    for (std::size_t k = 0; k < r.ns_rounds.size(); ++k)
      std::fprintf(f, "%s%.6g", k ? "," : "", r.ns_rounds[k]);
    std::fprintf(f, "],\"gops\":%.6f,\"ops_per_elem\":%d,\"checksum\":%.6g}%s\n", r.gops,
                 r.ops_per_elem, r.checksum, i + 1 < rs.size() ? "," : "");
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
