// Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Measures Add / Mul nodes: emitted as (a + b), (a - b), (a * b), (a / b),
// (-(x))  (see cpp_carpetx_visitor.py NArityOpExpr / UnOpExpr).
//
// Two modes per op:
//  * *_stream: c[i] = op(a[i], b[i]) — grid-loop-like, includes memory traffic.
//  * *_reg:    x = op(x, k) chained accumulation — isolates ALU throughput.
//
// GPU port: the inner scalar expression (ee_kernels::ee_*) is shared; a CUDA/HIP
// driver would wrap the same expression in a __global__ kernel over i and time
// with events. Skeleton:
//
//   template <typename Op>
//   __global__ void ee_stream_kernel(const vreal* a, const vreal* b, vreal* c,
//                                    std::size_t n, Op op) {
//     std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) c[i] = op(a[i], b[i]);
//   }

#include <cstddef>
#include <vector>

#include "../common/ee_bench.hpp"
#include "../common/kernels.hpp"

namespace {

using ee_bench::BenchConfig;
using ee_bench::BenchResult;
using ee_bench::vreal;

template <typename Op>
BenchResult run_stream(const char *name, const char *node, const BenchConfig &cfg, Op op,
                       int ops_per_elem = 1) {
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), b(n), c(n);
  ee_bench::fill_inputs(a.data(), b.data(), n);
  for (int w = 0; w < cfg.warmup; ++w)
    for (std::size_t i = 0; i < n; ++i)
      c[i] = op(a[i], b[i]);
  ee_escape(c.data());

  std::vector<double> runs;
  ee_bench::Timer t;
  for (int r = 0; r < cfg.repeats; ++r) {
    t.start();
    for (std::size_t i = 0; i < n; ++i)
      c[i] = op(a[i], b[i]);
    ee_clobber(c);
    runs.push_back(t.stop_ns());
  }
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += static_cast<double>(c[i]);
  ee_escape(&sum);
  return ee_bench::summarize("arith", node, name, runs, n, ops_per_elem, sum);
}

// Chained accumulation; REPEAT>1 amplifies ALU cost so loop overhead is negligible.
template <typename Op>
BenchResult run_reg(const char *name, const char *node, const BenchConfig &cfg, Op op,
                    int ops_per_elem = 1) {
  constexpr int REPEAT = 16;
  std::size_t iters = cfg.quick ? (1u << 20) : (1u << 24);
  vreal x = static_cast<vreal>(1.000001);
  const vreal k = static_cast<vreal>(1.000002);
  for (std::size_t w = 0; w < 2; ++w) {
    for (std::size_t i = 0; i < iters / 32; ++i)
      for (int j = 0; j < 32; ++j)
        x = op(x, k);
  }
  std::vector<double> runs;
  ee_bench::Timer t;
  for (int r = 0; r < cfg.repeats; ++r) {
    t.start();
    for (std::size_t i = 0; i < iters / REPEAT; ++i) {
      for (int j = 0; j < REPEAT; ++j)
        x = op(x, k);
    }
    ee_clobber(x);
    runs.push_back(t.stop_ns());
  }
  double sum = static_cast<double>(x);
  ee_escape(&sum);
  // Each inner step is one measured op; total elements = iters.
  std::vector<double> scaled;
  for (double v : runs)
    scaled.push_back(v);
  BenchResult r =
      ee_bench::summarize("arith", node, name, scaled, iters, ops_per_elem, sum);
  return r;
}

} // namespace

std::vector<BenchResult> bench_arith(const BenchConfig &cfg) {
  using namespace ee_kernels;
  std::vector<BenchResult> out;
  out.push_back(run_stream("add_stream", "Add", cfg, ee_add));
  out.push_back(run_stream("sub_stream", "Add", cfg, ee_sub));
  out.push_back(run_stream("mul_stream", "Mul", cfg, ee_mul));
  out.push_back(run_stream("div_stream", "Mul[div]", cfg, ee_div));
  out.push_back(run_stream(
      "neg_stream", "Mul[neg]", cfg, [](vreal a, vreal) { return ee_neg(a); }));
  out.push_back(run_reg("add_reg", "Add", cfg, ee_add));
  out.push_back(run_reg("mul_reg", "Mul", cfg, ee_mul));
  out.push_back(run_reg("div_reg", "Mul[div]", cfg, ee_div));
  return out;
}
