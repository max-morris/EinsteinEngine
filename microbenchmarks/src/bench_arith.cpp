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
// CUDA/ROCm build: same expressions, launched from ee_device.hpp and timed with events.

#include <cstddef>
#include <vector>

#include "../common/ee_bench.hpp"
#include "../common/kernels.hpp"
#if defined(EE_DEVICE_BUILD)
#include "../common/ee_device.hpp"
#endif

namespace {

using ee_bench::BenchConfig;
using ee_bench::BenchResult;
using ee_bench::vreal;

template <typename Op>
BenchResult run_stream(const char *name, const char *node, const BenchConfig &cfg, Op op,
                       int ops_per_elem = 1) {
#if defined(EE_DEVICE_BUILD)
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), b(n);
  ee_bench::fill_inputs(a.data(), b.data(), n);
  return ee_device::time_binary("arith", name, node, cfg, a.data(), b.data(), n, op, ops_per_elem);
#else
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
#endif
}

// Chained accumulation; REPEAT>1 amplifies ALU cost so loop overhead is negligible.
template <typename Op>
BenchResult run_reg(const char *name, const char *node, const BenchConfig &cfg, Op op,
                    int ops_per_elem = 1) {
#if defined(EE_DEVICE_BUILD)
  (void)ops_per_elem;
  return ee_device::time_reg(name, node, cfg, op);
#else
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
#endif
}

} // namespace

#if defined(EE_DEVICE_BUILD)
namespace {
struct AddOp {
  EE_HD_INLINE ee_bench::vreal operator()(ee_bench::vreal a, ee_bench::vreal b) const {
    return ee_kernels::ee_add(a, b);
  }
};
struct SubOp {
  EE_HD_INLINE ee_bench::vreal operator()(ee_bench::vreal a, ee_bench::vreal b) const {
    return ee_kernels::ee_sub(a, b);
  }
};
struct MulOp {
  EE_HD_INLINE ee_bench::vreal operator()(ee_bench::vreal a, ee_bench::vreal b) const {
    return ee_kernels::ee_mul(a, b);
  }
};
struct DivOp {
  EE_HD_INLINE ee_bench::vreal operator()(ee_bench::vreal a, ee_bench::vreal b) const {
    return ee_kernels::ee_div(a, b);
  }
};
struct NegOp {
  EE_HD_INLINE ee_bench::vreal operator()(ee_bench::vreal a, ee_bench::vreal) const {
    return ee_kernels::ee_neg(a);
  }
};
} // namespace
#endif

std::vector<ee_bench::BenchResult> bench_arith(const ee_bench::BenchConfig &cfg) {
  using namespace ee_kernels;
  std::vector<BenchResult> out;
#if defined(EE_DEVICE_BUILD)
  out.push_back(run_stream("add_stream", "Add", cfg, AddOp{}));
  out.push_back(run_stream("sub_stream", "Add", cfg, SubOp{}));
  out.push_back(run_stream("mul_stream", "Mul", cfg, MulOp{}));
  out.push_back(run_stream("div_stream", "Mul[div]", cfg, DivOp{}));
  out.push_back(run_stream("neg_stream", "Mul[neg]", cfg, NegOp{}));
  out.push_back(run_reg("add_reg", "Add", cfg, AddOp{}));
  out.push_back(run_reg("mul_reg", "Mul", cfg, MulOp{}));
  out.push_back(run_reg("div_reg", "Mul[div]", cfg, DivOp{}));
#else
  out.push_back(run_stream("add_stream", "Add", cfg, ee_add));
  out.push_back(run_stream("sub_stream", "Add", cfg, ee_sub));
  out.push_back(run_stream("mul_stream", "Mul", cfg, ee_mul));
  out.push_back(run_stream("div_stream", "Mul[div]", cfg, ee_div));
  out.push_back(run_stream(
      "neg_stream", "Mul[neg]", cfg, [](vreal a, vreal) { return ee_neg(a); }));
  out.push_back(run_reg("add_reg", "Add", cfg, ee_add));
  out.push_back(run_reg("mul_reg", "Mul", cfg, ee_mul));
  out.push_back(run_reg("div_reg", "Mul[div]", cfg, ee_div));
#endif
  return out;
}
