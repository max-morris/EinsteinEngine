// Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Measures Relational + Piecewise nodes:
//  * Relational (a < b, a <= b, ...) -> emitted comparison operators.
//  * Piecewise -> emitted if_else(cond, then, else_) (IfElseExpr) or if/else Stmt.
//
// Cases: data-dependent branch (unpredictable), predictable all-true, and the
// branchless ternary the emitter actually generates (if_else). The branchless
// form is what Piecewise costs in generated code; the branching if/else form
// is included to quantify mispredict penalties for runtime conditionals.
// CUDA/ROCm build: both the ternary and the real if/else run as device kernels.
// Divergence is part of the if/else measurement.

#include <cstddef>
#include <vector>

#include "../common/ee_bench.hpp"
#include "../common/kernels.hpp"
#if defined(EE_DEVICE_BUILD)
#include "../common/ee_device.hpp"
namespace {
struct CmpOp {
  EE_HD_INLINE char operator()(ee_bench::vreal a, ee_bench::vreal b) const {
    return static_cast<char>(a < b);
  }
};
struct IfElseOp {
  EE_HD_INLINE ee_bench::vreal operator()(ee_bench::vreal a, ee_bench::vreal b,
                                          ee_bench::vreal c) const {
    return ee_kernels::ee_if_else(a < b, c, -c);
  }
};
struct IfBranchOp {
  EE_HD_INLINE ee_bench::vreal operator()(ee_bench::vreal a, ee_bench::vreal b,
                                          ee_bench::vreal c) const {
    if (a < b)
      return c;
    return -c;
  }
};
} // namespace
#endif

namespace {

using ee_bench::BenchConfig;
using ee_bench::BenchResult;
using ee_bench::vreal;

BenchResult run_compare(const BenchConfig &cfg) {
#if defined(EE_DEVICE_BUILD)
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), b(n);
  ee_bench::fill_inputs(a.data(), b.data(), n);
  return ee_device::time_compare(cfg, a.data(), b.data(), n, CmpOp{});
#else
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), b(n);
  std::vector<char> out(n);
  ee_bench::fill_inputs(a.data(), b.data(), n);
  for (int w = 0; w < cfg.warmup; ++w)
    for (std::size_t i = 0; i < n; ++i)
      out[i] = static_cast<char>(a[i] < b[i]);
  ee_escape(out.data());
  std::vector<double> runs;
  ee_bench::Timer t;
  for (int r = 0; r < cfg.repeats; ++r) {
    t.start();
    for (std::size_t i = 0; i < n; ++i)
      out[i] = static_cast<char>(a[i] < b[i]);
    ee_clobber(out);
    runs.push_back(t.stop_ns());
  }
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += out[i];
  return ee_bench::summarize("branch", "Relational", "cmp_lt_stream", runs, n, 1, sum);
#endif
}

BenchResult run_if_else(const BenchConfig &cfg) {
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), b(n), c(n), d(n);
  ee_bench::fill_inputs(a.data(), b.data(), n);
  ee_bench::fill_inputs(c.data(), d.data(), n);
#if defined(EE_DEVICE_BUILD)
  return ee_device::time_ternary("branch", "if_else_stream", "Piecewise", cfg, a.data(), b.data(),
                               c.data(), n, IfElseOp{});
#else
  for (int w = 0; w < cfg.warmup; ++w)
    for (std::size_t i = 0; i < n; ++i)
      d[i] = ee_kernels::ee_if_else(a[i] < b[i], c[i], -c[i]);
  ee_escape(d.data());
  std::vector<double> runs;
  ee_bench::Timer t;
  for (int r = 0; r < cfg.repeats; ++r) {
    t.start();
    for (std::size_t i = 0; i < n; ++i)
      d[i] = ee_kernels::ee_if_else(a[i] < b[i], c[i], -c[i]);
    ee_clobber(d);
    runs.push_back(t.stop_ns());
  }
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += static_cast<double>(d[i]);
  ee_escape(&sum);
  return ee_bench::summarize("branch", "Piecewise", "if_else_stream", runs, n, 1, sum);
#endif
}

BenchResult run_branching(const BenchConfig &cfg) {
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), b(n), c(n), d(n);
  ee_bench::fill_inputs(a.data(), b.data(), n);
  ee_bench::fill_inputs(c.data(), d.data(), n);
#if defined(EE_DEVICE_BUILD)
  return ee_device::time_ternary("branch", "if_else_branch_stream", "Piecewise[if/else]", cfg,
                               a.data(), b.data(), c.data(), n, IfBranchOp{});
#else
  for (int w = 0; w < cfg.warmup; ++w)
    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] < b[i])
        d[i] = c[i];
      else
        d[i] = -c[i];
    }
  ee_escape(d.data());
  std::vector<double> runs;
  ee_bench::Timer t;
  for (int r = 0; r < cfg.repeats; ++r) {
    t.start();
    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] < b[i])
        d[i] = c[i];
      else
        d[i] = -c[i];
    }
    ee_clobber(d);
    runs.push_back(t.stop_ns());
  }
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += static_cast<double>(d[i]);
  ee_escape(&sum);
  return ee_bench::summarize("branch", "Piecewise[if/else]", "if_else_branch_stream",
                             runs, n, 1, sum);
#endif
}

} // namespace

std::vector<BenchResult> bench_branch(const BenchConfig &cfg) {
  return {run_compare(cfg), run_if_else(cfg), run_branching(cfg)};
}
