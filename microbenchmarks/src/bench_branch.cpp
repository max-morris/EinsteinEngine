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
// GPU port: prefer the if_else (ternary) case — GPUs execute both sides under
// divergence; the branchless timing is the portable number.

#include <cstddef>
#include <vector>

#include "../common/ee_bench.hpp"
#include "../common/kernels.hpp"

namespace {

using ee_bench::BenchConfig;
using ee_bench::BenchResult;
using ee_bench::vreal;

BenchResult run_compare(const BenchConfig &cfg) {
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
}

BenchResult run_if_else(const BenchConfig &cfg) {
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), b(n), c(n), d(n);
  ee_bench::fill_inputs(a.data(), b.data(), n);
  ee_bench::fill_inputs(c.data(), d.data(), n);
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
}

BenchResult run_branching(const BenchConfig &cfg) {
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), b(n), c(n), d(n);
  ee_bench::fill_inputs(a.data(), b.data(), n);
  ee_bench::fill_inputs(c.data(), d.data(), n);
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
}

} // namespace

std::vector<BenchResult> bench_branch(const BenchConfig &cfg) {
  return {run_compare(cfg), run_if_else(cfg), run_branching(cfg)};
}
