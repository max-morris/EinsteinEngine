// Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Measures the transcendental calls weighted 15 in sympy_complexity.py:
// sin, cos, exp, log, sqrt, cbrt — plus the sibling calls the emitter may
// produce (tan, sinh, cosh, tanh, erf) so future weight splits have data.
//
// Inputs: angles in [-0.5, 0.5] for trig/hyperbolic (no range-reduction noise),
// (0.5, 1.5) for exp/log/sqrt/cbrt (safe domains).
// GPU port: identical scalar calls; all exist in CUDA/HIP device libm.

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
                       bool angle_input) {
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), c(n);
  if (angle_input)
    ee_bench::fill_small_angles(a.data(), n);
  else {
    std::vector<vreal> tmp(n);
    ee_bench::fill_inputs(a.data(), tmp.data(), n);
  }
  for (int w = 0; w < cfg.warmup; ++w)
    for (std::size_t i = 0; i < n; ++i)
      c[i] = op(a[i]);
  ee_escape(c.data());
  std::vector<double> runs;
  ee_bench::Timer t;
  for (int r = 0; r < cfg.repeats; ++r) {
    t.start();
    for (std::size_t i = 0; i < n; ++i)
      c[i] = op(a[i]);
    ee_clobber(c);
    runs.push_back(t.stop_ns());
  }
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += static_cast<double>(c[i]);
  ee_escape(&sum);
  return ee_bench::summarize("transcendental", node, name, runs, n, 1, sum);
}

} // namespace

std::vector<BenchResult> bench_transcendental(const BenchConfig &cfg) {
  using namespace ee_kernels;
  std::vector<BenchResult> out;
  out.push_back(run_stream("sin_stream", "sin", cfg, ee_sin, true));
  out.push_back(run_stream("cos_stream", "cos", cfg, ee_cos, true));
  out.push_back(run_stream("tan_stream", "tan", cfg, ee_tan, true));
  out.push_back(run_stream("sinh_stream", "sinh", cfg, ee_sinh, true));
  out.push_back(run_stream("cosh_stream", "cosh", cfg, ee_cosh, true));
  out.push_back(run_stream("tanh_stream", "tanh", cfg, ee_tanh, true));
  out.push_back(run_stream("exp_stream", "exp", cfg, ee_exp, false));
  out.push_back(run_stream("log_stream", "log", cfg, ee_log, false));
  out.push_back(run_stream("erf_stream", "erf", cfg, ee_erf, true));
  out.push_back(run_stream("sqrt_stream", "sqrt", cfg, ee_sqrt, false));
  out.push_back(run_stream("cbrt_stream", "cbrt", cfg, ee_cbrt, false));
  return out;
}
