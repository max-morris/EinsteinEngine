// Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Measures the transcendental calls named in the active weight profile:
// sin, cos, exp, log, sqrt, cbrt — plus the sibling calls the emitter may
// produce (tan, sinh, cosh, tanh, erf) so future weight splits have data.
//
// Inputs: angles in [-0.5, 0.5] for trig/hyperbolic (no range-reduction noise),
// (0.5, 1.5) for exp/log/sqrt/cbrt (safe domains).
// CUDA/ROCm build: identical scalar calls, launched from ee_device.hpp.

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
                       bool angle_input) {
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n);
  if (angle_input)
    ee_bench::fill_small_angles(a.data(), n);
  else {
    std::vector<vreal> tmp(n);
    ee_bench::fill_inputs(a.data(), tmp.data(), n);
  }
#if defined(EE_DEVICE_BUILD)
  return ee_device::time_unary("transcendental", name, node, cfg, a.data(), n, op);
#else
  std::vector<vreal> c(n);
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
#endif
}

} // namespace

namespace {
struct SinOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_sin(a); }
};
struct CosOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_cos(a); }
};
struct TanOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_tan(a); }
};
struct SinhOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_sinh(a); }
};
struct CoshOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_cosh(a); }
};
struct TanhOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_tanh(a); }
};
struct ExpOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_exp(a); }
};
struct LogOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_log(a); }
};
struct ErfOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_erf(a); }
};
struct SqrtOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_sqrt(a); }
};
struct CbrtOp {
  EE_HD_INLINE vreal operator()(vreal a) const { return ee_kernels::ee_cbrt(a); }
};
} // namespace

std::vector<BenchResult> bench_transcendental(const BenchConfig &cfg) {
  std::vector<BenchResult> out;
  out.push_back(run_stream("sin_stream", "sin", cfg, SinOp{}, true));
  out.push_back(run_stream("cos_stream", "cos", cfg, CosOp{}, true));
  out.push_back(run_stream("tan_stream", "tan", cfg, TanOp{}, true));
  out.push_back(run_stream("sinh_stream", "sinh", cfg, SinhOp{}, true));
  out.push_back(run_stream("cosh_stream", "cosh", cfg, CoshOp{}, true));
  out.push_back(run_stream("tanh_stream", "tanh", cfg, TanhOp{}, true));
  out.push_back(run_stream("exp_stream", "exp", cfg, ExpOp{}, false));
  out.push_back(run_stream("log_stream", "log", cfg, LogOp{}, false));
  out.push_back(run_stream("erf_stream", "erf", cfg, ErfOp{}, true));
  out.push_back(run_stream("sqrt_stream", "sqrt", cfg, SqrtOp{}, false));
  out.push_back(run_stream("cbrt_stream", "cbrt", cfg, CbrtOp{}, false));
  return out;
}
