// Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Measures Pow nodes exactly as emitted (cpp_carpetx_visitor.py BinOpExpr):
//  * x**2        -> pow2(x)
//  * x**N (int)  -> pown<vreal>(x, N)
//  * x**(1/2)    -> sqrt(x)
//  * x**(1/3)    -> cbrt(x)
//  * otherwise   -> pow(vreal(x), y), incl. non-integer constant exponents.
//
// Guestimate under test (sympy_complexity.py): default c=15, integer
// c=max(2, log2(|p|)). We measure N in {2,3,4,8,16} plus generic pow with a
// variable exponent, pow(x, 2.5), sqrt, cbrt.
//
// GPU port: same expressions inside a __global__ kernel; pow/sqrt/cbrt all
// exist in CUDA/HIP device libm.

#include <cstddef>
#include <vector>

#include "../common/ee_bench.hpp"
#include "../common/kernels.hpp"

namespace {

using ee_bench::BenchConfig;
using ee_bench::BenchResult;
using ee_bench::vreal;

template <typename Op>
BenchResult run_stream(const char *name, const char *node, const BenchConfig &cfg, Op op) {
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), b(n), c(n);
  ee_bench::fill_inputs(a.data(), b.data(), n);
  // Keep variable exponents in a safe range for pow().
  for (std::size_t i = 0; i < n; ++i)
    b[i] = static_cast<vreal>(0.5 + (b[i] - 0.25));
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
  return ee_bench::summarize("pow", node, name, runs, n, 1, sum);
}

} // namespace

std::vector<BenchResult> bench_pow(const BenchConfig &cfg) {
  using namespace ee_kernels;
  std::vector<BenchResult> out;
  out.push_back(run_stream("pow2_stream", "Pow[x**2]", cfg,
                           [](vreal a, vreal) { return ee_pow2(a); }));
  out.push_back(run_stream("pown3_stream", "Pow[x**3]", cfg,
                           [](vreal a, vreal) { return ee_pown(a, 3); }));
  out.push_back(run_stream("pown4_stream", "Pow[x**4]", cfg,
                           [](vreal a, vreal) { return ee_pown(a, 4); }));
  out.push_back(run_stream("pown8_stream", "Pow[x**8]", cfg,
                           [](vreal a, vreal) { return ee_pown(a, 8); }));
  out.push_back(run_stream("pown16_stream", "Pow[x**16]", cfg,
                           [](vreal a, vreal) { return ee_pown(a, 16); }));
  out.push_back(run_stream("sqrt_stream", "Pow[sqrt]", cfg,
                           [](vreal a, vreal) { return ee_sqrt(a); }));
  out.push_back(run_stream("cbrt_stream", "Pow[cbrt]", cfg,
                           [](vreal a, vreal) { return ee_cbrt(a); }));
  out.push_back(run_stream("powvar_stream", "Pow[generic]", cfg,
                           [](vreal a, vreal b) { return ee_pow_generic(a, b); }));
  out.push_back(run_stream("pow2p5_stream", "Pow[x**2.5]", cfg,
                           [](vreal a, vreal) {
                             return ee_pow_generic(a, static_cast<vreal>(2.5));
                           }));
  return out;
}
