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
  const std::size_t n = cfg.n_stream;
  std::vector<vreal> a(n), b(n);
  ee_bench::fill_inputs(a.data(), b.data(), n);
#if defined(EE_DEVICE_BUILD)
  return ee_device::time_binary("arith", name, node, cfg, a.data(), b.data(), n, op, ops_per_elem);
#else
  return ee_bench::host_time_stream("arith", name, node, cfg, a.data(), b.data(), n, op, ops_per_elem);
#endif
}

// Carried chains x = op(x, k) in registers: no loads, no stores, no sink, so nothing but the op is timed. This is the
// reference the fitter divides by (add_reg). The host runs CHAINS independent chains, enough to cover the op's
// latency, so this is throughput, as in an emitted expression; the device runs one chain per thread, and the many
// threads cover it. x is reset at every repeat: a carried multiply by 1.000002 must not overflow, nor a carried
// divide drift into the subnormals.
template <typename Op>
BenchResult run_reg(const char *name, const char *node, const BenchConfig &cfg, Op op,
                    int ops_per_elem = 1) {
#if defined(EE_DEVICE_BUILD)
  (void)ops_per_elem;
  return ee_device::time_reg(name, node, cfg, op);
#else
  constexpr int CHAINS = 8;
  constexpr int REPEAT = 16;
  const std::size_t iters = cfg.quick ? (1u << 20) : (1u << 24);
  const std::size_t blocks = iters / (CHAINS * REPEAT);
  const vreal k = static_cast<vreal>(1.000002);
  vreal x[CHAINS];
  auto pass = [&] {
    for (int c = 0; c < CHAINS; ++c)
      x[c] = static_cast<vreal>(1.000001 + 1.0e-3 * c);
    for (std::size_t i = 0; i < blocks; ++i) {
      for (int j = 0; j < REPEAT; ++j)
        for (int c = 0; c < CHAINS; ++c)
          x[c] = op(x[c], k);
      // Keeps the chains scalar: the compiler must not pack them into SIMD lanes.
      for (int c = 0; c < CHAINS; ++c)
        ee_clobber(x[c]);
    }
  };
  for (int w = 0; w < cfg.warmup; ++w)
    pass();
  std::vector<double> runs;
  ee_bench::Timer t;
  for (int r = 0; r < cfg.repeats; ++r) {
    t.start();
    pass();
    runs.push_back(t.stop_ns());
  }
  double sum = 0;
  for (int c = 0; c < CHAINS; ++c)
    sum += static_cast<double>(x[c]);
  ee_escape(&sum);
  return ee_bench::summarize("arith", node, name, runs, blocks * CHAINS * REPEAT, ops_per_elem, sum);
#endif
}

} // namespace

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
// Streamed kernels call op(base + k*eps, other) and then pay a finiteness
// check and an xor sink. This functor returns that perturbed input so the same
// harness can time the overhead alone.
struct IdentityOp {
  EE_HD_INLINE ee_bench::vreal operator()(ee_bench::vreal a, ee_bench::vreal) const { return a; }
};
} // namespace

std::vector<ee_bench::BenchResult> bench_arith(const ee_bench::BenchConfig &cfg) {
  std::vector<BenchResult> out;
  out.push_back(run_stream("identity_stream", "identity", cfg, IdentityOp{}));
  out.push_back(run_stream("add_stream", "Add", cfg, AddOp{}));
  out.push_back(run_stream("sub_stream", "Add", cfg, SubOp{}));
  out.push_back(run_stream("mul_stream", "Mul", cfg, MulOp{}));
  out.push_back(run_stream("div_stream", "Mul[div]", cfg, DivOp{}));
  out.push_back(run_stream("neg_stream", "Mul[neg]", cfg, NegOp{}));
  out.push_back(run_reg("add_reg", "Add", cfg, AddOp{}));
  out.push_back(run_reg("mul_reg", "Mul", cfg, MulOp{}));
  out.push_back(run_reg("div_reg", "Mul[div]", cfg, DivOp{}));
  return out;
}
