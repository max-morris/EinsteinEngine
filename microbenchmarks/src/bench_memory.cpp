// Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Measures memory-ish weights:
//  * Symbol grid=10 vs local=1: array load (grid function) vs register-held local.
//  * stencil(sym,x,y,z): center=10, x-offset=40, y/z-offset=100.
//  * Indexed/Number/Atom (=1): baseline copy of a compile-time constant.
//
// Method: 3-D grid (nx*ny*nz, default 128^3 interior-ish with ghosts folded in)
// of vreal. Cases stream over the interior and load:
//  * center:      c = g[c]
//  * x-neighbor:  c = g[c] + g[c+1]     (same/contiguous cache line)
//  * y-neighbor:  c = g[c] + g[c+nx]    (one row stride away)
//  * z-neighbor:  c = g[c] + g[c+nx*ny] (one plane stride away)
//  * local:       register accumulator, no loads (reference for Symbol local=1)
//  * constant:    c = 1.0 + a[i]*0 (baseline Atom=1)
// The x/y/z cost ratios are the data behind the 10/40/100 stencil guestimates.
//
// GPU port: same index arithmetic (ee_lin) inside a __global__ kernel with one
// thread per interior point; time with events instead of chrono. Coalescing and
// cache behavior will differ from CPU — that is exactly why the weights must be
// re-measured per device (see README).

#include <cstddef>
#include <vector>

#include "../common/ee_bench.hpp"
#include "../common/kernels.hpp"

namespace {

using ee_bench::BenchConfig;
using ee_bench::BenchResult;
using ee_bench::vreal;

struct Grid {
  std::size_t nx = 128, ny = 128, nz = 128;
  std::size_t n() const { return nx * ny * nz; }
};

template <typename Op>
BenchResult run_grid(const char *name, const char *node, const BenchConfig &cfg, Grid g,
                     Op op) {
  const std::size_t n = g.n();
  std::vector<vreal> gf(n), out(n);
  std::vector<vreal> scratch(n);
  ee_bench::fill_inputs(gf.data(), scratch.data(), n);
  const std::size_t nx = g.nx, ny = g.ny, nz = g.nz;
  // Warmup (untimed), restricted to interior so offsets stay in bounds.
  for (int w = 0; w < cfg.warmup; ++w)
    for (std::size_t k = 1; k + 1 < nz; ++k)
      for (std::size_t j = 1; j + 1 < ny; ++j)
        for (std::size_t i = 1; i + 1 < nx; ++i) {
          const std::size_t c = ee_kernels::ee_lin(i, j, k, nx, ny);
          out[c] = op(gf.data(), out.data(), c, nx, ny);
        }
  ee_escape(out.data());
  std::vector<double> runs;
  ee_bench::Timer t;
  std::size_t n_eval = 0;
  for (int r = 0; r < cfg.repeats; ++r) {
    n_eval = 0;
    t.start();
    for (std::size_t k = 1; k + 1 < nz; ++k)
      for (std::size_t j = 1; j + 1 < ny; ++j)
        for (std::size_t i = 1; i + 1 < nx; ++i) {
          const std::size_t c = ee_kernels::ee_lin(i, j, k, nx, ny);
          out[c] = op(gf.data(), out.data(), c, nx, ny);
          ++n_eval;
        }
    ee_clobber(out);
    runs.push_back(t.stop_ns());
  }
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += static_cast<double>(out[i]);
  ee_escape(&sum);
  return ee_bench::summarize("memory", node, name, runs, n_eval, 1, sum);
}

} // namespace

std::vector<BenchResult> bench_memory(const BenchConfig &cfg) {
  using namespace ee_kernels;
  Grid g;
  if (cfg.quick) {
    g.nx = 64;
    g.ny = 64;
    g.nz = 64;
  }
  std::vector<BenchResult> out;
  // Baseline: constant (Atom=1) and pure-local register update (Symbol local=1).
  out.push_back(run_grid(
      "const_stream", "Atom", cfg, g,
      [](const vreal *, const vreal *, std::size_t, std::size_t, std::size_t) {
        return static_cast<vreal>(1.0);
      }));
  // Grid center load: Symbol grid + stencil center (weight 10).
  out.push_back(run_grid("center_stream", "stencil[center]", cfg, g,
                         [](const vreal *gf, const vreal *, std::size_t c, std::size_t,
                            std::size_t) { return gf[c]; }));
  // Grid load used arithmetically: Symbol grid=10 reference.
  out.push_back(run_grid(
      "grid_add_stream", "Symbol[grid]", cfg, g,
      [](const vreal *gf, const vreal *, std::size_t c, std::size_t, std::size_t) {
        return gf[c] + static_cast<vreal>(1.0);
      }));
  // x-neighbor (weight 40): contiguous line.
  out.push_back(run_grid(
      "stencil_x_stream", "stencil[x]", cfg, g,
      [](const vreal *gf, const vreal *, std::size_t c, std::size_t, std::size_t) {
        return gf[c] + gf[c + 1];
      }));
  // y-neighbor (weight 100): row stride.
  out.push_back(run_grid(
      "stencil_y_stream", "stencil[y]", cfg, g,
      [](const vreal *gf, const vreal *, std::size_t c, std::size_t nx, std::size_t) {
        return gf[c] + gf[c + nx];
      }));
  // z-neighbor (weight 100): plane stride.
  out.push_back(run_grid(
      "stencil_z_stream", "stencil[z]", cfg, g,
      [](const vreal *gf, const vreal *, std::size_t c, std::size_t nx, std::size_t ny) {
        return gf[c] + gf[c + nx * ny];
      }));
  return out;
}
