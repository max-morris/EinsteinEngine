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

// Measured kernels, mirroring exactly what the CarpetX C++ emitter generates
// (see EinsteinEngine/emit/code/cpp_carpetx/cpp_carpetx_visitor.py):
//
//  * Add/Mul      -> (a + b), (a * b), (-(x)), (a / b)
//  * Pow          -> pow2(x) for x**2, pown<vreal>(x,N) for int N,
//                    sqrt(x) for x**(1/2), cbrt(x) for x**(1/3),
//                    pow(vreal(x), y) otherwise
//  * Transcendentals -> sin/cos/tan/sinh/cosh/tanh/exp/erf/log(...)
//  * Piecewise    -> if_else(cond, then, else)
//  * Grid loads   -> access(gf, idx) / stencil(gf,i,j,k) index arithmetic
//
// GPU-portability rules for this file:
//  * Only EE_HD_INLINE scalar functions + index-based functors.
//  * <cmath> scalar calls only (all exist in CUDA/HIP device code).
//  * No <chrono>, no printf, no std::vector, no exceptions here.
//  * Host drivers in src/*.cpp time these. The CUDA and ROCm builds launch the
//    same functors from __global__ kernels (see ee_device.hpp).

#pragma once

#include <cmath>

#include "ee_bench.hpp" // EE_HD_INLINE, ee_bench::vreal

namespace ee_kernels {

using ee_bench::vreal;

// --- codegen mirrors -------------------------------------------------------

EE_HD_INLINE vreal ee_add(vreal a, vreal b) { return a + b; }
EE_HD_INLINE vreal ee_sub(vreal a, vreal b) { return a - b; }
EE_HD_INLINE vreal ee_mul(vreal a, vreal b) { return a * b; }
EE_HD_INLINE vreal ee_div(vreal a, vreal b) { return a / b; }
EE_HD_INLINE vreal ee_neg(vreal a) { return -a; }

// Emitted as pow2(x) for x**2.
EE_HD_INLINE vreal ee_pow2(vreal x) { return x * x; }

// Emitted as pown<vreal>(x, N) for integer exponents.
EE_HD_INLINE vreal ee_pown(vreal x, int n) {
  vreal r = static_cast<vreal>(1);
  for (int i = 0; i < n; ++i)
    r *= x;
  return r;
}

EE_HD_INLINE vreal ee_sqrt(vreal x) { return std::sqrt(x); }
EE_HD_INLINE vreal ee_cbrt(vreal x) { return std::cbrt(x); }
EE_HD_INLINE vreal ee_pow_generic(vreal x, vreal y) { return std::pow(x, y); }

EE_HD_INLINE vreal ee_sin(vreal x) { return std::sin(x); }
EE_HD_INLINE vreal ee_cos(vreal x) { return std::cos(x); }
EE_HD_INLINE vreal ee_tan(vreal x) { return std::tan(x); }
EE_HD_INLINE vreal ee_sinh(vreal x) { return std::sinh(x); }
EE_HD_INLINE vreal ee_cosh(vreal x) { return std::cosh(x); }
EE_HD_INLINE vreal ee_tanh(vreal x) { return std::tanh(x); }
EE_HD_INLINE vreal ee_exp(vreal x) { return std::exp(x); }
EE_HD_INLINE vreal ee_log(vreal x) { return std::log(x); }
EE_HD_INLINE vreal ee_erf(vreal x) { return std::erf(x); }

// Emitted for Piecewise as if_else(cond, then, else_).
EE_HD_INLINE vreal ee_if_else(bool c, vreal t, vreal e) { return c ? t : e; }

// Grid-index helpers mirroring stencil(gf,i,j,k) / access(gf, idx).
// Linear index for an (nx * ny * nz) grid with ghost zones included.
EE_HD_INLINE std::size_t ee_lin(std::size_t i, std::size_t j, std::size_t k, std::size_t nx,
                                std::size_t ny) {
  return (k * ny + j) * nx + i;
}

} // namespace ee_kernels
