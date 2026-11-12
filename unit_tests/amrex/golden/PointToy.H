// AMReX backend gate: point toy (toys.py)
#ifndef EE_POINTTOY_H_
#define EE_POINTTOY_H_

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Array.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Math.H>
#include <cmath>
#include "toy_enums.H"

namespace EE
{

// The recipe is written in three dimensions.
static_assert(AMREX_SPACEDIM == 3, "EE kernels are generated for AMREX_SPACEDIM == 3");

// rhs loop 0
// Outputs, in order: v_rhs.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rhs(
    int i,
    int j,
    int k,
    amrex::Array4<amrex::Real const> const & un,
    amrex::Array4<amrex::Real const> const & prim,
    amrex::GpuArray<amrex::Real, 3> const & prob_lo,
    amrex::GpuArray<amrex::Real, 3> const & dx,
    const ToyPar & parameters,
    amrex::Real & v_rhs)
{
    const amrex::Real ee_DXI = 1.0 / dx[0];
    const amrex::Real ee_DYI = 1.0 / dx[1];
    const amrex::Real ee_x = prob_lo[0] + i * dx[0];
    const amrex::Real ee_z = prob_lo[2] + k * dx[2];
    const amrex::Real ee_x2 = un(i - 2, j, k, vars::u);
    const amrex::Real ee_x3 = un(i + 2, j, k, vars::u);
    const amrex::Real ee_x4 = un(i - 1, j, k, vars::u);
    const amrex::Real ee_x5 = un(i + 1, j, k, vars::u);
    const amrex::Real ee_x0 = un(i, j, k, vars::u);
    v_rhs = (pow(ee_x0, -1.5) + (ee_DXI * (((1.0 / 12.0) * parameters.trK0 * ((8 * ((ee_x5 + (-(ee_x4))))) + ((ee_x2 + (-(ee_x3)))))) + ((1.0 / 144.0) * ee_DYI * ((-(((un(i - 2, j + 2, k, vars::u) + un(i + 2, j - 2, k, vars::u))))) + (-64 * ((un(i + 1, j - 1, k, vars::u) + un(i - 1, j + 1, k, vars::u)))) + (-8 * ((un(i + 1, j + 2, k, vars::u) + un(i + 2, j + 1, k, vars::u)))) + (-8 * ((un(i - 1, j - 2, k, vars::u) + un(i - 2, j - 1, k, vars::u)))) + (8 * ((un(i + 1, j - 2, k, vars::u) + un(i + 2, j - 1, k, vars::u)))) + (8 * ((un(i - 1, j + 2, k, vars::u) + un(i - 2, j + 1, k, vars::u)))) + (64 * ((un(i + 1, j + 1, k, vars::u) + un(i - 1, j - 1, k, vars::u)))) + ((un(i - 2, j - 2, k, vars::u) + un(i + 2, j + 2, k, vars::u))))))) + (prim(i, j, k, pvars::Df) * un(i, j, k, vars::trK)) + (ee_x * (ee_z * ee_z)));
}

// ko loop 0
// Outputs, in order: ko_u.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void ko(
    int i,
    int j,
    int k,
    amrex::Array4<amrex::Real const> const & un,
    [[maybe_unused]] amrex::Array4<amrex::Real const> const & prim,
    [[maybe_unused]] amrex::GpuArray<amrex::Real, 3> const & prob_lo,
    amrex::GpuArray<amrex::Real, 3> const & dx,
    const ToyPar & parameters,
    amrex::Real & ko_u)
{
    const amrex::Real ee_DXI = 1.0 / dx[0];
    const amrex::Real ee_DYI = 1.0 / dx[1];
    const amrex::Real ee_DZI = 1.0 / dx[2];
    const amrex::Real ee_x2 = un(i - 2, j, k, vars::u);
    const amrex::Real ee_x3 = un(i + 2, j, k, vars::u);
    const amrex::Real ee_x4 = un(i - 1, j, k, vars::u);
    const amrex::Real ee_x5 = un(i + 1, j, k, vars::u);
    const amrex::Real ee_x0 = un(i, j, k, vars::u);
    const amrex::Real ee_x1 = (30 * ee_x0);
    ko_u = ((parameters.eps > 0) ? amrex::Real(((-1.0 / 12.0) * parameters.eps * (((ee_DXI * ee_DXI) * (ee_x1 + (-16 * ((ee_x4 + ee_x5))) + ((ee_x2 + ee_x3)))) + ((ee_DYI * ee_DYI) * (ee_x1 + (-16 * ((un(i, j + 1, k, vars::u) + un(i, j - 1, k, vars::u)))) + ((un(i, j - 2, k, vars::u) + un(i, j + 2, k, vars::u))))) + ((ee_DZI * ee_DZI) * (ee_x1 + (-16 * ((un(i, j, k + 1, vars::u) + un(i, j, k - 1, vars::u)))) + ((un(i, j, k - 2, vars::u) + un(i, j, k + 2, vars::u)))))))) : amrex::Real(0));
}

} // namespace EE

#endif /* EE_POINTTOY_H_ */
