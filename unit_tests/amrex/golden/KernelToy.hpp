// AMReX backend gate: kernel toy (toys.py)
#ifndef EE_KERNELTOY_HPP_
#define EE_KERNELTOY_HPP_

#include <AMReX_Array4.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <cstddef>
#include <type_traits>
#include "toy_enums.H"

namespace EE
{

// The recipe is written in three dimensions.
static_assert(AMREX_SPACEDIM == 3, "EE kernels are generated for AMREX_SPACEDIM == 3");

class KernelToy
{
  protected:
    amrex::Real kappa{};
    amrex::Real DX{}, DXI{}, DYI{}, DZI{};

  public:
    KernelToy(
        amrex::Real a_dx,
        amrex::Real a_kappa)
    {
        kappa = a_kappa;
        DX = a_dx;
        DXI = 1.0 / (a_dx);
        DYI = 1.0 / (a_dx);
        DZI = 1.0 / (a_dx);
    }
    
    // Launches the device functions, one ParallelFor each, in order, over the valid cells of a_rhs.
    // The caller synchronizes the stream.
    void compute(amrex::MultiFab &a_rhs, const amrex::MultiFab &a_state) const
    {
        const auto &rhs_arrays = a_rhs.arrays();
        const auto &state_arrays = a_state.const_arrays();
        const KernelToy self = *this;
        // Scratch for first's tile temporaries (t, u_rhs), on the valid cells; stream-ordered
        // allocation, so freeing it at the end of compute() needs no synchronization.
        amrex::MultiFab tile_first(a_rhs.boxArray(), a_rhs.DistributionMap(), 2, 0,
                                     amrex::MFInfo().SetArena(amrex::The_Async_Arena()));
        const auto &tile_first_arrays = tile_first.arrays();
        amrex::ParallelFor(a_rhs, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                           { self.first_0(i, j, k, rhs_arrays[box_no], state_arrays[box_no], tile_first_arrays[box_no]); });
        amrex::ParallelFor(a_rhs, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                           { self.first_1(i, j, k, rhs_arrays[box_no], state_arrays[box_no], tile_first_arrays[box_no]); });
        amrex::ParallelFor(a_rhs, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                           { self.second(i, j, k, rhs_arrays[box_no], state_arrays[box_no]); });
        amrex::ParallelFor(a_rhs, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                           { self.third(i, j, k, rhs_arrays[box_no], state_arrays[box_no]); });
    }
    
    // first loop 0
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void first_0(
        int i, int j, int k, amrex::Array4<amrex::Real> const &rhs,
        amrex::Array4<amrex::Real const> const &state,
        amrex::Array4<amrex::Real> const &tile) const
    {
        const amrex::Real *const state_ptr = state.ptr(i, j, k);
        const int sy = static_cast<int>(state.stride.a[0]);
        const int sz = static_cast<int>(state.stride.a[1]);
        const std::ptrdiff_t sn = state.stride.a[2];
        amrex::Real *const rhs_ptr = rhs.ptr(i, j, k);
        const std::ptrdiff_t rn = rhs.stride.a[2];
        (void)sy;
        (void)sz;
        (void)sn;
        (void)rn;
        amrex::Real *const tile_ptr = tile.ptr(i, j, k);
        const std::ptrdiff_t tn = tile.stride.a[2];
        (void)tile_ptr;
        (void)tn;
        const amrex::Real *const in_u = state_ptr + static_cast<std::ptrdiff_t>(c_u) * sn;
        const amrex::Real *const in_w = state_ptr + static_cast<std::ptrdiff_t>(c_w) * sn;
        const amrex::Real x = (i + 0.5) * DX;
        const amrex::Real y = (j + 0.5) * DX;
        const amrex::Real z = (k + 0.5) * DX;
        const amrex::Real x3 = (DYI * DZI);
        const amrex::Real x1 = in_u[0];
        amrex::Real x0 = (DXI * DXI);
        amrex::Real x2 = (x0 * ((-16 * ((in_u[1] + in_u[- 1]))) + (30 * x1) + ((in_u[- 2] + in_u[2]))));
        x0 = ((1.0 / 12.0) * kappa);
        const amrex::Real t = (x0 * ((-(x2)) + (DZI * ((8 * (((-(in_w[- sz])) + in_w[sz]))) + (((-(in_w[2 * sz])) + in_w[- 2 * sz])))) + (x3 * ((8 * (((-((((1.0 / 12.0) * ((8 * (((-(in_u[- sy - sz])) + in_u[- sy + sz]))) + (((-(in_u[- sy + 2 * sz])) + in_u[- sy - 2 * sz]))))))) + (((1.0 / 12.0) * ((8 * (((-(in_u[sy - sz])) + in_u[sy + sz]))) + (((-(in_u[sy + 2 * sz])) + in_u[sy - 2 * sz])))))))) + (((-((((1.0 / 12.0) * ((8 * (((-(in_u[2 * sy - sz])) + in_u[2 * sy + sz]))) + (((-(in_u[2 * sy + 2 * sz])) + in_u[2 * sy - 2 * sz]))))))) + (((1.0 / 12.0) * ((8 * (((-(in_u[- 2 * sy - sz])) + in_u[- 2 * sy + sz]))) + (((-(in_u[- 2 * sy + 2 * sz])) + in_u[- 2 * sy - 2 * sz])))))))))));
        tile_ptr[0 * tn] = t;
        x2 = in_w[0];
        const amrex::Real u_rhs = ((-(z)) + ((1.0 / 12.0) * amrex::Math::abs((DYI * ((8 * ((-(((-(in_u[sy])) + in_u[- sy]))))) + (((-(in_u[2 * sy])) + in_u[- 2 * sy])))))) + (t * x2) + (x * y));
        rhs_ptr[static_cast<std::ptrdiff_t>(c_u) * rn] = u_rhs;
        tile_ptr[1 * tn] = u_rhs;
    }
    
    // first loop 1
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void first_1(
        int i, int j, int k, amrex::Array4<amrex::Real> const &rhs,
        amrex::Array4<amrex::Real const> const &state,
        amrex::Array4<amrex::Real> const &tile) const
    {
        const amrex::Real *const state_ptr = state.ptr(i, j, k);
        const int sy = static_cast<int>(state.stride.a[0]);
        const int sz = static_cast<int>(state.stride.a[1]);
        const std::ptrdiff_t sn = state.stride.a[2];
        amrex::Real *const rhs_ptr = rhs.ptr(i, j, k);
        const std::ptrdiff_t rn = rhs.stride.a[2];
        (void)sy;
        (void)sz;
        (void)sn;
        (void)rn;
        amrex::Real *const tile_ptr = tile.ptr(i, j, k);
        const std::ptrdiff_t tn = tile.stride.a[2];
        (void)tile_ptr;
        (void)tn;
        const amrex::Real *const in_u = state_ptr + static_cast<std::ptrdiff_t>(c_u) * sn;
        const amrex::Real *const in_w = state_ptr + static_cast<std::ptrdiff_t>(c_w) * sn;
        const amrex::Real t = tile_ptr[0 * tn];
        const amrex::Real u_rhs = tile_ptr[1 * tn];
        const amrex::Real x5 = in_w[0];
        const amrex::Real x7 = u_rhs;
        const amrex::Real x1 = in_u[0];
        const amrex::Real x6 = (x1 * x1);
        rhs_ptr[static_cast<std::ptrdiff_t>(c_w) * rn] = ((t * t) + (1.0 / (x6)) + sqrt((1 + x6)) + ((-1.0 / 3.0) * x5) + ((1.0 / 4.0) * x7));
    }
    
    // second loop 0
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void second(
        int i, int j, int k, amrex::Array4<amrex::Real> const &rhs,
        amrex::Array4<amrex::Real const> const &state) const
    {
        const amrex::Real *const state_ptr = state.ptr(i, j, k);
        const int sy = static_cast<int>(state.stride.a[0]);
        const int sz = static_cast<int>(state.stride.a[1]);
        const std::ptrdiff_t sn = state.stride.a[2];
        amrex::Real *const rhs_ptr = rhs.ptr(i, j, k);
        const std::ptrdiff_t rn = rhs.stride.a[2];
        (void)sy;
        (void)sz;
        (void)sn;
        (void)rn;
        const amrex::Real *const in_u = state_ptr + static_cast<std::ptrdiff_t>(c_u) * sn;
        const amrex::Real *const in_w = state_ptr + static_cast<std::ptrdiff_t>(c_w) * sn;
        const amrex::Real u_rhs = rhs_ptr[static_cast<std::ptrdiff_t>(c_u) * rn];
        const amrex::Real x4 = ((1.0 / 12.0) * kappa);
        amrex::Real x0 = (DXI * DXI);
        const amrex::Real x1 = in_u[0];
        const amrex::Real x2 = (x0 * ((-16 * ((in_u[1] + in_u[- 1]))) + (30 * x1) + ((in_u[- 2] + in_u[2]))));
        x0 = u_rhs;
        rhs_ptr[static_cast<std::ptrdiff_t>(c_u) * rn] = (x0 + ((1.0 / 12.0) * DXI * ((8 * (((-(in_w[- 1])) + in_w[1]))) + (((-(in_w[2])) + in_w[- 2])))) + ((kappa > 1) ? amrex::Real((-1 * x2 * x4)) : amrex::Real(0)));
    }
    
    // third loop 0
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void third(
        int i, int j, int k, amrex::Array4<amrex::Real> const &rhs,
        amrex::Array4<amrex::Real const> const &state) const
    {
        const amrex::Real *const state_ptr = state.ptr(i, j, k);
        const int sy = static_cast<int>(state.stride.a[0]);
        const int sz = static_cast<int>(state.stride.a[1]);
        const std::ptrdiff_t sn = state.stride.a[2];
        amrex::Real *const rhs_ptr = rhs.ptr(i, j, k);
        const std::ptrdiff_t rn = rhs.stride.a[2];
        (void)sy;
        (void)sz;
        (void)sn;
        (void)rn;
        const amrex::Real *const in_p = state_ptr + static_cast<std::ptrdiff_t>(c_p) * sn;
        const amrex::Real *const in_q = state_ptr + static_cast<std::ptrdiff_t>(c_q) * sn;
        const amrex::Real x3 = (DYI * DZI);
        const amrex::Real x0 = (DXI * DXI);
        const amrex::Real q_rhs = ((kappa > 1) ? amrex::Real(((-1.0 / 12.0) * (-1 + kappa) * ((x0 * ((-16 * ((in_q[1] + in_q[- 1]))) + (30 * in_q[0]) + ((in_q[- 2] + in_q[2])))) + (-1 * x3 * ((8 * (((-((((1.0 / 12.0) * ((8 * (((-(in_q[- sy - sz])) + in_q[- sy + sz]))) + (((-(in_q[- sy + 2 * sz])) + in_q[- sy - 2 * sz]))))))) + (((1.0 / 12.0) * ((8 * (((-(in_q[sy - sz])) + in_q[sy + sz]))) + (((-(in_q[sy + 2 * sz])) + in_q[sy - 2 * sz])))))))) + (((-((((1.0 / 12.0) * ((8 * (((-(in_q[2 * sy - sz])) + in_q[2 * sy + sz]))) + (((-(in_q[2 * sy + 2 * sz])) + in_q[2 * sy - 2 * sz]))))))) + (((1.0 / 12.0) * ((8 * (((-(in_q[- 2 * sy - sz])) + in_q[- 2 * sy + sz]))) + (((-(in_q[- 2 * sy + 2 * sz])) + in_q[- 2 * sy - 2 * sz])))))))))))) : amrex::Real(0));
        rhs_ptr[static_cast<std::ptrdiff_t>(c_q) * rn] = q_rhs;
        rhs_ptr[static_cast<std::ptrdiff_t>(c_p) * rn] = ((2 * q_rhs) + ((1.0 / 12.0) * DYI * in_p[0] * ((8 * (((-(in_p[- sy])) + in_p[sy]))) + (((-(in_p[2 * sy])) + in_p[- 2 * sy])))));
    }
};

static_assert(std::is_trivially_copyable_v<KernelToy>, "KernelToy is captured by value in device lambdas");

} // namespace EE

#endif /* EE_KERNELTOY_HPP_ */
