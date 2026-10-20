// Driver of the AMReX backend's differential gate (docs/AMREX-BACKEND-DESIGN.md, section 9, item 7): evaluates the
// kernel toy through compute() on a multi-box MultiFab with a nonzero box origin, and the point toy at every cell of a
// box with anisotropic dx and prob_lo, and writes every input and output value (%.17g) for gate.py to compare with its
// 50-digit evaluation of the baked equations. Usage: AMReXGate OUT_DIR
// With -DGATE_POINT_ONLY (make POINT_ONLY=TRUE) only the point toy is built: the kernel toy's StridedPointer access
// reads Array4::stride, which AMReX releases before 26.x (e.g. 25.11, MHDuet's) do not have.
#include "toy_enums.H"
#ifndef GATE_POINT_ONLY
#include "KernelToy.hpp"
#endif
#include "PointToy.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <cmath>
#include <cstdio>
#include <string>

namespace
{
double input(int i, int j, int k, int c) { return 1.0 + 0.25 * std::sin(0.37 * i + 0.61 * j + 0.29 * k + 1.1 * c); }

void dump(const amrex::MultiFab &mf, int ngrow, const std::string &path)
{
    std::FILE *f = std::fopen(path.c_str(), "w");
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const auto a = mf.const_array(mfi);
        amrex::LoopOnCpu(amrex::grow(mfi.validbox(), ngrow), mf.nComp(), [&](int i, int j, int k, int c)
                         { std::fprintf(f, "%d %d %d %d %.17g\n", i, j, k, c, a(i, j, k, c)); });
    }
    std::fclose(f);
}
} // namespace

int main(int argc, char *argv[])
{
    const std::string out = argc > 1 ? argv[1] : ".";
    int amrex_argc = 1;
    amrex::Initialize(amrex_argc, argv, false);
    {
        const amrex::MFInfo info = amrex::MFInfo().SetArena(amrex::The_Managed_Arena());
#ifndef GATE_POINT_ONLY
        // Kernel toy: 2 components (c_u, c_w), 8 boxes from (5, -3, 7) to (12, 4, 14), 3 ghost cells.
        const amrex::Box domain(amrex::IntVect(5, -3, 7), amrex::IntVect(12, 4, 14));
        amrex::BoxArray ba(domain);
        ba.maxSize(4);
        const amrex::DistributionMapping dm(ba);
        amrex::MultiFab state(ba, dm, 2, 3, info), rhs(ba, dm, 2, 0, info);
        for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi)
        {
            const auto a = state.array(mfi);
            amrex::LoopOnCpu(mfi.fabbox(), 2, [&](int i, int j, int k, int c) { a(i, j, k, c) = input(i, j, k, c); });
        }
        rhs.setVal(std::nan(""));
        EE::KernelToy(0.1, 1.3).compute(rhs, state);
        amrex::Gpu::streamSynchronize();
        dump(state, 3, out + "/kernel_inputs.txt");
        dump(rhs, 0, out + "/kernel_outputs.txt");
#endif

        // Point toy: un (u, trK) and prim (Df) on one box from (-2, 3, 1), anisotropic dx and prob_lo.
        const amrex::Box pbox(amrex::IntVect(-2, 3, 1), amrex::IntVect(3, 8, 6));
        const amrex::BoxArray pba(pbox);
        const amrex::DistributionMapping pdm(pba);
        amrex::MultiFab un(pba, pdm, 2, 3, info), prim(pba, pdm, 1, 3, info), pout(pba, pdm, 2, 0, info);
        for (amrex::MFIter mfi(un); mfi.isValid(); ++mfi)
        {
            const auto a = un.array(mfi);
            const auto p = prim.array(mfi);
            amrex::LoopOnCpu(mfi.fabbox(), 2, [&](int i, int j, int k, int c) { a(i, j, k, c) = input(i, j, k, c); });
            amrex::LoopOnCpu(mfi.fabbox(), [&](int i, int j, int k) { p(i, j, k, 0) = input(i, j, k, 7); });
        }
        const amrex::GpuArray<amrex::Real, 3> dx{0.1, 0.2, 0.3}, prob_lo{-1.5, 0.25, 2.0};
        const ToyPar par{0.7, 0.2};
        for (amrex::MFIter mfi(pout); mfi.isValid(); ++mfi)
        {
            const auto u = un.const_array(mfi);
            const auto p = prim.const_array(mfi);
            const auto o = pout.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   amrex::Real v_rhs, ko_u;
                                   EE::rhs(i, j, k, u, p, prob_lo, dx, par, v_rhs);
                                   EE::ko(i, j, k, u, p, prob_lo, dx, par, ko_u);
                                   o(i, j, k, 0) = v_rhs;
                                   o(i, j, k, 1) = ko_u;
                               });
        }
        amrex::Gpu::streamSynchronize();
        dump(un, 3, out + "/point_un.txt");
        dump(prim, 3, out + "/point_prim.txt");
        dump(pout, 0, out + "/point_outputs.txt");
    }
    amrex::Finalize();
    return 0;
}
