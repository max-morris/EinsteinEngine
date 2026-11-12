// Enums and the parameter struct of the AMReX backend's gate toys (unit_tests/amrex/toys.py).
#ifndef TOY_ENUMS_H_
#define TOY_ENUMS_H_
#include <AMReX_REAL.H>
enum { c_u = 0, c_w = 1, c_q = 2, c_p = 3 };
namespace vars { enum { u = 0, trK = 1 }; }
namespace pvars { enum { Df = 0 }; }
struct ToyPar
{
    amrex::Real trK0;
    amrex::Real eps;
};
#endif
