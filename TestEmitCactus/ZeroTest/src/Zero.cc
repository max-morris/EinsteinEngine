#define CARPETX_GF3D5
#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"
#include <loop_device.hxx>
#include <simd.hxx>
#include <defs.hxx>
#include <vect.hxx>
#define access(GF) (GF(GF ## _layout, p.I))

using namespace Arith;
using namespace Loop;

void CheckZero(CCTK_ARGUMENTS)
{
    DECLARE_CCTK_ARGUMENTSX_CheckZero;
    DECLARE_CCTK_PARAMETERS;
    //using vreal = Arith::simd<CCTK_REAL>;
    //constexpr std::size_t vsize = std::tuple_size_v<vreal>;
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define ZeroVal_layout VVV_layout
    double rms = 0;
    double maxv = 0;
    int n = 0;
    int nancount = 0;
    grid.loop_int<VVV_centered[0], VVV_centered[1], VVV_centered[2]>(grid.nghostzones, [&] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        CCTK_REAL zero = fabs(access(ZeroVal));
        if(std::isnan(zero)) {
            nancount ++;
            return;
        }
        if(verbose && zero > max_tol) {
            printf("Zero[%d,%d,%d] = %g\n", p.I[0], p.I[1], p.I[2], zero);
        }
        rms += zero*zero;
        if(zero > maxv) {
            maxv = zero;
        }
        n ++;
    });
    rms = sqrt(rms/n);
    printf("---> nancount=%d, rms=%g, max=%g\n", nancount, rms, maxv);
    if(nancount > 0) printf("::ERROR:: nancount is not zero\n");
    if(rms > rms_tol) printf("::ERROR:: rms value=%g exceeds tolerance: %g\n", rms, rms_tol);
    if(maxv > max_tol) printf("::ERROR:: max value=%g exceeds tolerance: %g\n", maxv, max_tol);
}
