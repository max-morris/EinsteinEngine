#define CARPETX_GF3D5
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>
#include <simd.hxx>
#include <defs.hxx>
#include <vect.hxx>
#include <cmath>
#include <tuple>
#define access(GF) (GF(p.mask, GF ## _layout, p.I))
#define store(GF, VAL) (GF.store(p.mask, GF ## _layout, p.I, VAL))
#define noop(OP) (OP)
// 1st derivatives
#define divx(GF) (GF(p.mask, GF ## _layout, p.I + p.DI[0]) - GF(p.mask, GF ## _layout, p.I - p.DI[0]))/(2*CCTK_DELTA_SPACE(0))
#define divy(GF) (GF(p.mask, GF ## _layout, p.I + p.DI[1]) - GF(p.mask, GF ## _layout, p.I - p.DI[1]))/(2*CCTK_DELTA_SPACE(1))
#define divz(GF) (GF(p.mask, GF ## _layout, p.I + p.DI[2]) - GF(p.mask, GF ## _layout, p.I - p.DI[2]))/(2*CCTK_DELTA_SPACE(2))
// 2nd derivatives
#define divxx(GF) (GF(p.mask, GF ## _layout, p.I + p.DI[0]) + GF(p.mask, GF ## _layout, p.I - p.DI[0]) - 2*GF(p.mask, GF ## _layout, p.I))/(CCTK_DELTA_SPACE(0)*CCTK_DELTA_SPACE(0))
#define divyy(GF) (GF(p.mask, GF ## _layout, p.I + p.DI[1]) + GF(p.mask, GF ## _layout, p.I - p.DI[1]) - 2*GF(p.mask, GF ## _layout, p.I))/(CCTK_DELTA_SPACE(1)*CCTK_DELTA_SPACE(1))
#define divzz(GF) (GF(p.mask, GF ## _layout, p.I + p.DI[2]) + GF(p.mask, GF ## _layout, p.I - p.DI[2]) - 2*GF(p.mask, GF ## _layout, p.I))/(CCTK_DELTA_SPACE(2)*CCTK_DELTA_SPACE(2))
#define stencil(GF, IX, IY, IZ) (GF(p.mask, GF ## _layout, p.I + IX*p.DI[0] + IY*p.DI[1] + IZ*p.DI[2]))
using namespace Arith;
using namespace Loop;
using std::cbrt,std::fmax,std::fmin,std::sqrt;
void bssn2adm(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_bssn2adm;
    DECLARE_CCTK_PARAMETERS;
    using vreal = Arith::simd<CCTK_REAL>;
    constexpr std::size_t vsize = std::tuple_size_v<vreal>;
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define AtDD00_layout VVV_layout
    #define AtDD01_layout VVV_layout
    #define AtDD02_layout VVV_layout
    #define AtDD11_layout VVV_layout
    #define AtDD12_layout VVV_layout
    #define AtDD22_layout VVV_layout
    #define alp_layout VVV_layout
    #define betax_layout VVV_layout
    #define betay_layout VVV_layout
    #define betaz_layout VVV_layout
    #define evo_lapse_layout VVV_layout
    #define evo_shiftU0_layout VVV_layout
    #define evo_shiftU1_layout VVV_layout
    #define evo_shiftU2_layout VVV_layout
    #define gtDD00_layout VVV_layout
    #define gtDD01_layout VVV_layout
    #define gtDD02_layout VVV_layout
    #define gtDD11_layout VVV_layout
    #define gtDD12_layout VVV_layout
    #define gtDD22_layout VVV_layout
    #define gxx_layout VVV_layout
    #define gxy_layout VVV_layout
    #define gxz_layout VVV_layout
    #define gyy_layout VVV_layout
    #define gyz_layout VVV_layout
    #define gzz_layout VVV_layout
    #define kxx_layout VVV_layout
    #define kxy_layout VVV_layout
    #define kxz_layout VVV_layout
    #define kyy_layout VVV_layout
    #define kyz_layout VVV_layout
    #define kzz_layout VVV_layout
    #define phi_layout VVV_layout
    #define trK_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    grid.loop_all_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        store(alp, access(evo_lapse));
        store(betax, access(evo_shiftU0));
        store(betay, access(evo_shiftU1));
        store(betaz, access(evo_shiftU2));
        vreal x0 = exp((4 * access(phi)));
        vreal x7 = (0.333333333333333 * access(trK));
        vreal x5 = (access(gtDD12) * x0);
        vreal x2 = (access(gtDD01) * x0);
        vreal x3 = (access(gtDD11) * x0);
        vreal x4 = (access(gtDD02) * x0);
        vreal x6 = (access(gtDD22) * x0);
        vreal x1 = (access(gtDD00) * x0);
        vreal kyz_1 = (access(AtDD12) * x0);
        vreal kyz_2 = (x5 * x7);
        store(kyz, (kyz_1 + kyz_2));
        store(gyz, x5);
        kyz_1 = (access(AtDD01) * x0);
        kyz_2 = (x2 * x7);
        store(kxy, (kyz_1 + kyz_2));
        store(gxy, x2);
        x5 = (access(AtDD11) * x0);
        kyz_1 = (x3 * x7);
        store(kyy, (kyz_1 + x5));
        store(gyy, x3);
        store(gxz, x4);
        kyz_2 = (access(AtDD02) * x0);
        x2 = (x4 * x7);
        store(kxz, (kyz_2 + x2));
        store(gzz, x6);
        kyz_1 = (access(AtDD22) * x0);
        x5 = (x6 * x7);
        store(kzz, (kyz_1 + x5));
        store(gxx, x1);
        x3 = (access(AtDD00) * x0);
        x4 = (x1 * x7);
        store(kxx, (x3 + x4));    
    });
}