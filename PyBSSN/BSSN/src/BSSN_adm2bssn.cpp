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
void adm2bssn(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_adm2bssn;
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
    #define ConfConnectU0_layout VVV_layout
    #define ConfConnectU1_layout VVV_layout
    #define ConfConnectU2_layout VVV_layout
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
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        store(evo_lapse, access(alp));
        store(evo_shiftU0, access(betax));
        store(evo_shiftU1, access(betay));
        store(evo_shiftU2, access(betaz));
        vreal x12 = (0.333333333333333 * access(gxx));
        vreal x73 = (((1.0 / 12.0) * noop(((-(stencil(gxx,0,2,0))) + stencil(gxx,0,-2,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gxx,0,-1,0))) + stencil(gxx,0,1,0)))));
        vreal x58 = (((1.0 / 12.0) * noop(((-(stencil(gxx,0,0,2))) + stencil(gxx,0,0,-2)))) + ((2.0 / 3.0) * noop(((-(stencil(gxx,0,0,-1))) + stencil(gxx,0,0,1)))));
        vreal x83 = (((1.0 / 12.0) * noop(((-(stencil(gxx,2,0,0))) + stencil(gxx,-2,0,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gxx,-1,0,0))) + stencil(gxx,1,0,0)))));
        vreal x87 = (((1.0 / 12.0) * noop(((-(stencil(gxy,2,0,0))) + stencil(gxy,-2,0,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gxy,-1,0,0))) + stencil(gxy,1,0,0)))));
        vreal x78 = (((1.0 / 12.0) * noop(((-(stencil(gxy,0,2,0))) + stencil(gxy,0,-2,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gxy,0,-1,0))) + stencil(gxy,0,1,0)))));
        vreal x64 = (((1.0 / 12.0) * noop(((-(stencil(gxy,0,0,2))) + stencil(gxy,0,0,-2)))) + ((2.0 / 3.0) * noop(((-(stencil(gxy,0,0,-1))) + stencil(gxy,0,0,1)))));
        vreal x33 = (0.333333333333333 * access(gxy));
        vreal x1 = pow2(access(gxy));
        vreal x29 = (access(gxy) * access(gxz));
        vreal x2 = pow2(access(gxz));
        vreal x39 = (0.333333333333333 * access(gxz));
        vreal x76 = (((1.0 / 12.0) * noop(((-(stencil(gxz,0,2,0))) + stencil(gxz,0,-2,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gxz,0,-1,0))) + stencil(gxz,0,1,0)))));
        vreal x85 = (((1.0 / 12.0) * noop(((-(stencil(gxz,2,0,0))) + stencil(gxz,-2,0,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gxz,-1,0,0))) + stencil(gxz,1,0,0)))));
        vreal x62 = (((1.0 / 12.0) * noop(((-(stencil(gxz,0,0,2))) + stencil(gxz,0,0,-2)))) + ((2.0 / 3.0) * noop(((-(stencil(gxz,0,0,-1))) + stencil(gxz,0,0,1)))));
        vreal x50 = (((1.0 / 12.0) * noop(((-(stencil(gyy,2,0,0))) + stencil(gyy,-2,0,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gyy,-1,0,0))) + stencil(gyy,1,0,0)))));
        vreal x36 = (0.333333333333333 * access(gyy));
        vreal x56 = (((1.0 / 12.0) * noop(((-(stencil(gyy,0,0,2))) + stencil(gyy,0,0,-2)))) + ((2.0 / 3.0) * noop(((-(stencil(gyy,0,0,-1))) + stencil(gyy,0,0,1)))));
        vreal x72 = (((1.0 / 12.0) * noop(((-(stencil(gyy,0,2,0))) + stencil(gyy,0,-2,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gyy,0,-1,0))) + stencil(gyy,0,1,0)))));
        vreal x17 = (access(gxx) * access(gyy));
        vreal x26 = (access(gxy) * access(gyz));
        vreal x74 = (((1.0 / 12.0) * noop(((-(stencil(gyz,0,2,0))) + stencil(gyz,0,-2,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gyz,0,-1,0))) + stencil(gyz,0,1,0)))));
        vreal x21 = (access(gxz) * access(gyz));
        vreal x52 = (((1.0 / 12.0) * noop(((-(stencil(gyz,2,0,0))) + stencil(gyz,-2,0,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gyz,-1,0,0))) + stencil(gyz,1,0,0)))));
        vreal x0 = pow2(access(gyz));
        vreal x60 = (((1.0 / 12.0) * noop(((-(stencil(gyz,0,0,2))) + stencil(gyz,0,0,-2)))) + ((2.0 / 3.0) * noop(((-(stencil(gyz,0,0,-1))) + stencil(gyz,0,0,1)))));
        vreal x42 = (0.333333333333333 * access(gyz));
        vreal x8 = (access(gyy) * access(gzz));
        vreal x48 = (((1.0 / 12.0) * noop(((-(stencil(gzz,2,0,0))) + stencil(gzz,-2,0,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gzz,-1,0,0))) + stencil(gzz,1,0,0)))));
        vreal x70 = (((1.0 / 12.0) * noop(((-(stencil(gzz,0,2,0))) + stencil(gzz,0,-2,0)))) + ((2.0 / 3.0) * noop(((-(stencil(gzz,0,-1,0))) + stencil(gzz,0,1,0)))));
        vreal x13 = (access(gxx) * access(gzz));
        vreal x55 = (((1.0 / 12.0) * noop(((-(stencil(gzz,0,0,2))) + stencil(gzz,0,0,-2)))) + ((2.0 / 3.0) * noop(((-(stencil(gzz,0,0,-1))) + stencil(gzz,0,0,1)))));
        vreal x45 = (0.333333333333333 * access(gzz));
        vreal x6 = (-(access(kxx)));
        vreal x32 = (-(access(kxy)));
        vreal x38 = (-(access(kxz)));
        vreal x35 = (-(access(kyy)));
        vreal x41 = (-(access(kyz)));
        vreal x44 = (-(access(kzz)));
        vreal x96 = (DYI * access(gyz));
        vreal x95 = (DYI * access(gzz));
        vreal x92 = (DZI * access(gyz));
        vreal x59 = (DZI * x58);
        vreal x88 = (DXI * x87);
        vreal x79 = (DYI * x78);
        vreal x65 = (DZI * x64);
        vreal x30 = ((-(x29)) + (access(gxx) * access(gyz)));
        vreal x77 = (DYI * x76);
        vreal x86 = (DXI * x85);
        vreal x63 = (DZI * x62);
        vreal x51 = (DXI * x50);
        vreal x57 = (DZI * x56);
        vreal x18 = (x1 + (-(x17)));
        vreal x27 = ((-(x26)) + (access(gxz) * access(gyy)));
        vreal x75 = (DYI * x74);
        vreal x22 = ((-(x21)) + (access(gxy) * access(gzz)));
        vreal x84 = (DXI * x52);
        vreal x3 = ((access(gxx) * x0) + (access(gyy) * x2) + (access(gzz) * x1) + (-1 * access(gxx) * access(gyy) * access(gzz)) + (-2 * access(gxy) * access(gxz) * access(gyz)));
        vreal x61 = (DZI * x60);
        vreal x9 = (x0 + (-(x8)));
        vreal x49 = (DXI * x48);
        vreal x71 = (DYI * x70);
        vreal x14 = (x2 + (-(x13)));
        vreal x31 = (access(kyz) * x30);
        vreal x28 = (access(kxz) * x27);
        vreal x23 = (access(kxy) * x22);
        vreal x7 = pown(x3, -1);
        vreal x82 = pown(x3, -2);
        vreal x4 = (-(x3));
        x3 = ((x13 * x57) + (x59 * x8) + (2 * x21 * x65) + (2 * x26 * x63) + (2 * x29 * x61) + (DZI * x17 * x55) + (-1 * DZI * x0 * x58) + (-1 * DZI * x1 * x55) + (-1 * DZI * x2 * x56) + (-2 * DZI * access(gxx) * access(gyz) * x60) + (-2 * DZI * access(gxy) * access(gzz) * x64) + (-2 * DZI * access(gxz) * access(gyy) * x62));
        x55 = ((x13 * x51) + (x17 * x49) + (2 * x21 * x88) + (2 * x26 * x86) + (2 * x29 * x84) + (DXI * x8 * x83) + (-1 * DXI * x0 * x83) + (-1 * DXI * x1 * x48) + (-1 * DXI * x2 * x50) + (-2 * DXI * access(gxx) * access(gyz) * x52) + (-2 * DXI * access(gxy) * access(gzz) * x87) + (-2 * DXI * access(gxz) * access(gyy) * x85));
        x60 = ((x17 * x71) + (2 * x21 * x79) + (2 * x26 * x77) + (2 * x29 * x75) + (DYI * x13 * x72) + (DYI * x73 * x8) + (-1 * DYI * x0 * x73) + (-1 * DYI * x1 * x70) + (-1 * DYI * x2 * x72) + (-2 * DYI * access(gxx) * access(gyz) * x74) + (-2 * DYI * access(gxy) * access(gzz) * x78) + (-2 * DYI * access(gxz) * access(gyy) * x76));
        x48 = (x36 * x7);
        x83 = (x7 * x9);
        x87 = (x45 * x7);
        x0 = (x33 * x7);
        x1 = (x42 * x7);
        x13 = (x12 * x7);
        x17 = (x14 * x7);
        x2 = (2 * x7);
        x21 = (x39 * x7);
        x26 = (x18 * x7);
        x29 = cbrt(pown(x4, -1));
        x70 = ((1.0 / 3.0) * pow(x4, (-2.0 / 3.0)));
        x72 = 0.0833333333333333;
        x74 = log(x4);
        store(phi, (x72 * x74));
        x8 = cbrt(x4);
        x72 = (-(x3));
        x74 = (-(x55));
        x4 = (-(x60));
        vreal x11 = (access(kxx) * x83);
        vreal x16 = (access(kyy) * x17);
        vreal x20 = (access(kzz) * x26);
        vreal gtDD02_1 = access(gxz);
        vreal gtDD02_2 = x29;
        store(gtDD02, (gtDD02_1 * gtDD02_2));
        gtDD02_1 = access(gxx);
        gtDD02_2 = x29;
        store(gtDD00, (gtDD02_1 * gtDD02_2));
        gtDD02_1 = access(gyy);
        gtDD02_2 = x29;
        store(gtDD11, (gtDD02_1 * gtDD02_2));
        gtDD02_1 = (2 * x29);
        gtDD02_2 = access(gxy);
        vreal gtDD01_2 = x29;
        store(gtDD01, (gtDD01_2 * gtDD02_2));
        gtDD01_2 = access(gzz);
        gtDD02_2 = x29;
        store(gtDD22, (gtDD01_2 * gtDD02_2));
        gtDD01_2 = access(gyz);
        gtDD02_2 = x29;
        store(gtDD12, (gtDD01_2 * gtDD02_2));
        gtDD01_2 = (x7 * x70);
        gtDD02_2 = (x55 * x70);
        x55 = (x7 * x8);
        vreal x94 = (x72 * x82);
        vreal trK_1 = x11;
        vreal trK_2 = x16;
        vreal trK_3 = x20;
        vreal trK_4 = (x2 * x23);
        vreal trK_5 = (x2 * x28);
        vreal trK_6 = (x2 * x31);
        store(trK, (trK_1 + trK_2 + trK_3 + trK_4 + trK_5 + trK_6));
        x2 = (gtDD02_1 * ((-(x32)) + (-1 * x0 * x23)));
        trK_1 = (gtDD02_1 * ((-(x32)) + (-1 * x0 * x28)));
        trK_2 = (gtDD02_1 * ((-(x32)) + (-1 * x0 * x31)));
        trK_3 = (x29 * ((-(x32)) + (-1 * x11 * x33)));
        trK_4 = (x29 * ((-(x32)) + (-1 * x16 * x33)));
        trK_5 = (x29 * ((-(x32)) + (-1 * x20 * x33)));
        store(AtDD01, (trK_1 + trK_2 + trK_3 + trK_4 + trK_5 + x2));
        trK_6 = (gtDD02_1 * ((-(x35)) + (-1 * x23 * x48)));
        x0 = (gtDD02_1 * ((-(x35)) + (-1 * x28 * x48)));
        x32 = (gtDD02_1 * ((-(x35)) + (-1 * x31 * x48)));
        x33 = (x29 * ((-(x35)) + (-1 * x11 * x36)));
        trK_1 = (x29 * ((-(x35)) + (-1 * x16 * x36)));
        trK_2 = (x29 * ((-(x35)) + (-1 * x20 * x36)));
        store(AtDD11, (trK_1 + trK_2 + trK_6 + x0 + x32 + x33));
        trK_3 = (gtDD02_1 * ((-(x38)) + (-1 * x21 * x23)));
        trK_4 = (gtDD02_1 * ((-(x38)) + (-1 * x21 * x28)));
        trK_5 = (gtDD02_1 * ((-(x38)) + (-1 * x21 * x31)));
        x2 = (x29 * ((-(x38)) + (-1 * x11 * x39)));
        x48 = (x29 * ((-(x38)) + (-1 * x16 * x39)));
        x35 = (x29 * ((-(x38)) + (-1 * x20 * x39)));
        store(AtDD02, (trK_3 + trK_4 + trK_5 + x2 + x35 + x48));
        x36 = (gtDD02_1 * ((-(x6)) + (-1 * x13 * x23)));
        trK_1 = (gtDD02_1 * ((-(x6)) + (-1 * x13 * x28)));
        trK_2 = (gtDD02_1 * ((-(x6)) + (-1 * x13 * x31)));
        trK_6 = (x29 * ((-(x6)) + (-1 * x11 * x12)));
        x0 = (x29 * ((-(x6)) + (-1 * x12 * x16)));
        x32 = (x29 * ((-(x6)) + (-1 * x12 * x20)));
        store(AtDD00, (trK_1 + trK_2 + trK_6 + x0 + x32 + x36));
        x33 = (gtDD02_1 * ((-(x41)) + (-1 * x1 * x23)));
        x21 = (gtDD02_1 * ((-(x41)) + (-1 * x1 * x28)));
        x38 = (gtDD02_1 * ((-(x41)) + (-1 * x1 * x31)));
        x39 = (x29 * ((-(x41)) + (-1 * x11 * x42)));
        trK_3 = (x29 * ((-(x41)) + (-1 * x16 * x42)));
        trK_4 = (x29 * ((-(x41)) + (-1 * x20 * x42)));
        store(AtDD12, (trK_3 + trK_4 + x21 + x33 + x38 + x39));
        trK_5 = (gtDD02_1 * ((-(x44)) + (-1 * x23 * x87)));
        x2 = (gtDD02_1 * ((-(x44)) + (-1 * x28 * x87)));
        x35 = (gtDD02_1 * ((-(x44)) + (-1 * x31 * x87)));
        x48 = (x29 * ((-(x44)) + (-1 * x11 * x45)));
        x13 = (x29 * ((-(x44)) + (-1 * x16 * x45)));
        x12 = (x29 * ((-(x44)) + (-1 * x20 * x45)));
        store(AtDD22, (trK_5 + x12 + x13 + x2 + x35 + x48));
        x6 = (gtDD01_2 * x60);
        trK_1 = (gtDD01_2 * x3);
        trK_2 = (gtDD02_2 * x7);
        trK_6 = (-1 * gtDD02_2 * x83);
        x0 = (-1 * x22 * x6);
        x32 = (-1 * trK_1 * x27);
        x36 = (-1 * x8 * ((x7 * ((access(gxy) * x71) + (x78 * x95) + (-1 * access(gxz) * x75) + (-1 * x76 * x96))) + (-1 * x22 * x4 * x82)));
        x1 = (-1 * x8 * ((x7 * ((-1 * access(gxy) * x61) + (-1 * x64 * x92) + (DZI * access(gxz) * x56) + (DZI * access(gyy) * x62))) + (-1 * x27 * x94)));
        x41 = (-1 * x55 * ((-1 * access(gyy) * x49) + (-1 * access(gzz) * x51) + (2 * DXI * access(gyz) * x52)));
        x42 = (x74 * x8 * x82 * x9);
        store(ConfConnectU0, (trK_6 + x0 + x1 + x32 + x36 + x41 + x42));
        trK_3 = (-1 * trK_2 * x27);
        trK_4 = (-1 * x30 * x6);
        x21 = (-1 * x55 * ((-1 * access(gxx) * x57) + (-1 * access(gyy) * x59) + (2 * DZI * access(gxy) * x64)));
        x33 = (-1 * x55 * ((access(gxx) * x75) + (x73 * x96) + (-1 * access(gxy) * x77) + (-1 * access(gxz) * x79)));
        x38 = (-1 * x55 * ((-1 * access(gxy) * x84) + (-1 * access(gyz) * x88) + (DXI * access(gxz) * x50) + (DXI * access(gyy) * x85)));
        x39 = (-1 * x26 * x3 * x70);
        x23 = (x18 * x72 * x8 * x82);
        x28 = (x27 * x74 * x8 * x82);
        gtDD02_1 = (x30 * x4 * x8 * x82);
        store(ConfConnectU2, (gtDD02_1 + trK_3 + trK_4 + x21 + x23 + x28 + x33 + x38 + x39));
        x31 = (-1 * trK_2 * x22);
        x87 = (-1 * trK_1 * x30);
        x11 = (-1 * x8 * ((x7 * ((access(gxx) * x61) + (x58 * x92) + (-1 * access(gxy) * x63) + (-1 * access(gxz) * x65))) + (-1 * x30 * x94)));
        x16 = (-1 * x55 * ((-1 * access(gxx) * x71) + (-1 * x73 * x95) + (2 * DYI * access(gxz) * x76)));
        x20 = (-1 * x55 * ((access(gxy) * x49) + (access(gzz) * x88) + (-1 * access(gxz) * x84) + (-1 * access(gyz) * x86)));
        x29 = (-1 * x17 * x60 * x70);
        x44 = (x14 * x4 * x8 * x82);
        x45 = (x22 * x74 * x8 * x82);
        store(ConfConnectU1, (x11 + x16 + x20 + x29 + x31 + x44 + x45 + x87));    
    });
}