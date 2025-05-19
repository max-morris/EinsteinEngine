#!/usr/bin/env python3
from EmitCactus.emit.ccl.schedule.schedule_tree import GroupOrFunction, ScheduleBlock, AtOrIn
from EmitCactus.emit.tree import Identifier, String, Centering
from EmitCactus.generators.cpp_carpetx_generator import CppCarpetXGenerator
from EmitCactus.dsl.sympywrap import do_sqrt, mkMatrix
from sympy import Indexed, Symbol

if __name__ == "__main__":

    """
    The waveequation! It can't be solved too many times.
    """

    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import *
    from sympy import Expr, Idx, sin, cos, Matrix
    import nrpy.helpers.conditional_file_updater as cfu
    from math import pi

    from EmitCactus.generators.wizards import CppCarpetXWizard

    class Message:
        def __init__(self)->None:
            print("BEGIN MESSAGE")
        def __del__(self)->None:
            print("END MESSAGE")

    msg = Message()

    # If we change our configuration, this will show us diffs of the
    # new output and the old.
    cfu.verbose = True


    flat_metric = mkMatrix([
        [1,0,0],
        [0,1,0],
        [0,0,1]])

    # Create a set of grid functions
    gf = ThornDef("TestEmitCactus", "WaveEqn")

    # Use a NRPy calculated stencil instead
    # of simply calling functions such as divx()
    gf.set_div_stencil(5)

    # Declare gfs
    v_t = gf.decl("v_t", [], centering=Centering.VVV)
    v = gf.decl("v", [], centering=Centering.VVV, rhs=v_t)
    u_t = gf.decl("u_t", [], centering=Centering.VVV)
    u = gf.decl("u", [], centering=Centering.VVV, rhs=u_t)
    ZeroVal = gf.decl("ZeroVal", [], from_thorn="ZeroTest")

    # Declare the metric
    g = gf.decl("g", [li, lj])
    gf.add_sym(g[li, lj], li, lj)

    # Declare params
    spd = gf.add_param("spd", default=1.0, desc="The wave speed")
    kx = gf.add_param("kx", default=pi / 20, desc="The wave number in the x-direction")
    ky = gf.add_param("ky", default=pi / 20, desc="The wave number in the y-direction")
    kz = gf.add_param("kz", default=pi / 20, desc="The wave number in the z-direction")
    amp = gf.add_param("amp", default=10, desc="The amplitude")
    # c = w/k
    w = spd*do_sqrt(kx**2 + ky**2 + kz**2)

    # Fill in values
    gf.mk_subst(g[li, lj], flat_metric)
    gf.mk_subst(g[ui, uj], flat_metric)

    # stencil(la) -> [stencil(f,1,0,0), stencil(f,0,1,0), stencil(f,0,0,1)]

    mydiv = gf.mk_stencil("mydiv",la,(-2*stencil(0)+stencil(la)+stencil(-la))/(DD(la)**2))

    ## gf.mk_stencil(mydiv,la,la,(stencil(la)-2*stencil(0)+stencil(-la))/(DD[la]**2))
    ## gf.mk_stencil(mydiv,la,lb,(stencil(la+lb)-stencil(la-lb)+stencil(-la-lb)-stencil(-la+lb))/(2*DD[la]*DD[lb]))
    ## # la + lb, if la==l0 and lb==l1, [1,0,0] + [0,1,0] => [1,1,0]
    ## # la + l0, if la==l0, [1,0,0] + [1,0,0] => [2,0,0]
    ## # la + l0, if la==l1, [0,1,0] + [1,0,0] => [1,1,0]
    # gf.mk_stencil(mydiv,la,c*stencil(2*la)+b*stencil(la)+a*stencil(0))/(2*DD[la]))
    ## # if la==l0, (c*stencil(f,2,0,0) + b*stencil(f,1,0,0) + a*stencil(f,0,0,0))/(2*DX)

    t, x, y, z = gf.mk_coords(with_time=True)

    # Add the equations we want to evolve.
    fun = gf.create_function("newwave_evo", ScheduleBin.Evolve)
    fun.add_eqn(v_t, u)
    stencil = mkFunction("stencil")
    def mydiv(arg, ind1, ind2):
        return stencil(arg, ind1, ind2, 0)
    fun.add_eqn(u_t, spd ** 2 * g[ui, uj] * mydiv(v, li, lj))
    print('*** ThornFunction wave_evo:')
    fun.bake()

    # Dump
    fun.dump()

    # Show tensortypes
    fun.show_tensortypes()

    # Again for wave_init
    # du/dt = spd**2 * ((d/dx)**2 u + (d/dy)**2 u)
    # dv/dt = u
    vfun = amp*sin(kx * x) * sin(ky * y) * sin(kz * z) * sin(w * t)
    ufun = vfun.diff(t)
    fun = gf.create_function("newwave_init", ScheduleBin.Init)
    fun.add_eqn(u,  ufun)
    fun.add_eqn(v,  vfun)
    fun.bake()
    fun.dump()
    fun.show_tensortypes()

    fun = gf.create_function("refine", ScheduleBin.EstimateError)
    regrid_error = gf.decl("regrid_error", [], centering=Centering.CCC, from_thorn='CarpetXRegrid')
    #fun.add_eqn(regrid_error, 2*v*v)
    fun.add_eqn(regrid_error, do_sympify(0)) #9/((x-20)**2 + (y-20)**2))
    fun.bake()

    fun = gf.create_function("WaveZero", ScheduleBin.Analysis)
    fun.add_eqn(ZeroVal, u - ufun)
    fun.bake()

    fun.dump()

    check_zero = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier('CheckZeroGroup'),
        at_or_in=AtOrIn.At,
        schedule_bin=Identifier('analysis'),
        description=String('Do the check'),
        after=[Identifier('RicZero')]
    )

    CppCarpetXWizard(gf, CppCarpetXGenerator(gf, extra_schedule_blocks=[check_zero])).generate_thorn()

