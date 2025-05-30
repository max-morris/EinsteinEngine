if __name__ == '__main__':
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import mkMatrix, do_inv, do_sympify
    from EmitCactus.emit.ccl.interface.interface_visitor import InterfaceVisitor
    from EmitCactus.emit.ccl.param.param_visitor import ParamVisitor
    from EmitCactus.emit.ccl.schedule.schedule_visitor import ScheduleVisitor
    from EmitCactus.emit.code.cpp.cpp_visitor import CppVisitor
    from typing import cast, Any
    from sympy import Expr, Idx, cos, sin
    from EmitCactus.emit.tree import Centering
    from EmitCactus.generators.wizards import CppCarpetXWizard

    # Create a set of grid functions
    gf = ThornDef("TestAffine", "Affine")
    gf.set_derivative_stencil(3)

    # Declare gfs
    g = gf.decl("g", [li, lj], sym=[(li,lj,1)], from_thorn="ADMBaseX")
    x, y, z = gf.mk_coords()
    G = gf.decl("Affine", [ua, lb, lc], sym=[(lb,lc,1)])
    Ric = gf.decl("Ric", [la, lb], sym=[(la,lb,1)])

    gf.mk_subst(g[la, lb], mksymbol_for_tensor_xyz)

    gmat = gf.get_matrix(g[la, lb])
    imat = do_inv(gmat)
    gf.mk_subst(g[ua, ub], imat)
    gf.mk_subst(D(g[la, lb], lc))  # D(g[l0,l1],l2) -> gDD01_dD2
    gf.mk_subst(D(g[ua, ub], lc))
    gf.mk_subst(G[la, lb, lc], (D(g[la, lb], lc) + D(g[la, lc], lb) - D(g[lb, lc], la)) / 2)
    gf.mk_subst(G[ud, lb, lc])  # , g[ud,ua]*G[la, lb, lc])

    fun = gf.create_function("setAff", ScheduleBin.Analysis)

    fun.add_eqn(G[ud, lb, lc], g[ud, ua] * G[la, lb, lc])
    # Optimizations:
    # Does it pull the 2nd G out of the sum of: G[ua, la, lb] * G[ub, li, lj]? Check
    # -1*Foo should be -Foo
    # Comment:
    # Check non-diagonal metric

    # Ensure the equations make sense
    fun.bake(do_recycle_temporaries=True)

    CppCarpetXWizard(gf).generate_thorn()
