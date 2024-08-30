# see https://docs.einsteintoolkit.org/et-docs/images/0/05/PeterDiener15-MacLachlan.pdf
if __name__ == "__main__":

    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import do_inv, do_det
    from EmitCactus.generators.wizards import CppCarpetXWizard

    # Create a set of grid functions
    gf = ThornDef("PY_BSSN", "py_bssn")
    gf.set_div_stencil(5) # 4th order

    # From ADMBaseX
    g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
    gf.add_sym(g[li, lj], li, lj)

    k = gf.decl("k", [li, lj], from_thorn="ADMBaseX")
    gf.add_sym(k[li, lj], li, lj)

    alp = gf.decl("alp", [], from_thorn="ADMBaseX")
    beta = gf.decl("beta", [ua], from_thorn="ADMBaseX")
    ###

    x,y,z = gf.mk_coords()

    # BSSN Vars
    gt = gf.decl("gt", [li,lj]) # \tilde{g}
    gf.add_sym(gt[li, lj], li, lj)
    gt_dt = gf.decl("gt_dt", [li,lj]) # \tilde{g}
    gf.add_sym(gt_dt[li, lj], li, lj)

    phi = gf.decl("phi", [])
    phi_dt = gf.decl("phi_dt", [])

    At = gf.decl("At", [li,lj]) # \tilde{A}
    gf.add_sym(At[li, lj], li, lj)

    trK = gf.decl("trK", []) # trace of Extrinsic Curvature 

    Gt = gf.decl("Gt", [ui]) # \tilde{\Gamma}^i
    ###

    gf.mk_subst(gt_dt[li,lj])
    gf.mk_subst(gt[li,lj])
    gf.mk_subst(At[li,lj])
    gf.mk_subst(beta[ui])
    gf.mk_subst(Gt[ui])
    gmat = gt.get_matrix(gt[li,lj])
    imat = do_inv(gmat)*do_det(gmat) # Use the fact that det(gmat) = 1

    fun = gf.create_function("evo", ScheduleBin.Evolve)
    fun.add_eqn(gt_dt[li,lj], -2*alp*At[li,lj] + beta[uk]*div(gt[li,lj],lk) + gt[li,lk]*div(beta[uk],lj) - (2/3)*gt[li,lj]*div(beta[uk],lk))
    fun.add_eqn(phi_dt, -(1/6)*alp*trK + div(phi,lk)*beta[uk] + (1/6)*div(beta[uk],lk))
    fun.add_eqn(Gt_dt[ui], gt[uj,uk] div(beta[ui],lj,lk))

    fun.bake()

    CppCarpetXWizard(gf).generate_thorn()

