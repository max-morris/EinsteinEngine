from EmitCactus import *

gen_opts = {
    "do_cse": False,
    "do_madd": False,
    "do_recycle_temporaries": False,
    "do_split_output_eqns": True
}

# Create a set of grid functions
gf = ThornDef("TestEmitCactus", "Ricci")
gf.set_derivative_stencil(5)

a = gf.add_param("a", default=10.0, desc="Just a constant")
b = gf.add_param("b", default=0.2, desc="Just a constant")
c = gf.add_param("c", default=0.1, desc="Just a constant")

# Declare gfs
g = gf.decl("g", [li, lj], symmetries=[(li, lj)], from_thorn="ADMBaseX")
x, y, z = gf.mk_coords()

Ric = gf.decl("Ric", [la, lb], symmetries=[(la, lb)])
ZeroVal = gf.decl("ZeroVal", [], from_thorn="ZeroTest")
G = gf.decl("Affine", [ua, lb, lc], symmetries=[(lb, lc)], substitution_rule=None)

gmat = gf.get_matrix(g[la,lb])
print(gmat)
imat = simplify(inv(gmat)*det(gmat)) 
gf.add_substitution_rule(g[ua, ub], imat)

# Metric
grr = sqrt(1+c**2)*(a+b*x**2)
gqq = sqrt(1+c**2)/(a+b*x**2)
gpp = sympify(1)
Z = sympify(0)
gmat = mkMatrix([
[grr,   c,   Z],
[  c, gqq,   Z],
[  Z,   Z, gpp]])
assert det(gmat) == 1

# Define the affine connections
gf.add_substitution_rule(G[la, lb, lc], (D(g[la, lb], lc) + D(g[la, lc], lb) - D(g[lb, lc], la)) / 2)
gf.add_substitution_rule(G[ud, lb, lc], g[ud, ua] * G[la, lb, lc])

fun = gf.create_function("setGL", ScheduleBin.Analysis)

fun.add_eqn(Ric[li, lj],
             D(G[ua, li, lj], la) - D(G[ua, la, li], lj) +
             G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])

fun.bake(**gen_opts)

fun = gf.create_function("MetricSet", ScheduleBin.Analysis, schedule_before=["setGL"])
fun.add_eqn(g[li, lj], gmat)
fun.bake(**gen_opts)

fun = gf.create_function("RicZero", ScheduleBin.Analysis, schedule_after=["setGL"])
fun.add_eqn(ZeroVal, Ric[l0,l0]-b*(a*c**2 + a - 3*b*c**2*x**2 - 3*b*x**2)/(a**2 + 2*a*b*x**2 + b**2*x**4))
fun.bake(**gen_opts)

gf.do_global_cse()

check_zero = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier('CheckZeroGroup'),
    at_or_in=AtOrIn.At,
    schedule_bin=Identifier('analysis'),
    description=String('Do the check'),
    after=[Identifier('RicZero')]
)

CppCarpetXWizard(
    gf,
    CppCarpetXGenerator(
        gf,
        interior_sync_mode=InteriorSyncMode.MixedRhs,
        extra_schedule_blocks=[check_zero]
    )
).generate_thorn()
