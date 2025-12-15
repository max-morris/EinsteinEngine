#!/usr/bin/env python3
from EmitCactus import *
import nrpy.helpers.conditional_file_updater as cfu

# If we change our configuration, this will show us diffs of the
# new output and the old.
cfu.verbose = True

###
# Thorn creation
# We implement Eqs. (5)-(9) of https://arxiv.org/abs/gr-qc/0602104v2
###
nfweq = ThornDef("TestEmitCactus", "NonFlatWaveEqn")
nfweq.set_derivative_stencil(5)

###
# Initial data parameters
###
amplitude = nfweq.add_param(
    "amplitude",
    default=1.0,
    desc="The wave amplitude"
)

kx = nfweq.add_param(
    "kx",
    default=1.0,
    desc="The wave number in the x-direction"
)

ky = nfweq.add_param(
    "ky",
    default=1.0,
    desc="The wave number in the y-direction"
)

kz = nfweq.add_param(
    "kz",
    default=1.0,
    desc="The wave number in the z-direction"
)

###
# ADM variables
###
g = nfweq.decl(
    "g",
    [la, lb],
    symmetries=[(la, lb)],
    from_thorn="ADMBaseX"
)

alp = nfweq.decl("alp", [], from_thorn="ADMBaseX")

beta = nfweq.decl("beta", [ua], from_thorn="ADMBaseX")

###
# Test variables
###
ZeroVal = nfweq.decl("ZeroVal", [], from_thorn="ZeroTest")

###
# Parities and centering
###
parity_scalar = parities(+1, +1, +1)
parity_vector = parities(-1, +1, +1,  +1, -1, +1,  +1, +1, -1)
vertex_centering = Centering.VVV

###
# State and RHS
###
u_rhs = nfweq.decl(
    "u_rhs",
    [],
    centering=vertex_centering,
    parity=parity_scalar
)

u = nfweq.decl(
    "u",
    [],
    centering=vertex_centering,
    parity=parity_scalar,
    rhs=u_rhs
)

rho_rhs = nfweq.decl(
    "rho_rhs",
    [],
    centering=vertex_centering,
    parity=parity_scalar
)

rho = nfweq.decl(
    "rho",
    [],
    centering=vertex_centering,
    parity=parity_scalar,
    rhs=rho_rhs
)

v_rhs = nfweq.decl(
    "v_rhs",
    [la],
    centering=vertex_centering,
    parity=parity_vector
)

v = nfweq.decl(
    "v",
    [la],
    centering=vertex_centering,
    parity=parity_vector,
    rhs=v_rhs
)

flux = nfweq.decl(
    "flux",
    [ua],
    centering=vertex_centering,
    parity=parity_scalar
)

###
# Substitution rules
###
g_mat = nfweq.get_matrix(g[la, lb])
g_imat = inv(g_mat)
detg = det(g_mat)
nfweq.add_substitution_rule(g[ua, ub], g_imat)

###
# Initialization
###
t, x, y, z = nfweq.mk_coords(with_time=True)

# For the wave equation: c = w/k
omega = sqrt(kx**2 + ky**2 + kz**2)
id_func = amplitude * \
    cos(2 * omega * pi * t) * \
    cos(2 * kx * pi * x) * \
    cos(2 * ky * pi * y) * \
    cos(2 * kz * pi * z)

nfweq_init = nfweq.create_function(
    "nfweq_init",
    ScheduleBin.Init
)

nfweq_init.add_eqn(
    u,
    id_func
)

nfweq_init.add_eqn(
    rho,
    diff(id_func, t)
)

nfweq_init.add_eqn(
    v[l0],
    diff(id_func, x)
)

nfweq_init.add_eqn(
    v[l1],
    diff(id_func, y)
)

nfweq_init.add_eqn(
    v[l2],
    diff(id_func, z)
)

nfweq_init.bake()

###
# RHS Equations
###
nfweq_rhs = nfweq.create_function(
    "nfweq_rhs",
    ScheduleBin.Evolve
)

nfweq.add_substitution_rule(
    flux[ua],
    sqrt(detg) / alp * (
        beta[ua] * rho
        + alp**2 * (
            g[ua, ub] * v[lb]
            - beta[ua] * beta[ub] * v[lb] / alp**2
        )
    )
)

nfweq_rhs.add_eqn(
    u_rhs,
    rho
)

nfweq_rhs.add_eqn(
    rho_rhs,
    beta[ua] * D(rho, la)
    + alp / sqrt(detg) * D(flux[ua], la)
)

nfweq_rhs.add_eqn(
    v_rhs[la],
    D(rho, la)
)

nfweq_rhs.bake()

###
# Analysis
###
nfweq_zero = nfweq.create_function(
    "WaveZero",
    ScheduleBin.Analysis
)

nfweq_zero.add_eqn(ZeroVal, u - id_func)
nfweq_zero.bake(do_recycle_temporaries=False)

###
# Zero test group
###
zero_test_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CheckZeroGroup"),
    at_or_in=AtOrIn.At,
    schedule_bin=Identifier("analysis"),
    description=String("Do the check"),
    after=[Identifier("RicZero")]
)

###
# Generate
###
CppCarpetXWizard(
    nfweq,
    CppCarpetXGenerator(
        nfweq,
        interior_sync_mode=InteriorSyncMode.MixedRhs,
        extra_schedule_blocks=[zero_test_group]
    )
).generate_thorn()
