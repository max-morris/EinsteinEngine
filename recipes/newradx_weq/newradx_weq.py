#!/usr/bin/env python3

#  Copyright (C) 2026 Lucas Timotheo Sanches, Max Morris, and other Einstein Engine contributors.
#
#  This file is part of the Einstein Engine (EinsteinEngine).
#
#  EinsteinEngine is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  EinsteinEngine is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from sympy import Expr

from EinsteinEngine import *

newradx_weq = ThornDef("TestEinsteinEngine", "NewRadXWeq")
newradx_weq.set_derivative_stencil(5)

###
# Parities and centering
###
parity_scalar = parities(+1, +1, +1)
vertex_centering = Centering.VVV

###
# State and RHS
###
u_rhs = newradx_weq.decl(
    "u_rhs",
    [],
    centering=vertex_centering,
    parity=parity_scalar
)

u = newradx_weq.decl(
    "u",
    [],
    centering=vertex_centering,
    parity=parity_scalar,
    rhs=u_rhs
)

rho_rhs = newradx_weq.decl(
    "rho_rhs",
    [],
    centering=vertex_centering,
    parity=parity_scalar
)

rho = newradx_weq.decl(
    "rho",
    [],
    centering=vertex_centering,
    parity=parity_scalar,
    rhs=rho_rhs
)

###
# Initialization
###
t, x, y, z = newradx_weq.mk_coords(with_time=True)

GAUSSIAN_AMPLITUDE = sympify(1)
GAUSSIAN_WIDTH = sympify(1)


def radius(X: Expr, Y: Expr, Z: Expr) -> Expr:
    return sqrt(X**2 + Y**2 + Z**2)


def gaussian(r: Expr) -> Expr:
    return GAUSSIAN_AMPLITUDE * exp(-r**2 / (2 * GAUSSIAN_WIDTH**2))  # type: ignore[no-any-return]


test_newradx_init = newradx_weq.create_function(
    "test_newradx_init",
    ScheduleBin.Init
)

test_newradx_init.add_eqn(
    u,
    (gaussian(t - radius(x, y, z)) - gaussian(t + radius(x, y, z))) / radius(x, y, z)
)

test_newradx_init.add_eqn(
    rho,
    -(gaussian(t - radius(x, y, z)) * (t - radius(x, y, z)) - gaussian(t +
      radius(x, y, z)) * (t + radius(x, y, z))) / ((GAUSSIAN_WIDTH**2) * radius(x, y, z))
)

###
# RHS Equations
###
test_newradx_rhs = newradx_weq.create_function(
    "test_newradx_rhs",
    ScheduleBin.Evolve
)

test_newradx_rhs.add_eqn(
    u_rhs,
    rho
)

test_newradx_rhs.add_eqn(
    rho_rhs,
    div(u, l0, l0) + div(u, l1, l1) + div(u, l2, l2)
)

newradx_u = NewRadXBoundaryBatch(
    u,
    sympify(0),
    sympify(1),
    sympify(1),
    ScheduleBin.Evolve,
    schedule_after=["test_newradx_rhs"]
)

newradx_rho = NewRadXBoundaryBatch(
    rho,
    sympify(0),
    sympify(1),
    sympify(1),
    ScheduleBin.Evolve,
    schedule_after=["test_newradx_rhs"]
)

newradx_weq.bake()

###
# Generate
###
CppCarpetXWizard(
    newradx_weq,
    CppCarpetXGenerator(
        newradx_weq,
        sync_mode=SyncMode.EmulatePresync,
        new_rad_x_boundary_fns=[newradx_u, newradx_rho]
    )
).generate_thorn()
