#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
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

"""
Toy modules for the AMReX backend's gates (docs/AMREX-BACKEND-DESIGN.md, section 9, items 6 and 7): one per emission
mode, each exercising stencils in three directions, coordinates, parameters, a guard, and (kernel mode) a hard split
with a tile temporary, an output read back by a later loop and by its own loop, and an overwrite. `values()` evaluates a baked module's equations at 50 digits on given data, the
reference the compiled code is compared with; on an unbaked module (`bake=False`) it evaluates the equations as the
recipe wrote them, before the guard fold and CSE, so that a bake that changes values shows too.
"""

from typing import Callable, Mapping

import mpmath
import sympy as sy
from sympy import Abs, Piecewise, Rational

from EinsteinEngine import *

STATE_COMPONENTS = {'u': 0, 'w': 1, 'q': 2, 'p': 3}  # kernel toy: enum values
POINT_COMPONENTS = {'u': 0, 'trK': 1}  # point toy: vars:: enum values; Df is pvars::Df = 0


def kernel_toy(bake: bool = True) -> AMReXModule:
    cc = AMReXModule("KernelToy",
                     arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                     binder=lambda base: None, rhs_binder=FieldBinder("state", lambda c: f"c_{c}"),
                     coordinates=CellCentered(), mode=KernelLaunchMode(),
                     params=MemberParams(ctor_args=[CtorArg("amrex::Real", "a_dx"), CtorArg("amrex::Real", "a_kappa")],
                                         dx="a_dx"),
                     includes=["toy_enums.H"])
    kappa = cc.add_param("kappa", init="a_kappa")
    x, y, z = cc.mk_coords()
    u_rhs = cc.output("u_rhs", [])
    u = cc.decl("u", [], rhs=u_rhs)
    w_rhs = cc.output("w_rhs", [])
    w = cc.decl("w", [], rhs=w_rhs)
    q_rhs = cc.output("q_rhs", [])
    q = cc.decl("q", [], rhs=q_rhs)
    p_rhs = cc.output("p_rhs", [])
    p = cc.decl("p", [], rhs=p_rhs)
    t = cc.decl("t", [])
    f1 = cc.create_function("first")
    f1.add_eqn(t, kappa * (D(u, l0, l0) + D(u, l1, l2) + D(w, l2)))
    f1.add_eqn(u_rhs, t * w + x * y - z + Abs(D(u, l1)))
    f1.split_loop()
    f1.add_eqn(w_rhs, t * t - Rational(1, 3) * w + sy.sqrt(1 + u * u) + u ** -2 + Rational(1, 4) * u_rhs)  # read back
    f2 = cc.create_function("second")
    f2.add_eqn(cc.overwrite(u_rhs), u_rhs + D(w, l0) + Piecewise((kappa, kappa > 1), (0, True)) * D(u, l0, l0))
    # A guard alone on q (the driver's false-guard run puts a NaN in q).
    f3 = cc.create_function("third")
    f3.add_eqn(q_rhs, Piecewise((kappa - 1, kappa > 1), (0, True)) * (D(q, l0, l0) + D(q, l1, l2)))
    f3.add_eqn(p_rhs, 2 * q_rhs + p * D(p, l1))  # reads q_rhs back in its own loop (the Gamma-driver shape)
    if bake:
        cc.bake()
    return cc


def point_toy(bake: bool = True) -> AMReXModule:
    cc = AMReXModule("PointToy",
                     arrays=[ArrayParam("un", "amrex::Array4<amrex::Real const> const &", Array4Call()),
                             ArrayParam("prim", "amrex::Array4<amrex::Real const> const &", Array4Call())],
                     binder=lambda base: (FieldBinder("un", lambda c: f"vars::{c}") if base in ("u", "trK")
                                          else FieldBinder("prim", lambda c: f"pvars::{c}") if base == "Df" else None),
                     coordinates=NodeCentered(), mode=PointFunctionMode(),
                     params=StructParams(arg="parameters", cpp_type="const ToyPar &"), includes=["toy_enums.H"])
    trK0 = cc.add_param("trK0")
    eps = cc.add_param("eps")
    x, y, z = cc.mk_coords()
    u = cc.decl("u", [])
    trK = cc.decl("trK", [])
    Df = cc.decl("Df", [])
    v_rhs = cc.output("v_rhs", [])
    ko_u = cc.output("ko_u", [])
    f = cc.create_function("rhs")
    f.add_eqn(v_rhs, D(u, l0) * trK0 + trK * Df + x * z ** 2 + D(u, l0, l1) + u ** Rational(-3, 2))
    k = cc.create_function("ko")
    k.add_eqn(ko_u, Piecewise((eps, eps > 0), (0, True)) * (D(u, l0, l0) + D(u, l1, l1) + D(u, l2, l2)))
    if bake:
        cc.bake()
    return cc


def values(cc: AMReXModule, field: Callable[[str, tuple[int, int, int]], mpmath.mpf],
           params: Mapping[str, mpmath.mpf], spacing: tuple[mpmath.mpf, mpmath.mpf, mpmath.mpf],
           coords: tuple[mpmath.mpf, mpmath.mpf, mpmath.mpf]) -> dict[str, mpmath.mpf]:
    """
    Evaluates every loop's baked equations in order at 50 digits. `field(var, offset)` gives a field component's value
    at an offset; outputs computed by earlier loops are kept; returns the final value of every output component.
    """
    mpmath.mp.dps = 50
    env: dict[str, mpmath.mpf] = {**params, 'DXI': 1 / spacing[0], 'DYI': 1 / spacing[1], 'DZI': 1 / spacing[2],
                                  'x': coords[0], 'y': coords[1], 'z': coords[2]}
    outputs: dict[str, mpmath.mpf] = dict()
    for name in cc.function_order:
        fn = cc.functions[name]
        for eqn_list in fn.eqn_complex.eqn_lists:
            # Baked: EE's order; unbaked: the recipe's (add_eqn) order, which the toys write in dependency order.
            eqns = eqn_list.sorted_eqns if fn.been_baked else list(eqn_list.eqns.items())
            for lhs, rhs in eqns:
                expr = rhs.replace(lambda e: isinstance(e, sy.Function) and getattr(e.func, 'name', '') == 'stencil',
                                   lambda e: sy.Float(field(str(e.args[0]), tuple(int(a) for a in e.args[1:])), 60))
                expr = expr.replace(lambda e: isinstance(e, sy.Function) and getattr(e.func, 'name', '') == 'noop',
                                    lambda e: e.args[0])
                subs = {}
                for s in expr.free_symbols:
                    n = str(s).replace("'", "")
                    if n in env:
                        subs[s] = sy.Float(env[n], 60)
                    elif n in outputs:
                        subs[s] = sy.Float(outputs[n], 60)
                    elif cc.field_binding(n) is not None:
                        subs[s] = sy.Float(field(n, (0, 0, 0)), 60)
                    else:
                        raise KeyError(f'{lhs}: no value for {n}')
                value = mpmath.mpf(str(sy.N(expr.xreplace(subs), 50)))
                key = str(lhs).replace("'", "")
                if cc.output_info(key) is not None:
                    outputs[key] = value
                else:
                    env[str(lhs)] = value
    return outputs
