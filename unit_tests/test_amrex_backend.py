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
Tests of the AMReX kernel-class backend (frontend/dsl/amrex, generators/amrex_generator.py, emit/code/amrex) on a toy
class: state reads at stencil offsets, cell coordinates, a hard split whose second loop reads a temporary of the first
(a tile temporary, kept in a scratch MultiFab), a later function updating an rhs component through an overwrite, and
the generator's refusals. Checks the printed C++ text; compiling it needs AMReX. Run as a plain script.
"""

from sympy import Rational

from EinsteinEngine import *
from EinsteinEngine.generators.generator_exception import GeneratorException


def toy(read_unwritten: bool = False) -> str:
    cc = AMReXKernelClass("Toy", ctor_args=[AMReXCtorArg("amrex::Real", "a_dx")], dx="a_dx")
    kappa = cc.add_param("kappa", init="1.5")
    x, y, z = cc.mk_coords()
    u_rhs = cc.decl("u_rhs", [])
    u = cc.decl("u", [], rhs=u_rhs)
    w_rhs = cc.decl("w_rhs", [])
    w = cc.decl("w", [], rhs=w_rhs)
    t = cc.decl("t", [])
    s = cc.decl("s", [])

    f1 = cc.create_function("first")
    f1.add_eqn(t, kappa * (D(u, l0, l0) + D(u, l1, l1) + D(u, l2, l2)))
    f1.add_eqn(u_rhs, t * w + x)
    f1.add_eqn(s, kappa * w * w)  # read only by loop 1: an output of loop 0 that goes to the scratch, not the rhs
    f1.split_loop()
    f1.add_eqn(w_rhs, t * t - Rational(1, 3) * w + s)

    f2 = cc.create_function("second")
    u_rhs2 = cc.overwrite(u_rhs)
    f2.add_eqn(u_rhs2, u_rhs + D(w, l0))
    if read_unwritten:
        f3 = cc.create_function("third")
        v_rhs = cc.decl("v_rhs", [])
        cc.decl("v", [], rhs=v_rhs)
        q_rhs = cc.decl("q_rhs", [])
        cc.decl("q", [], rhs=q_rhs)
        f3.add_eqn(q_rhs, v_rhs)
    cc.bake()
    return AMReXVisitor_text(cc)


def AMReXVisitor_text(cc: AMReXKernelClass) -> str:
    from EinsteinEngine.emit.code.amrex.amrex_visitor import AMReXVisitor
    return str(AMReXVisitor().visit(AMReXGenerator(cc).generate_kernel_class("// toy")))


def main() -> None:
    code = toy()
    for needle in (
        'void first_0(', 'void first_1(', 'void second(',  # one device function per loop
        'amrex::MultiFab tile_first(', 'The_Async_Arena()',  # the scratch for t
        'tile_first_arrays[box_no]',
        'tile_ptr[1 * tn] = t;', 'const amrex::Real t = tile_ptr[1 * tn];',  # stored by loop 0, loaded by loop 1
        'tile_ptr[0 * tn] = s;', 'const amrex::Real s = tile_ptr[0 * tn];',
        'const amrex::Real u_rhs = rhs_ptr[static_cast<std::ptrdiff_t>(c_u) * rn];',  # read before the update
        'rhs_ptr[static_cast<std::ptrdiff_t>(c_u) * rn] =',
        'const amrex::Real x = (i + 0.5) * DX;',
        'in_u[- 1]', 'in_u[sy]', 'in_u[2 * sz]',  # stencil offsets in x, y, z
    ):
        assert needle in code, f'missing: {needle}\n{code}'
    # Launch order: first_0, first_1, second.
    order = [code.index(f'self.{f}(') for f in ('first_0', 'first_1', 'second')]
    assert order == sorted(order), order
    # Loop 1 does not recompute t, and loop 0 does not read w_rhs.
    first_1 = code[code.index('void first_1('):code.index('void second(')]
    assert 'in_u[' not in first_1, first_1
    try:
        toy(read_unwritten=True)
    except GeneratorException as e:
        assert 'no earlier loop writes' in str(e), e
    else:
        raise AssertionError('reading an rhs component no earlier loop writes must fail')
    print('test_amrex_backend: OK')


if __name__ == '__main__':
    main()
