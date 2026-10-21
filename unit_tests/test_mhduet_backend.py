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
Tests of MHDuet's subclass of the shared AMReX backend (MHDuetPointFunction, docs/AMREX-BACKEND-DESIGN.md section 10,
step 4) on a toy module: Simflowny's component names, the two Array4Call arrays and their binding by declaration,
rhs= fields in `state`, node coordinates, `parameters.<name>`, `amrex::Real &` outputs in the fixed signature order, and
the refusals. The MHDuet header itself is checked against MHDuet's code by mhduet/validation/exact/check_emitted.py.
Run as a plain script.
"""

from sympy import Piecewise

from EinsteinEngine import *
from EinsteinEngine.frontend.dsl.dsl_exception import DslException


def text(cc: AMReXModule) -> str:
    wizard = AMReXWizard(cc)
    return str(wizard.code_visitor.visit(wizard.generator.generate("// toy")))


def toy() -> MHDuetPointFunction:
    return MHDuetPointFunction("Toy", fields={"Df": "state", "vfd": "aux", "pf": "aux"}, namespace="ee_toy",
                               includes=["AmrCoreProblem.H"])


def expect(code: str, *needles: str) -> None:
    for needle in needles:
        assert needle in code, f'missing: {needle}\n{code}'


def test_toy() -> None:
    cc = toy()
    eps = cc.add_param("dissipation_factor_gtd_xy")
    trK0 = cc.add_param("trK0")
    x, y, z = cc.mk_coords()
    rhs_g = cc.output("RHS_gtd", [li, lj], symmetries=[(li, lj)])
    gtd = cc.decl("gtd", [li, lj], symmetries=[(li, lj)], rhs=rhs_g)
    rhs_b = cc.output("RHS_Betau", [ui])
    Betau = cc.decl("Betau", [ui], rhs=rhs_b)
    rhs_t = cc.output("RHS_trK", [])
    trK = cc.decl("trK", [], rhs=rhs_t)
    Df = cc.decl("Df", [])
    vfd = cc.decl("vfd", [li])
    pf = cc.decl("pf", [])
    tmp = cc.decl("Ct", [li, lj], symmetries=[(li, lj)])  # a temporary: not in `fields`, no rhs=
    ko = cc.output("ko_gtd", [li, lj], symmetries=[(li, lj)])
    f = cc.create_function("ccz4_rhs")
    f.add_eqn(tmp[li, lj], Betau[u0] * gtd[li, lj])
    f.add_eqn(rhs_g[li, lj], tmp[li, lj] * vfd[l2] + D(gtd[li, lj], l1))
    f.add_eqn(rhs_b[ui], Betau[ui] * (pf + x))
    f.add_eqn(rhs_t, trK * Df + trK0 * z)
    k = cc.create_function("ccz4_ko")
    k.add_eqn(ko[li, lj], Piecewise((eps, eps > 0), (0, True)) * D(gtd[li, lj], l0, l0))
    cc.bake()
    code = text(cc)
    expect(code, 'namespace ee_toy', '#include "AmrCoreProblem.H"',
           'state(i, j + 1, k, simflowny_vars::gtd_xy)', 'state(i, j, k, simflowny_vars::Betau_z)',
           'state(i, j, k, simflowny_vars::Df)', 'aux(i, j, k, simflowny_vars::vfd_z)', 'aux(i, j, k, simflowny_vars::pf)',
           'const amrex::Real ee_x = prob_lo[0] + i * dx[0];', 'const amrex::Real ee_z = prob_lo[2] + k * dx[2];',
           'parameters.trK0', 'Simflowny_par const & parameters', 'amrex::Real & RHS_gtd_xz', 'amrex::Real & ko_gtd_yz',
           '((parameters.dissipation_factor_gtd_xy > 0) ? amrex::Real(')
    assert 'simflowny_vars::Ct' not in code, code
    assert 'gtd_yx' not in code and 'vfd_y)' not in code.split('void ccz4_ko(')[1], code
    # The fixed signature order (design 5.4): i, j, k; state, aux; prob_lo, dx; parameters; the outputs in
    # declaration order (a symmetric tensor's components in EE's order 00, 01, 11, 02, 12, 22).
    sig = code[code.index('void ccz4_rhs(') + len('void ccz4_rhs('):code.index(')\n{')]
    names = [p.split()[-1] for p in sig.split(',\n')]
    assert names == ['i', 'j', 'k', 'state', 'aux', 'prob_lo', 'dx', 'parameters',
                     'RHS_gtd_xx', 'RHS_gtd_xy', 'RHS_gtd_yy', 'RHS_gtd_xz', 'RHS_gtd_yz', 'RHS_gtd_zz',
                     'RHS_Betau_x', 'RHS_Betau_y', 'RHS_Betau_z', 'RHS_trK'], names
    ko_fn = code[code.index('void ccz4_ko('):]
    expect(ko_fn, '[[maybe_unused]] amrex::Array4<amrex::Real const> const & aux',
           '[[maybe_unused]] amrex::GpuArray<amrex::Real, 3> const & prob_lo')
    assert cc.field_binding('gtd_xy') == FieldBinding('state', 'simflowny_vars::gtd_xy')
    assert cc.field_binding('vfd_x') == FieldBinding('aux', 'simflowny_vars::vfd_x')
    assert cc.field_binding('Ct_xy') is None and cc.field_binding('RHS_trK') is None


def test_naming() -> None:
    cc = toy()
    a = cc.decl("A", [ui, lj])
    s = cc.decl("S", [li, lj], symmetries=[(li, lj)])
    names = set(cc.indices_of)
    assert {"A_xy", "A_yx", "A_zz", "S_xy", "S_zz"} <= names and "S_yx" not in names, sorted(names)


def test_refusals() -> None:
    def raises(fn) -> None:
        try:
            fn()
        except DslException:
            return
        raise AssertionError("no DslException")
    raises(lambda: MHDuetPointFunction("Bad", fields={"Df": "prim"}, namespace="ee_bad"))
    cc = toy()
    raises(lambda: cc.output("vfd", [li]))  # bound as a field: cannot be an output


def main() -> None:
    test_toy()
    test_naming()
    test_refusals()
    print('test_mhduet_backend: OK')


if __name__ == '__main__':
    main()
