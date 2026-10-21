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
Tests of the GRChombo compute-class backend (frontend/dsl/grchombo, generators/grchombo_generator.py,
emit/code/grchombo) on a toy class:
- the class shape: namespace, constructor arguments with a default, constructor check, includes, parameters as
  const members, and `template <class data_t> void compute(Cell<data_t>) const`;
- naming and access: GRChombo's c_ enum names, SIMDIFY loads of the state, stencil offsets, a symmetric tensor stored
  once per independent component, and store_vars of each rhs into its state variable's slot;
- |x| as ee_abs, which is simd_max(x, -x) (GRChombo's simd abs truncates to int depending on include order);
- the GR_SPACEDIM == 3 assert;
- the one-function rule.

A golden md5 of the CCZ4 header (recipes/foreign/grchombo/ccz4_grchombo.py --fd-order 4) pins the generated kernel
validated against GRChombo. The golden check takes about two minutes; set EE_SKIP_GOLDEN=1 to skip it. Checks the
printed C++ text; compiling it needs GRChombo. Run as a plain script.
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from sympy import Abs, Rational

from EinsteinEngine import *
from EinsteinEngine.emit.code.grchombo.grchombo_visitor import GRChomboVisitor
from EinsteinEngine.frontend.dsl.dsl_exception import DslException

# The CCZ4 header validated against GRChombo (foreign-kernel-speedups/grchombo, levels 1-3 and the stencil-bias proof).
CCZ4_HEADER_MD5 = 'c5e059d51dfcc8ebbc3852c9682d6471'  # with the nested mixed derivatives (PR #114); cb9770a3 before


def toy() -> tuple[GRChomboComputeClass, str]:
    cc = GRChomboComputeClass(
        "Toy",
        ctor_args=[GRChomboCtorArg("double", "a_dx"), GRChomboCtorArg("double", "a_kappa", "1.5")],
        dx="a_dx",
        ctor_checks=[GRChomboCtorCheck("a_kappa < 0", "kappa must not be negative")],
        includes=["ToyIncludes.hpp"],
    )
    kappa = cc.add_param("kappa", init="a_kappa")
    u_rhs = cc.decl("u_rhs", [])
    u = cc.decl("u", [], rhs=u_rhs)
    g_rhs = cc.decl("g_rhs", [li, lj], symmetries=[(li, lj)])
    g = cc.decl("g", [li, lj], rhs=g_rhs, symmetries=[(li, lj)])
    fn = cc.create_function("compute")
    fn.add_eqn(u_rhs, kappa * (D(u, l0, l0) + D(u, l1, l1) + D(u, l2, l2)) + Abs(g[l0, l1]))
    fn.add_eqn(g_rhs[li, lj], Rational(1, 2) * D(u, li, lj) - kappa * g[li, lj])
    cc.bake()
    return cc, str(GRChomboVisitor().visit(GRChomboGenerator(cc).generate_compute_class("// toy")))


def check_shape(code: str) -> None:
    for needle in (
        '#ifndef EE_TOY_HPP_', '#include "Cell.hpp"', '#include "simd.hpp"', '#include "UserVariables.hpp"',
        '#include "ToyIncludes.hpp"', 'namespace EE\n{', 'class Toy\n{', '} // namespace EE',
        'const double kappa;', 'const double DXI, DYI, DZI;',  # parameters and inverse spacings as const members
        'double a_kappa = 1.5)', ': kappa(a_kappa),', 'DXI(1.0 / (a_dx))',  # constructor
        'if (a_kappa < 0)\n            MayDay::Error("EE::Toy: kappa must not be negative");',
        'template <class data_t> void compute(Cell<data_t> current_cell) const',
    ):
        assert needle in code, f'missing: {needle}\n{code}'


def check_access(code: str) -> None:
    components = ['g11', 'g12', 'g13', 'g22', 'g23', 'g33', 'u']
    for c in components:
        load = f'const auto in_{c} = SIMDIFY<data_t>(box_pointers.m_in_ptr[c_{c}]);'
        assert code.count(load) == 1, f'load of {c}: {code.count(load)}\n{code}'
        # Each rhs is stored once, into its state variable's slot.
        assert code.count(f', c_{c});') == 1, f'store of {c}: {code.count(f", c_{c});")}\n{code}'
    for absent in ('c_g21', 'c_g31', 'c_g32', 'in_g21', 'c_u_rhs', 'c_g_rhs'):
        assert absent not in code, f'unexpected: {absent}\n{code}'
    for needle in (
        'const int sy = box_pointers.m_in_stride[1];', 'const int sz = box_pointers.m_in_stride[2];',
        'data_t(in_u[idx])', 'in_u[idx + 2]', 'in_u[idx - 2 * sy]', 'in_u[idx + 2 * sz]',  # x, y, z offsets
        'in_u[idx + 1 - sz]',  # a mixed second derivative
    ):
        assert needle in code, f'missing: {needle}\n{code}'


def check_abs_and_dims(code: str) -> None:
    helper = 'template <class data_t> ALWAYS_INLINE data_t ee_abs(const data_t &x) { return simd_max(x, -x); }'
    assert code.count(helper) == 1, code
    assert 'ee_abs(x0)' in code, code
    body = code[code.index('class Toy'):]  # the header comment explains why abs is avoided, naming abs(int)
    assert re.search(r'(?<![_a-z])abs\(', body) is None, f'a bare abs call\n{code}'
    assert 'static_assert(GR_SPACEDIM == 3, ' in code, code


def check_one_function() -> None:
    cc = GRChomboComputeClass("One", ctor_args=[GRChomboCtorArg("double", "a_dx")], dx="a_dx")
    cc.create_function("compute")
    try:
        cc.create_function("other")
    except DslException as e:
        assert 'exactly one function' in str(e), e
    else:
        raise AssertionError('a second function must be refused')


def check_ccz4_golden() -> None:
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as out:
        subprocess.run([sys.executable, str(root / 'recipes/foreign/grchombo/ccz4_grchombo.py'), '--fd-order', '4',
                        '--out-dir', out], check=True, capture_output=True, env={**os.environ, 'PYTHONPATH': str(root)})
        md5 = hashlib.md5((Path(out) / 'CCZ4RHS4.hpp').read_bytes()).hexdigest()
    assert md5 == CCZ4_HEADER_MD5, f'CCZ4RHS4.hpp md5 {md5}, expected {CCZ4_HEADER_MD5}'


def main() -> None:
    _, code = toy()
    check_shape(code)
    check_access(code)
    check_abs_and_dims(code)
    check_one_function()
    print('toy class: OK')
    if os.environ.get('EE_SKIP_GOLDEN') == '1':
        print('golden CCZ4 header: skipped (EE_SKIP_GOLDEN=1)')
    else:
        check_ccz4_golden()
        print(f'golden CCZ4 header: OK ({CCZ4_HEADER_MD5})')
    print('test_grchombo_backend: OK')


if __name__ == '__main__':
    main()
