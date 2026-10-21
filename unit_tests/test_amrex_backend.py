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
Tests of EE's shared AMReX backend (docs/AMREX-BACKEND-DESIGN.md, section 9, tests 1-4) on toy modules: classification
by declaration, both emission modes, hard splits with tile temporaries, rhs reloads, the refusals, the printing
rules, and guards (folding, the post-CSE error, the hoisting warning). Checks the printed C++ text; compiling and
running it are the gates of section 9 (6, 7). Run as a plain script.
"""

from typing import Callable

import sympy as sy
from sympy import Piecewise, Rational

from EinsteinEngine import *
from EinsteinEngine.emit.code.amrex.amrex_visitor import AMReXVisitor
from EinsteinEngine.generators.amrex.guards import fold_guards
from EinsteinEngine.generators.generator_exception import GeneratorException
from EinsteinEngine.frontend.dsl.dsl_exception import DslException


def text(cc: AMReXModule) -> str:
    wizard = AMReXWizard(cc)
    return str(wizard.code_visitor.visit(wizard.generator.generate("// toy")))


def kernel_module() -> AMReXModule:
    return AMReXModule("Toy", arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                       binder=lambda base: None, rhs_binder=FieldBinder("state", lambda c: f"c_{c}"),
                       coordinates=CellCentered(), mode=KernelLaunchMode(),
                       params=MemberParams(ctor_args=[CtorArg("amrex::Real", "a_dx")], dx="a_dx"))


def point_module(**kw: object) -> AMReXModule:
    return AMReXModule("Point", arrays=[ArrayParam("un", "amrex::Array4<amrex::Real const> const &", Array4Call()),
                                        ArrayParam("prim", "amrex::Array4<amrex::Real const> const &", Array4Call())],
                       binder=lambda base: (FieldBinder("un", lambda c: f"vars::{c}") if base in ("u", "trK")
                                            else FieldBinder("prim", lambda c: f"pvars::{c}") if base == "Df" else None),
                       coordinates=NodeCentered(), mode=PointFunctionMode(**kw),  # type: ignore[arg-type]
                       params=StructParams(arg="parameters", cpp_type="const Par &"))


def expect(code: str, *needles: str) -> None:
    for needle in needles:
        assert needle in code, f'missing: {needle}\n{code}'


def refuses(build: Callable[[], object], fragment: str, kinds: tuple[type[BaseException], ...] = (GeneratorException,
                                                                                                    DslException)) -> None:
    try:
        build()
    except kinds as e:
        assert fragment in str(e), e
    else:
        raise AssertionError(f'expected a refusal mentioning {fragment!r}')


def test_kernel_mode() -> None:
    cc = kernel_module()
    kappa = cc.add_param("kappa", init="1.5")
    x, y, z = cc.mk_coords()
    u_rhs = cc.output("u_rhs", [])
    u = cc.decl("u", [], rhs=u_rhs)
    w_rhs = cc.output("w_rhs", [])
    w = cc.decl("w", [], rhs=w_rhs)
    t = cc.decl("t", [])
    s = cc.decl("s", [])
    f1 = cc.create_function("first")
    f1.add_eqn(t, kappa * (D(u, l0, l0) + D(u, l1, l1) + D(u, l2, l2)))
    f1.add_eqn(u_rhs, t * w + x)
    f1.add_eqn(s, kappa * w * w)  # read only by loop 1: goes to the scratch
    f1.split_loop()
    f1.add_eqn(w_rhs, t * t - Rational(1, 3) * w + s)
    f2 = cc.create_function("second")
    f2.add_eqn(cc.overwrite(u_rhs), u_rhs + D(w, l0))
    cc.bake()
    code = text(cc)
    expect(code, 'void first_0(', 'void first_1(', 'void second(', 'amrex::MultiFab tile_first(',
           'The_Async_Arena()', 'tile_first_arrays[box_no]', 'tile_ptr[1 * tn] = t;',
           'const amrex::Real t = tile_ptr[1 * tn];', 'tile_ptr[0 * tn] = s;', 'const amrex::Real s = tile_ptr[0 * tn];',
           'const amrex::Real u_rhs = rhs_ptr[static_cast<std::ptrdiff_t>(c_u) * rn];',
           'rhs_ptr[static_cast<std::ptrdiff_t>(c_u) * rn] =', 'const amrex::Real x = (i + 0.5) * DX;',
           'in_u[- 1]', 'in_u[sy]', 'in_u[2 * sz]', 'kappa = 1.5;')
    order = [code.index(f'self.{f}(') for f in ('first_0', 'first_1', 'second')]
    assert order == sorted(order), order
    first_1 = code[code.index('void first_1('):code.index('void second(')]
    assert 'in_u[' not in first_1, first_1


def test_point_mode(outputs: str) -> None:
    cc = point_module(outputs=outputs)
    trK0 = cc.add_param("trK0")
    eps = cc.add_param("eps")
    x, y, z = cc.mk_coords()
    u = cc.decl("u", [])
    trK = cc.decl("trK", [])
    Df = cc.decl("Df", [])  # bound by the binder to the second array
    v_rhs = cc.output("v_rhs", [])
    ko = cc.output("ko_u", [])
    t = cc.decl("t", [])
    f = cc.create_function("rhs")
    f.add_eqn(t, D(u, l0) * trK0)
    f.add_eqn(v_rhs, t + trK * Df + z)
    k = cc.create_function("ko")
    k.add_eqn(ko, Piecewise((eps, eps > 0), (0, True)) * (D(u, l0, l0) + D(u, l1, l1)))
    cc.bake()
    code = text(cc)
    expect(code, 'AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE', 'un(i + 1, j, k, vars::u)', 'prim(i, j, k, pvars::Df)',
           'parameters.trK0', 'const amrex::Real ee_z = prob_lo[2] + k * dx[2];', 'ee_DXI = 1.0 / dx[0];',
           '#include <AMReX_Math.H>')
    assert 'trK0[' not in code and 'vars::trK0' not in code, code  # a parameter named like a field stays a parameter
    rhs = code[code.index(' rhs('):code.index(' ko(')]
    assert '[[maybe_unused]] amrex::Array4<amrex::Real const> const & prim' not in rhs, rhs  # rhs reads prim
    ko_fn = code[code.index(' ko('):]
    expect(ko_fn, '[[maybe_unused]] amrex::Array4<amrex::Real const> const & prim', '? ')
    if outputs == 'refs':
        expect(code, 'amrex::Real & v_rhs', 'v_rhs = ', 'amrex::Real & ko_u')
    else:
        expect(code, 'struct rhs_outputs', 'amrex::Real v_rhs;', 'out.v_rhs = ', 'return out;')
    # Folded: the guard selects the whole product, so a false guard gives exactly 0 without 0 * S.
    assert ko_fn.count('?') == 1 and 'ko_u = ((parameters.eps > 0) ? amrex::Real(' in ko_fn, ko_fn


def test_refusals() -> None:
    def unbound() -> None:
        cc = point_module()
        u = cc.decl("u", [])
        o = cc.output("o", [])
        f = cc.create_function("f")
        f.add_eqn(o, u * sy.Symbol("DT"))
        cc.bake()
        text(cc)
    refuses(unbound, "unbound symbols are refused")

    def stencil_of_temp() -> None:
        cc = kernel_module()
        r = cc.output("u_rhs", [])
        cc.decl("u", [], rhs=r)
        t = cc.decl("t", [])
        f = cc.create_function("f")
        f.add_eqn(t, cc.decl("q", []) * 2)
        f.add_eqn(r, D(t, l0))
        cc.bake()
        text(cc)
    refuses(stencil_of_temp, "Temporary passed to stencil", (Exception,))  # EE's core refuses it already

    def reach() -> None:
        cc = AMReXModule("R", arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                         binder=lambda base: None, rhs_binder=FieldBinder("state", lambda c: f"c_{c}"),
                         coordinates=CellCentered(), mode=KernelLaunchMode(),
                         params=MemberParams(ctor_args=[CtorArg("amrex::Real", "a_dx")], dx="a_dx"),
                         derivative_stencil_width=7, max_stencil_offset=2)
        r = cc.output("u_rhs", [])
        u = cc.decl("u", [], rhs=r)
        f = cc.create_function("f")
        f.add_eqn(r, D(u, l0))
        cc.bake()
        text(cc)
    refuses(reach, "max_stencil_offset")

    refuses(lambda: AMReXModule("C", arrays=[], binder=lambda b: None, coordinates=CellCentered(),
                                mode=PointFunctionMode(), params=MemberParams(ctor_args=[], dx="a")),
            "Unsupported combination")

    def read_output_in_point_mode() -> None:
        cc = point_module()
        u = cc.decl("u", [])
        a = cc.output("a", [])
        b = cc.output("b", [])
        f = cc.create_function("f")
        f.add_eqn(a, u)
        g = cc.create_function("g")
        g.add_eqn(b, a + u)
        cc.bake()
        text(cc)
    refuses(read_output_in_point_mode, "reads only fields")


def test_guards() -> None:
    f, S, T, c = sy.symbols('f S T c')
    g = Piecewise((f, f > 0), (0, True))
    assert fold_guards(g * S) == Piecewise((f * S, f > 0), (0, True))
    assert fold_guards(3 * g * S * T + 2 * f) == 2 * f + Piecewise((3 * S * T * f, f > 0), (0, True))
    assert fold_guards(Piecewise((0, c > 0), (f, True)) * S) == Piecewise((0, c > 0), (f * S, True))
    e = S + T
    assert fold_guards(e) is e

    # A guard held in a temporary is not folded: the post-CSE check raises.
    def guard_temp() -> None:
        cc = point_module()
        u = cc.decl("u", [])
        eps = cc.add_param("eps")
        gt = cc.decl("gt", [])
        o = cc.output("o", [])
        fn = cc.create_function("f")
        fn.add_eqn(gt, Piecewise((eps, eps > 0), (0, True)))
        fn.add_eqn(o, gt * D(u, l0, l0))
        cc.bake()
        text(cc)
    refuses(guard_temp, "multiplies a guard")

    # A recipe-supplied CSE list cannot drop the fold, and neither can do_cse=False.
    for opts in ({'cse_optimizations': []}, {'do_cse': False}):
        cc = point_module()
        u = cc.decl("u", [])
        eps = cc.add_param("eps")
        o = cc.output("o", [])
        fn = cc.create_function("f")
        fn.add_eqn(o, Piecewise((eps, eps > 0), (0, True)) * D(u, l0, l0))
        cc.bake(**opts)  # type: ignore[arg-type]
        code = text(cc)
        assert '> 0' in code and '? ' in code, code


def test_printing() -> None:
    v = AMReXVisitor()
    from EinsteinEngine.emit.code.amrex.amrex_sympy_visitor import AMReXSympyVisitor
    sv = AMReXSympyVisitor(is_field=lambda n: False)
    x = sy.Symbol('x')
    p = lambda e: v.visit(sv.visit(e))  # noqa: E731
    assert p(sy.pi) == '3.141592653589793', p(sy.pi)
    assert p(sy.Float('1.0e-6')) == '1e-06', p(sy.Float('1.0e-6'))
    assert p(x ** 2) == '(x * x)', p(x ** 2)
    assert p(x ** 4) == '(x * x * x * x)', p(x ** 4)
    assert p(x ** -1) == '(1.0 / (x))', p(x ** -1)
    assert p(x ** -2) == '(1.0 / (x * x))', p(x ** -2)
    assert p(x ** -4) == '(1.0 / (x * x * x * x))', p(x ** -4)
    assert p(x ** 5) == 'pow(x, 5.0)', p(x ** 5)
    assert p(sy.sqrt(x)) == 'sqrt(x)', p(sy.sqrt(x))
    assert p(x ** Rational(-1, 2)) == '(1.0 / sqrt(x))', p(x ** Rational(-1, 2))
    assert p(x ** Rational(-3, 2)) == 'pow(x, -1.5)', p(x ** Rational(-3, 2))
    assert p(sy.Abs(x)) == 'amrex::Math::abs(x)', p(sy.Abs(x))
    assert p(sy.Max(x, 1)) == 'amrex::max(x, 1.0)' or p(sy.Max(x, 1)) == 'amrex::max(1.0, x)', p(sy.Max(x, 1))
    assert '&&' in p(sy.And(x > 0, x < 1)) and '(!(' in p(sy.Not(x > 0, evaluate=False)), p(sy.And(x > 0, x < 1))


def test_outputs_read_back() -> None:
    """An output that its own loop, or a later loop, reads is still written (review round 1, B: B1)."""
    cc = kernel_module()
    u_rhs = cc.output("u_rhs", [])
    u = cc.decl("u", [], rhs=u_rhs)
    w_rhs = cc.output("w_rhs", [])
    w = cc.decl("w", [], rhs=w_rhs)
    f = cc.create_function("f")
    f.add_eqn(u_rhs, D(u, l0) + w)
    f.add_eqn(w_rhs, 2 * u_rhs - w)  # the Gamma-driver shape: B_rhs = G_rhs - eta*B
    cc.bake()
    code = text(cc)
    expect(code, 'rhs_ptr[static_cast<std::ptrdiff_t>(c_u) * rn] = u_rhs;', 'rhs_ptr[static_cast<std::ptrdiff_t>(c_w) * rn] =')

    cc = kernel_module()
    u_rhs = cc.output("u_rhs", [])
    u = cc.decl("u", [], rhs=u_rhs)
    w_rhs = cc.output("w_rhs", [])
    w = cc.decl("w", [], rhs=w_rhs)
    f = cc.create_function("f")
    f.add_eqn(u_rhs, D(u, l0) + w)
    f.split_loop()
    f.add_eqn(w_rhs, 2 * u_rhs - w)  # written in loop 0, read in loop 1
    cc.bake()
    code = text(cc)
    assert code.count('rhs_ptr[static_cast<std::ptrdiff_t>(c_u) * rn] =') == 1, code
    expect(code, 'rhs_ptr[static_cast<std::ptrdiff_t>(c_w) * rn] =')

    for outputs in ('refs', 'struct'):
        cc = point_module(outputs=outputs)
        u = cc.decl("u", [])
        a = cc.output("a", [])
        b = cc.output("b", [])
        f = cc.create_function("f")
        f.add_eqn(a, D(u, l0))
        f.add_eqn(b, 2 * a + u)
        cc.bake()
        code = text(cc)
        target = 'out.a' if outputs == 'struct' else 'a'
        expect(code, f'{target} = ee_a;')
        if outputs == 'refs':
            expect(code, 'amrex::Real & a', 'amrex::Real & b')

    def unwritten() -> None:
        cc = point_module()
        u = cc.decl("u", [])
        a = cc.output("a", [])
        cc.output("never", [])
        f = cc.create_function("f")
        f.add_eqn(a, u)
        cc.bake()
        text(cc)
    refuses(unwritten, "declared but no function writes them")


def test_declarations() -> None:
    """Classification by declaration is unique, and the generated code's own names are reserved (review round 1)."""
    def assign_field() -> None:
        cc = point_module()
        u = cc.decl("u", [])
        o = cc.output("o", [])
        f = cc.create_function("f")
        f.add_eqn(u, sy.Integer(3))
        f.add_eqn(o, u * u)
        cc.bake()
        text(cc)
    refuses(assign_field, "is a field", (Exception,))

    def param_then_field() -> None:
        cc = point_module()
        cc.add_param("u")
        cc.decl("u", [])
    refuses(param_then_field, "already declared as a parameter")

    def field_then_param() -> None:
        cc = point_module()
        cc.decl("trK", [])
        cc.add_param("trK")
    refuses(field_then_param, "already declared as a variable")

    refuses(lambda: point_module().add_param("DXI"), "reserved")
    refuses(lambda: point_module().add_param("dx"), "reserved")
    refuses(lambda: kernel_module().add_param("sy", init="1.0"), "reserved")
    refuses(lambda: kernel_module().decl("rn", []), "reserved")
    refuses(lambda: kernel_module().decl("in_u", []), "component pointers")
    refuses(lambda: point_module(outputs='struct').output("out", []), "reserved")

    def rhs_twice() -> None:
        cc = kernel_module()
        r = cc.output("u_rhs", [])
        cc.decl("u", [], rhs=r)
        cc.decl("w", [], rhs=r)
    refuses(rhs_twice, "is already the rhs of u")

    refuses(lambda: AMReXModule("P", arrays=[ArrayParam("s", "amrex::Array4<amrex::Real const> const &",
                                                        StridedPointer())],
                                binder=lambda b: None, coordinates=NodeCentered(), mode=PointFunctionMode(),
                                params=StructParams(arg="parameters", cpp_type="const Par &")), "Array4Call")
    refuses(lambda: AMReXModule("P", arrays=[], binder=lambda b: None, coordinates=CellCentered(),
                                mode=PointFunctionMode(), params=StructParams(arg="parameters", cpp_type="const Par &")),
            "Unsupported combination")


def test_numbers() -> None:
    """Section 6's number and function rules (review round 1)."""
    v = AMReXVisitor()
    from EinsteinEngine.emit.code.amrex.amrex_sympy_visitor import AMReXSympyVisitor
    sv = AMReXSympyVisitor(is_field=lambda n: False)
    x = sy.Symbol('x')
    p = lambda e: v.visit(sv.visit(e))  # noqa: E731
    assert p(sy.E) == '2.718281828459045', p(sy.E)
    assert p(sy.exp(1) * x) in ('(2.718281828459045 * x)', '(x * 2.718281828459045)'), p(sy.exp(1) * x)
    assert p(Rational(1, 3)) == '(1.0 / 3.0)', p(Rational(1, 3))
    big = Rational(2 ** 60 + 1, 3)
    assert p(big) == repr(float(2 ** 60 + 1) / 3.0) or p(big) == repr((2 ** 60 + 1) / 3), p(big)
    from fractions import Fraction
    assert p(big) == repr(float(Fraction(2 ** 60 + 1, 3))), p(big)
    assert p(x ** Rational(2 ** 60 + 1, 2 ** 59)) == f'pow(x, {float(Fraction(2 ** 60 + 1, 2 ** 59))!r})', \
        p(x ** Rational(2 ** 60 + 1, 2 ** 59))
    refuses(lambda: p(sy.Float('0.1', 3)), "bits of precision")
    refuses(lambda: p(sy.GoldenRatio * x), "not supported")
    refuses(lambda: p(sy.Function('foo')(x)), "undefined function foo")


def test_guards_round1() -> None:
    """A Float zero branch is a guard; a guard carried across a hard split is checked; the hoisting warning is
    transitive (review round 1)."""
    import contextlib
    import io
    zero = sy.Float(0.0)
    f, S = sy.symbols('f S')
    assert fold_guards(Piecewise((f, f > 0), (zero, True)) * S) == Piecewise((f * S, f > 0), (zero, True))

    def float_zero_guard() -> None:
        cc = point_module()
        u = cc.decl("u", [])
        eps = cc.add_param("eps")
        gt = cc.decl("gt", [])
        o = cc.output("o", [])
        fn = cc.create_function("f")
        fn.add_eqn(gt, Piecewise((eps, eps > 0), (zero, True)))
        fn.add_eqn(o, gt * D(u, l0, l0))
        cc.bake()
        text(cc)
    refuses(float_zero_guard, "multiplies a guard")

    def guard_across_split() -> None:
        cc = kernel_module()
        kappa = cc.add_param("kappa", init="1.5")
        r = cc.output("u_rhs", [])
        u = cc.decl("u", [], rhs=r)
        gt = cc.decl("gt", [])
        fn = cc.create_function("f")
        fn.add_eqn(gt, Piecewise((kappa, kappa > 1), (0, True)))
        fn.split_loop()
        fn.add_eqn(r, gt * D(u, l0, l0))
        cc.bake()
        text(cc)
    refuses(guard_across_split, "multiplies a guard")

    cc = point_module()
    u = cc.decl("u", [])
    eps = cc.add_param("eps")
    o = cc.output("o", [])
    fn = cc.create_function("f")
    fn.add_eqn(o, Piecewise((eps, eps > 0), (0, True)) * (D(u, l0, l0) + D(u, l1, l1)) + u)
    cc.bake()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        text(cc)
    assert 'used only inside guarded branches' in buf.getvalue(), buf.getvalue()


def test_golden_and_determinism() -> None:
    """The gate toys' headers match the golden copies, byte for byte, under PYTHONHASHSEED 0, 1 and 7 (section 9,
    tests 2 and 5; GRTeclyn's header is checked by the regression of section 10)."""
    import os
    import subprocess
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for seed in ('0', '1', '7'):
        with tempfile.TemporaryDirectory() as out:
            env = {**os.environ, 'PYTHONHASHSEED': seed, 'PYTHONPATH': root}
            subprocess.run([os.sys.executable, os.path.join(here, 'amrex', 'gate.py'), 'generate', out], env=env,
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for name in ('KernelToy.hpp', 'PointToy.H'):
                with open(os.path.join(out, name)) as a, open(os.path.join(here, 'amrex', 'golden', name)) as b:
                    assert a.read() == b.read(), f'{name} differs from golden/{name} under PYTHONHASHSEED={seed}'


def main() -> None:
    test_kernel_mode()
    test_point_mode('refs')
    test_point_mode('struct')
    test_refusals()
    test_guards()
    test_printing()
    test_outputs_read_back()
    test_declarations()
    test_numbers()
    test_guards_round1()
    test_golden_and_determinism()
    print('test_amrex_backend: OK')


if __name__ == '__main__':
    main()
