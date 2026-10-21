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

    # MHDuet's backend review A, finding 7 (mhduet/analysis/reviews/backend-A-work/guard_probe.py): a sum or a power
    # of guards is zero when they all are false, so it must not multiply a non-constant factor; A + P is not a guard.
    from EinsteinEngine.generators.amrex.guards import check_guards
    a_, b_ = sy.symbols('a b')
    g1 = Piecewise((f, f > 0), (0, True))
    g2 = Piecewise((a_, a_ > 0), (0, True))
    for label, e in (("(g1 + g2) * S", (g1 + g2) * S), ("2*(g1 + 3*g2) * S", 2 * (g1 + 3 * g2) * S)):
        refuses(lambda: check_guards({sy.Symbol('o'): fold_guards(e)}, set(), label), "multiplies a guard")
    assert fold_guards(g1 ** 2 * S) == Piecewise((S * f ** 2, f > 0), (0, True))  # sympy squares the branches: folded
    gt = sy.Symbol('gt')
    refuses(lambda: check_guards({gt: g1, sy.Symbol('o'): gt ** 2 * S}, {gt}, "gt**2 * S"), "multiplies a guard")
    check_guards({sy.Symbol('o'): fold_guards((a_ + g1) * S)}, set(), "A + P")  # safe: A where P is false
    check_guards({sy.Symbol('o'): fold_guards(g1 * S)}, set(), "P * S")  # folded

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


def test_round2() -> None:
    """Implementation review round 2: guards recognized by their property; constructor names; one spacing owner;
    array and local names; number limits; no false guard errors across functions; tensor output order."""
    from EinsteinEngine.generators.amrex.guards import check_guards
    f, a, S, n, c, DXI = sy.symbols('f a S n c DXI')
    g1 = Piecewise((f, f > 0), (0, True))
    g2 = Piecewise((a, a > 0), (0, True))
    o, t = sy.Symbol('o'), sy.Symbol('t')
    guards = {'Abs(g1)': sy.Abs(g1), 'Abs(g1 + g2)': sy.Abs(g1 + g2), 'sin(g1)': sy.sin(g1), 'tanh(g1)': sy.tanh(g1),
              'Max(g1, g2)': sy.Max(g1, g2), 'PW(g1, g2)': Piecewise((g1, c > 0), (g2, True)), 'g1**n': g1 ** n,
              'g1 + g2': g1 + g2}
    for label, e in guards.items():
        refuses(lambda: check_guards({o: e * S}, set(), label), "multiplies a guard")
    refuses(lambda: check_guards({t: sy.Abs(g1), o: t * S}, {t}, "temp"), "multiplies a guard")
    for label, e in {'a + g1': a + g1, 'g1 + 1': g1 + 1, 'exp(g1)': sy.exp(g1), 'cos(g1)': sy.cos(g1),
                     'DXI*(a + g1)': DXI * (a + g1)}.items():
        check_guards({o: e * S}, set(), label)
    # A temporary that is a guard in one function and not in another; an output overwritten with a non-guard.
    cc = kernel_module()
    kappa = cc.add_param("kappa", init="1.5")
    r = cc.output("u_rhs", [])
    u = cc.decl("u", [], rhs=r)
    rw = cc.output("w_rhs", [])
    w = cc.decl("w", [], rhs=rw)
    tt = cc.decl("tt", [])
    f1 = cc.create_function("f1")
    f1.add_eqn(r, Piecewise((kappa, kappa > 1), (0, True)) * D(u, l0, l0))
    f1.add_eqn(tt, Piecewise((kappa, kappa > 2), (0, True)))  # a guard in f1 ...
    f1.add_eqn(rw, tt + w)
    f2 = cc.create_function("f2")
    f2.add_eqn(cc.overwrite(r), r + u)  # u_rhs is no longer a guard
    f3 = cc.create_function("f3")
    f3.add_eqn(tt, D(u, l1))  # ... and an unrelated value in f3
    f3.add_eqn(cc.overwrite(rw), r * tt + w)
    cc.bake()
    text(cc)
    # The fold keeps a number times a sum a product.
    b_ = sy.Symbol('b')
    folded = fold_guards(Rational(1, 3) * g1 * (a + b_))
    assert folded.args[0][0] != f * a / 3 + f * b_ / 3, folded
    # Constructor arguments and prelude identifiers are not parameter names.
    def ctor_arg_param() -> None:
        cc = AMReXModule("K", arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                         binder=lambda base: None, rhs_binder=FieldBinder("state", lambda c: f"c_{c}"),
                         coordinates=CellCentered(), mode=KernelLaunchMode(),
                         params=MemberParams(ctor_args=[CtorArg("amrex::Real", "a_dx"), CtorArg("amrex::Real", "sigma")],
                                             dx="a_dx"))
        cc.add_param("sigma", init="sigma")
    refuses(ctor_arg_param, "reserved")
    def prelude_param() -> None:
        cc = AMReXModule("K", arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                         binder=lambda base: None, rhs_binder=FieldBinder("state", lambda c: f"c_{c}"),
                         coordinates=CellCentered(), mode=KernelLaunchMode(),
                         params=MemberParams(ctor_args=[CtorArg("amrex::Real", "a_dx")], dx="a_dx",
                                             ctor_prelude=['amrex::Real eta{};', 'pp.get("gauge.eta", eta);']))
        cc.add_param("eta", init="eta")
    refuses(prelude_param, "constructor prelude")
    # A name inside a prelude string literal is not an identifier.
    cc = AMReXModule("K", arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                     binder=lambda base: None, rhs_binder=FieldBinder("state", lambda c: f"c_{c}"),
                     coordinates=CellCentered(), mode=KernelLaunchMode(),
                     params=MemberParams(ctor_args=[CtorArg("amrex::Real", "a_dx")], dx="a_dx",
                                         ctor_prelude=['amrex::Real a_sigma{};', 'pp.get("evolution.sigma", a_sigma);']))
    cc.add_param("sigma", init="a_sigma")
    # One owner of the spacing.
    refuses(lambda: AMReXModule("P", arrays=[], binder=lambda b: None, coordinates=NodeCentered(dx='h'),
                                mode=PointFunctionMode(), params=StructParams(cpp_type="const Par &")), "same spacing")
    refuses(lambda: AMReXModule("K", arrays=[], binder=lambda b: None, coordinates=CellCentered(spacing='H'),
                                mode=KernelLaunchMode(), params=MemberParams(ctor_args=[], dx="a")), "spacing member")
    # Array names and generated locals.
    refuses(lambda: AMReXModule("P", arrays=[ArrayParam("out", "amrex::Array4<amrex::Real const> const &", Array4Call())],
                                binder=lambda b: None, coordinates=NodeCentered(), mode=PointFunctionMode(),
                                params=StructParams(cpp_type="const Par &")), "reserved")
    refuses(lambda: point_module().output("ee_a", []), "generated locals")
    def binders_disagree() -> None:
        cc = AMReXModule("K", arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                         binder=lambda base: FieldBinder("state", lambda c: f"v::{c}") if base == "u" else None,
                         rhs_binder=FieldBinder("state", lambda c: f"c_{c}"), coordinates=CellCentered(),
                         mode=KernelLaunchMode(), params=MemberParams(ctor_args=[CtorArg("amrex::Real", "a_dx")],
                                                                     dx="a_dx"))
        r = cc.output("u_rhs", [])
        cc.decl("u", [], rhs=r)
    refuses(binders_disagree, "bind it differently")
    # Number limits.
    v = AMReXVisitor()
    from EinsteinEngine.emit.code.amrex.amrex_sympy_visitor import AMReXSympyVisitor
    sv = AMReXSympyVisitor(is_field=lambda n: False)
    x = sy.Symbol('x')
    p = lambda e: v.visit(sv.visit(e))  # noqa: E731
    big = sy.Integer(2 ** 70) * x
    assert p(big) in (f'({float(2 ** 70)!r} * x)', f'(x * {float(2 ** 70)!r})'), p(big)
    assert p(sy.Integer(3) * x) == '(3 * x)', p(sy.Integer(3) * x)
    refuses(lambda: p(Rational(10 ** 400, 3)), "not a finite double")
    refuses(lambda: p(Rational(1, 10 ** 400)), "underflows")
    # cse_optimizations=None means the default list.
    cc = point_module()
    u = cc.decl("u", [])
    o2 = cc.output("o", [])
    fn = cc.create_function("f")
    fn.add_eqn(o2, D(u, l0))
    cc.bake(cse_optimizations=None)  # type: ignore[arg-type]
    text(cc)
    # Tensor outputs: the point signature lists components in the module's order, which the header states.
    cc = point_module()
    _, li = cc.mk_pair("i")
    _, lj = cc.mk_pair("j")
    u = cc.decl("u", [])
    hr = cc.output("hr", [li, lj], symmetries=[(li, lj)])
    fn = cc.create_function("f")
    fn.add_eqn(hr[li, lj], D(u, li, lj))
    cc.bake()
    code = text(cc)
    expect(code, '// Outputs, in order: hr11, hr12, hr22, hr13, hr23, hr33.')


def test_round3() -> None:
    """Implementation review round 3: zero forms through temporaries (cancellations), guards in one branch of a
    choice, parameters read before they are set, prelude identifiers, binder agreement, Float underflow."""
    from EinsteinEngine.generators.amrex.guards import check_guards
    f, S, a, b, k, c2, A, DXI = sy.symbols("f S a b kappa c2 A DXI")
    g = Piecewise((f, f > 0), (0, True))
    h = Piecewise((a, a > 0), (0, True))
    o, t = sy.Symbol('o'), sy.Symbol('t')
    rows = [  # reviewer B's round-3 table (review-impl-B/r3_fixsketch.py) and reviewer A's B6
        ('pass', {o: 2 * g * 3}), ('pass', {o: g * S}), ('pass', {o: (A + g) * S}), ('pass', {o: DXI * (A + g)}),
        ('pass', {o: (g + 1) * S}), ('pass', {o: sy.exp(g) * S}), ('pass', {o: sy.cos(g) * S}),
        ('pass', {t: g, o: (t + A) * S}), ('pass', {t: sy.exp(g), o: t * S}), ('pass', {t: 1 + g, o: (t + A) * S}),
        ('pass', {o: g / 2 + h / 3}), ('raise', {o: sy.Max(g, b) * S}),  # Max(0, b) is 0 where b <= 0 (round 4)
        ('raise', {o: sy.Abs(g) * S}), ('raise', {o: sy.Max(g, h) * S}), ('raise', {o: g ** k * S}),
        ('raise', {o: Piecewise((g, c2 > 0), (h, True)) * S}), ('raise', {o: Piecewise((g, c2 > 0), (b, True)) * S}),
        ('raise', {t: Piecewise((g, c2 > 0), (b, True)), o: t * S}), ('raise', {o: Piecewise((b, b > g), (g, True)) * S}),
        ('raise', {t: sy.exp(g), o: (t - 1) * S}), ('raise', {t: 1 + g, o: (t - 1) * S}),
        ('raise', {o: (sy.exp(g) - 1) * S}), ('raise', {t: sy.Abs(g), o: t * S}),
        ('raise', {o: sy.Abs(Piecewise((g, c2 > 0), (A, True))) * S}),
        # Round 4 (A B7 and N28, B B1 and B2): copies, a tile across a split, a choice whose guarded branch cancels,
        # 0**n stored exactly, Min, cancellation after expansion.
        ('raise', {t: 1 + g ** k, o: (t - 1) * S}), ('pass', {o: (1 + g ** k) * S}),
        ('raise', {o: (Piecewise((1 + g, c2 > 1), (b, True)) - 1) * S}), ('raise', {o: sy.Min(g, b) * S}),
        ('raise', {o: (sy.exp(g) * A * (b + 1) - A * b - A) * S}),
        ('raise', {t: A * b, sy.Symbol('t1'): t, o: ((t + g) - sy.Symbol('t1')) * S}),
    ]
    for want, eqns in rows:
        folded = {lhs: fold_guards(rhs) for lhs, rhs in eqns.items()}
        if want == 'raise':
            refuses(lambda: check_guards(folded, {t}, str(eqns)), "multiplies a guard")
        else:
            check_guards(folded, {t}, str(eqns))
    # Carried across a split: a temporary computed from a guard, cancelled in the next loop.
    def cancel_across_split() -> None:
        cc = kernel_module()
        kappa = cc.add_param("kappa", init="1.5")
        r = cc.output("u_rhs", [])
        u = cc.decl("u", [], rhs=r)
        tt = cc.decl("tt", [])
        fn = cc.create_function("f")
        fn.add_eqn(tt, 1 + Piecewise((kappa, kappa > 1), (0, True)))
        fn.split_loop()
        fn.add_eqn(r, (tt - 1) * D(u, l0, l0))
        cc.bake()
        text(cc)
    refuses(cancel_across_split, "multiplies a guard")
    # Parameters are assigned in name order, before the spacing: no init or check may read one.
    refuses(lambda: (lambda cc: (cc.add_param("beta", init="1.0"), cc.add_param("alpha", init="2.0 * beta")))(
        kernel_module()), "has not set")
    refuses(lambda: kernel_module().add_param("alpha", init="0.5 * DX"), "has not set")
    refuses(lambda: (lambda cc: (cc.add_param("alpha", init="2.0 * beta"), cc.add_param("beta", init="1.0")))(
        kernel_module()), "reads it, unset")
    def check_reads_param() -> None:
        cc = AMReXModule("K", arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                         binder=lambda base: None, rhs_binder=FieldBinder("state", lambda c: f"c_{c}"),
                         coordinates=CellCentered(), mode=KernelLaunchMode(),
                         params=MemberParams(ctor_args=[CtorArg("amrex::Real", "a_dx")], dx="a_dx",
                                             ctor_checks=[CtorCheck("kappa <= 0", "kappa must be positive")]))
        cc.add_param("kappa", init="1.5")
    refuses(check_reads_param, "before it is set")
    # Prelude identifiers ignore comments, members and numbers.
    cc = AMReXModule("K", arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                     binder=lambda base: None, rhs_binder=FieldBinder("state", lambda c: f"c_{c}"),
                     coordinates=CellCentered(), mode=KernelLaunchMode(),
                     params=MemberParams(ctor_args=[CtorArg("amrex::Real", "a_dx")], dx="a_dx",
                                         ctor_prelude=['params_t gauge{};  // eta is set below', 'gauge.eta = 1.0e-3;']))
    cc.add_param("eta", init="gauge.eta")
    cc.add_param("e", init="1.0")
    # A binder and rhs_binder that agree are accepted.
    cc = AMReXModule("K", arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                     binder=lambda base: FieldBinder("state", lambda c: f"c_{c}") if base == "u" else None,
                     rhs_binder=FieldBinder("state", lambda c: f"c_{c}"), coordinates=CellCentered(),
                     mode=KernelLaunchMode(), params=MemberParams(ctor_args=[CtorArg("amrex::Real", "a_dx")], dx="a_dx"))
    cc.decl("u", [], rhs=cc.output("u_rhs", []))
    # A Float that underflows is refused.
    from EinsteinEngine.emit.code.amrex.amrex_sympy_visitor import AMReXSympyVisitor
    sv = AMReXSympyVisitor(is_field=lambda n: False)
    refuses(lambda: sv.visit(sy.Float('1e-400', 30)), "underflows")


def test_round5() -> None:
    """Implementation review round 5: a guard larger than the inlining bound stays a guard (A B8, B B1); one shared
    placeholder per reassigned name (A B9); 0**n inside choices and after the fold (B B2); bounded cost: expansion,
    Max/Min and the fold (A B10, B B3)."""
    import time
    from EinsteinEngine.generators.amrex.guards import MAX_INLINE, check_guards, tree_size
    f, S, c, A, B, C, k, eps = sy.symbols('f S c A B C kappa eps')
    G = Piecewise((f, f > 0), (0, True))
    o, t, x0 = sy.Symbol('o'), sy.Symbol('t'), sy.Symbol('x0')
    big = sum(Rational(1, n) * A ** n * B ** (n % 7) * C ** (n % 5) for n in range(1, 60))
    assert tree_size(big) > MAX_INLINE
    rows = [
        # B8: a large guard, in a temporary or folded, then multiplied.
        ('raise', {t: Piecewise((eps * big, eps > 0), (0, True)), o: t * S}),
        ('raise', {t: fold_guards(G * big), o: t * S}),
        ('pass', {t: fold_guards(G * big), o: (t + A) * S}),
        ('raise', {x0: big, t: G * x0, o: t * S}),
        # B B2: 0**n inside a choice, under a function, and rebuilt (Abs(0**k)**k).
        ('raise', {o: sy.sin(sy.Abs(Piecewise((C, c > 0), (G, True))) ** k) * S}),
        ('raise', {o: sy.sin(Piecewise((G, c > 0), (2, True)) ** k) * S}),
        ('raise', {o: sy.Abs(sy.Min(G, sy.Abs(G) ** k)) ** k * S}),
        ('raise', {o: (2 * sy.Abs(G) ** k) ** k * S}),
        ('pass', {o: (sy.sin(Piecewise((G, c > 0), (2, True)) ** k) + 2) * S}),
    ]
    for want, eqns in rows:
        folded = {lhs: fold_guards(rhs) for lhs, rhs in eqns.items()}
        if want == 'raise':
            refuses(lambda: check_guards(folded, set(folded) - {o}, str(o)), "multiplies a guard")
        else:
            check_guards(folded, set(folded) - {o}, str(o))
    # B8 across a split: an output that is a large guard, carried by name.
    zf = check_guards({sy.Symbol('q_rhs'): fold_guards(G * big)}, set(), 'r0', carried=['q_rhs'])
    refuses(lambda: check_guards({o: sy.Symbol('q_rhs') * S}, set(), 'r1', zf), "multiplies a guard")
    # Cost (A B10, B B3): each of these legal or refused forms is decided within a few seconds.
    chain = {sy.Symbol('t0'): A + G}
    for i in range(1, 23):
        chain[sy.Symbol(f't{i}')] = sy.Symbol(f't{i - 1}') * (sy.Symbol(f'a{i}') + sy.Symbol(f'b{i}'))
    chain[o] = sy.Symbol('t22') * S
    T = sy.sin(A) * sy.exp(B) + sy.cos(C) ** 2
    cases = [('pass', chain), ('pass', {o: (sy.Max(G, Piecewise((G, c > 0), (T, True))) + 1) * S}),
             ('pass', {o: (A + G) * sy.Mul(*[(sy.Symbol(f'a{i}') + sy.Symbol(f'b{i}')) for i in range(18)]) * B})]
    for want, eqns in cases:
        t0 = time.time()
        check_guards(eqns, set(eqns) - {o}, 'cost')
        assert time.time() - t0 < 10, f'slow: {time.time() - t0:.1f} s'
    # Identifier scan (A N35): members through (*this)., and a quote inside a character literal.
    ids = AMReXModule.cpp_identifiers("char q = '\"'; alpha = eta; b = \"x\"; c = 2.0 * (*this).beta; n = 1'000;")
    assert {'alpha', 'eta', 'beta'} <= ids and 'x' not in ids, ids
    refuses(lambda: (lambda cc: (cc.add_param("alpha", init="2.0 * (*this).beta"), cc.add_param("beta", init="1.0")))(
        kernel_module()), "reads it, unset")
    # Round 6 (B B1, A N36): one choice shared by many terms is counted once by the fold's estimate (a radius floor in
    # every term of a guarded sum), and a sum with many upwinded terms beside a guard is not refused.
    rfloor = Piecewise((Rational(1, 10 ** 6), A < Rational(1, 10 ** 6)), (A, True))
    shared = sum(sy.Symbol(f'w{i}') * rfloor for i in range(12)) + G
    check_guards({o: (shared + B) * S}, set(), 'shared')
    up = sum(Piecewise((sy.Symbol(f'p{i}'), sy.Symbol(f'b{i % 3}') > 0), (sy.Symbol(f'm{i}'), True)) for i in range(12))
    check_guards({o: sy.Symbol('DXI') * (up + G) * S}, set(), 'upwind')
    # Round 6 (B B2): negative and fractional powers of sums are expanded by sympy too; the estimate bounds them.
    s12 = sum(sy.Symbol(f'z{i}') for i in range(12))
    t0 = time.time()
    check_guards({o: ((A + G) / s12 ** 12 + B) * S}, set(), 'negpow')
    check_guards({o: ((A + G) * s12 ** Rational(25, 2) + B) * S}, set(), 'fracpow')
    assert time.time() - t0 < 10, f'slow: {time.time() - t0:.1f} s'
    # Round 7 (A N39): a chain of 20 Max/Min around a guard folds to 21 branches, not 2**20, and passes with + 2.
    chain2 = {sy.Symbol('t0'): G}
    for i in range(1, 21):
        chain2[sy.Symbol(f't{i}')] = (sy.Min if i % 2 else sy.Max)(sy.Symbol(f't{i - 1}'), sy.Symbol(f'x{i}'))
    chain2[o] = (sy.Symbol('t20') + 2) * S
    t0 = time.time()
    check_guards(chain2, set(chain2) - {o}, 'maxchain')
    assert time.time() - t0 < 10, f'slow: {time.time() - t0:.1f} s'
    # Identifier scan (round 6, B): a character literal after a keyword, and a raw string.
    ids = AMReXModule.cpp_identifiers("switch (k) { case'a': w = R\"(gamma)\"; } h = 0x1F'FF;")
    assert 'a' not in ids and 'gamma' not in ids and 'R' not in ids and 'FF' not in ids, ids
    # Many choices around a guard are refused, loudly, rather than folded without end.
    many = sum(Piecewise((G, sy.Symbol(f'c{i}') > 0), (sy.Symbol(f'd{i}'), True)) for i in range(12))
    refuses(lambda: check_guards({o: many * S}, set(), 'many'), "combinations of choices")


def test_round5_reassigned() -> None:
    """Round 5, A B9: two carried forms that name the same old value share one placeholder, also when an unrelated
    overwrite or a function boundary sits between the writer and the reader (end to end, kernel mode)."""
    A_, B_ = sy.symbols('A_ B_')

    def build(middle: str) -> None:
        cc = kernel_module()
        eps = cc.add_param("eps", init="0.1")
        bb = cc.add_param("bb", init="2.0")
        outs = {}
        for nm in ('u', 'w', 'q', 'p'):
            r = cc.output(f"{nm}_rhs", [])
            outs[nm] = (cc.decl(nm, [], rhs=r), r)
        (u, ru), (w, rw), (q, rq), (p, rp) = (outs[x] for x in 'uwqp')
        G = Piecewise((eps, eps > 0), (0, True))
        H = Piecewise((bb, bb > 1), (0, True))
        big = sum(Rational(1, n) * u ** n * w ** (n % 7) * q ** (n % 5) for n in range(1, 60))
        f = cc.create_function("f")
        if middle == 'tile':
            tt = cc.decl("tt", [])
            f.add_eqn(tt, big)
            f.split_loop()
            f.add_eqn(ru, tt + G)
            f.add_eqn(rw, tt + H)
            f.add_eqn(rq, q)
        else:
            f.add_eqn(rq, big)
            f.add_eqn(ru, rq + G)
            f.add_eqn(rw, rq + H)
            if middle == 'overwrite':
                g = cc.create_function("g")
                g.add_eqn(cc.overwrite(rq), 2 * rq)
        h = cc.create_function("h")
        h.add_eqn(rp, (ru - rw) * D(p, l0, l0))
        cc.bake()
        text(cc)
    for middle in ('none', 'overwrite', 'tile'):
        refuses(lambda: build(middle), "multiplies a guard")

    # Round 6 (A N38): a copy of the old value in the overwrite's own loop (u - w = G must raise), and the must-pass
    # control against the new value (u - q' = G - q_old is not a guard).
    def copy_in_overwrite_loop(reader: str) -> None:
        cc = kernel_module()
        eps = cc.add_param("eps", init="0.1")
        outs = {}
        for nm in ('u', 'w', 'q', 'p'):
            r = cc.output(f"{nm}_rhs", [])
            outs[nm] = (cc.decl(nm, [], rhs=r), r)
        (u, ru), (w, rw), (q, rq), (p, rp) = (outs[x] for x in 'uwqp')
        G = Piecewise((eps, eps > 0), (0, True))
        big = sum(Rational(1, n) * u ** n * w ** (n % 7) * q ** (n % 5) for n in range(1, 60))
        f = cc.create_function("f")
        f.add_eqn(rq, big)
        f.add_eqn(ru, rq + G)
        g = cc.create_function("g")
        g.add_eqn(rw, rq)
        g.add_eqn(cc.overwrite(rq), 2 * rq)
        h = cc.create_function("h")
        h.add_eqn(rp, (ru - (rw if reader == 'w' else rq)) * D(p, l0, l0))
        cc.bake()
        text(cc)
    refuses(lambda: copy_in_overwrite_loop('w'), "multiplies a guard")
    copy_in_overwrite_loop('q')


def test_random_guards(trials: int = 300) -> None:
    """Review round 4 (B): random equation systems with guards. Wherever the output is NaN with the guards false and
    S infinite at two generic points of the other symbols, but not with the guards true (a guard-caused 0 * S), the
    check must raise, in one loop and after a split at a random point (the first part's zero forms carried, its values
    read by name). Round 5 (A N34, B): Min among the operations, and at the split only a random half of the first
    part's values carried by name (the others are locals: the second part reads their values). Symbolic powers, whose
    exact evaluation here is slow, are covered by test_round5."""
    import contextlib
    import io
    import random
    from EinsteinEngine.generators.amrex.guards import check_guards
    f, a, A, B, c, S = sy.symbols('f a A B c S')
    G1, G2 = Piecewise((f, f > 0), (0, True)), Piecewise((a, a > 0), (0, True))
    rng = random.Random(20261020)

    def expr(depth: int, temps: list) -> sy.Basic:
        if depth == 0 or rng.random() < 0.25:
            return rng.choice([G1, G2, A, B, sy.Integer(rng.randint(1, 3))] + temps)
        x, y = expr(depth - 1, temps), expr(depth - 1, temps)
        return [x + y, x - y, x * y, sy.Abs(x), sy.exp(x), sy.sin(x), sy.Max(x, y), Piecewise((x, c > 0), (y, True)),
                x ** 2, sy.Min(x, y)][rng.randrange(10)]

    def verdict(eqns: dict, split: int | None) -> str:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if split is None:
                    check_guards(eqns, set(eqns), 'r')
                else:
                    items = list(eqns.items())
                    first, second = dict(items[:split]), dict(items[split:])
                    keep = {str(x) for x in first if rng.random() < 0.5}
                    zf = check_guards(first, set(first), 'r0', carried=sorted(keep))
                    carried = {n: v for n, v in zf.items() if n in keep and not (isinstance(v, sy.Symbol) and str(v) == n)}
                    local = {x: v for x, v in first.items() if str(x) not in keep}
                    for _ in range(len(first)):  # locals are not readable by name in the second part: their values
                        second = {x: v.xreplace(local) for x, v in second.items()}
                    check_guards(second, set(second), 'r1', carried)
            return 'pass'
        except GeneratorException:
            return 'raise'

    def nan(eqns: dict, vals: dict) -> bool:
        env: dict = {}
        for lhs, rhs in eqns.items():
            env[lhs] = sy.sympify(rhs).xreplace(env).subs(vals)
        out = env[sy.Symbol('o')]
        return out is sy.nan or out.has(sy.nan)

    for _ in range(trials):
        temps: list = []
        eqns: dict = {}
        for k in range(rng.randint(0, 3)):
            t = sy.Symbol(f't{k}')
            eqns[t] = fold_guards(expr(2, temps))
            temps.append(t)
        eqns[sy.Symbol('o')] = fold_guards(expr(2, temps) * S)
        # Guard-caused: NaN with the guards false at two generic points (a zero of the data, as |B| - B at B > 0, is
        # outside R10: docs, section 7), and not with the guards true.
        points = ({A: Rational(7, 10), B: Rational(13, 10)}, {A: Rational(-6, 10), B: Rational(-17, 10)})
        hazard = any(all(nan(eqns, {**pt, c: cv, S: sy.oo, f: -1, a: -1}) for pt in points)
                     and not nan(eqns, {**points[0], c: cv, S: sy.oo, f: Rational(37, 100), a: Rational(53, 100)})
                     for cv in (1, -1))
        if hazard:
            assert verdict(eqns, None) == 'raise', f'escape: {eqns}'
            if len(eqns) > 1:
                assert verdict(eqns, rng.randint(1, len(eqns) - 1)) == 'raise', f'escape after a split: {eqns}'


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
    test_round2()
    test_round3()
    test_round5()
    test_round5_reassigned()
    test_random_guards()
    test_golden_and_determinism()
    print('test_amrex_backend: OK')


if __name__ == '__main__':
    main()
