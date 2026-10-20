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
Tests of the AthenaK backend (frontend.dsl.athenak, generators.athenak_generator, emit.code.athenak): grid
variables map to view(m, enum, k, j, i) at their stencil offsets, parameters are initialized on the host and only
emitted when used, each function is one par_for in creation order, a loop may update its own outputs at offset 0,
and the generator rejects what AthenaK's par_for cannot run (stencil reads of a variable written in the same loop,
hard splits). Run as a plain script.
"""

from typing import Callable

from sympy import Piecewise, Rational

from EinsteinEngine import *
from EinsteinEngine.emit.code.athenak.athenak_visitor import AthenaKVisitor
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.generators.athenak_generator import AthenaKGenerator
from EinsteinEngine.generators.generator_exception import GeneratorException


def kernel(name: str) -> AthenaKKernel:
    return AthenaKKernel(
        name,
        args=[("MeshBlockPack *", "pmy_pack")],
        preamble=["auto &u0 = pmy_pack->u0;", "auto &rhs = pmy_pack->rhs;"],
        loop_bounds="0, nmb - 1, ks, ke, js, je, is, ie",
        inv_spacing=("1/dx(m)", "1/dy(m)", "1/dz(m)"),
        checks=[AthenaKCheck("bad", "not implemented")],
        derivative_stencil_width=3,
    )


def emit(kk: AthenaKKernel) -> str:
    kk.bake(do_cse=False)
    return str(AthenaKVisitor().visit(AthenaKGenerator(kk).generate_kernel_function("// test")))


def expect(exc: type[Exception], fn: Callable[[], object], *what: str) -> None:
    """fn raises exc (or a subclass) with one of the messages `what`."""
    try:
        fn()
    except exc as e:
        assert any(w in str(e) for w in what), e
        return
    raise AssertionError(f"expected {exc.__name__} ({what})")


def test_loads_stores_params() -> None:
    kk = kernel("K1")
    a = kk.add_param("a", init="opt.a")
    kk.add_param("unused", init="opt.unused")
    u = kk.decl("u", [], view="u0", enum="I_U")
    v = kk.decl("v", [ui], view="u0", enum="I_V")
    u_rhs = kk.decl("u_rhs", [], view="rhs", enum="I_U")
    fn = kk.create_function("f", label="my label")
    pw = Piecewise((v[u1], v[u1] < 0), (2 * v[u1], True))  # type: ignore[no-untyped-call]
    fn.add_eqn(u_rhs, a * D(u, l0) + pw * D(u, l2, l2))
    code = emit(kk)
    assert 'par_for("my label", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,' in code, code
    assert "u0(m, I_U, k, j, i + 1)" in code and "u0(m, I_U, k, j, i - 1)" in code, code
    assert "u0(m, I_U, k + 1, j, i)" in code and "u0(m, I_U, k - 1, j, i)" in code, code
    assert "u0(m, I_VY, k, j, i)" in code, code
    assert "rhs(m, I_U, k, j, i) = " in code, code
    assert "const Real a = opt.a;" in code and "unused" not in code, code
    assert "const Real DXI = 1/dx(m);" in code, code
    assert 'if (bad) {' in code and 'not implemented' in code, code
    assert " ? " in code and " < 0" in code, code


def test_two_loops_and_update() -> None:
    kk = kernel("K2")
    u = kk.decl("u", [], view="u0", enum="I_U")
    w = kk.decl("w", [li, lj], symmetries=[(li, lj)], view="aux", enum="I_W")
    u_rhs = kk.decl("u_rhs", [], view="rhs", enum="I_U")
    f1 = kk.create_function("first")
    f1.add_eqn(w[li, lj], D(u, li) * D(u, lj))
    f2 = kk.create_function("second")
    f2.add_eqn(u_rhs, D(w[l0, l1], l0) + Rational(1, 2) * u)  # stencil read of an earlier loop's output: allowed
    f3 = kk.create_function("third")
    up = kk.overwrite(u_rhs)
    f3.add_eqn(up, u_rhs + u)  # update of this loop's own output at offset 0: allowed
    code = emit(kk)
    assert code.index('"EE K2 first"') < code.index('"EE K2 second"') < code.index('"EE K2 third"'), code
    assert "aux(m, I_WXY, k, j, i) = " in code and "aux(m, I_WXY, k, j, i + 1)" in code, code
    assert "rhs(m, I_U, k, j, i) = (" in code.split('"EE K2 third"')[1], code


def test_rejects_same_loop_stencil_read() -> None:
    kk = kernel("K3")
    u = kk.decl("u", [], view="u0", enum="I_U")
    w = kk.decl("w", [], view="aux", enum="I_W")
    u_rhs = kk.decl("u_rhs", [], view="rhs", enum="I_U")
    fn = kk.create_function("f")
    fn.add_eqn(w, u * u)
    fn.add_eqn(u_rhs, D(w, l0))
    # EE's bake already refuses it (the read sees a temporary); the generator's own check backs that up.
    expect(Exception, lambda: emit(kk), "would race", "Temporary passed to stencil")


def test_rejects_hard_split() -> None:
    kk = kernel("K4")
    u = kk.decl("u", [], view="u0", enum="I_U")
    t = kk.decl("t", [])
    u_rhs = kk.decl("u_rhs", [], view="rhs", enum="I_U")
    fn = kk.create_function("f")
    fn.add_eqn(t, u * u)
    fn.split_loop()
    fn.add_eqn(u_rhs, D(t, l0))
    expect(GeneratorException, lambda: emit(kk), "hard split")


def test_decl_needs_view_and_enum() -> None:
    kk = kernel("K5")
    expect(DslException, lambda: kk.decl("u", [], view="u0"), "both view and enum")


if __name__ == "__main__":
    for test in [test_loads_stores_params, test_two_loops_and_update, test_rejects_same_loop_stencil_read,
                 test_rejects_hard_split, test_decl_needs_view_and_enum]:
        test()
        print(f"ok: {test.__name__}")
    print("All AthenaK backend tests passed.")
