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
Tests that floating-point constants are emitted as literals that read back as exactly the intended double: pi as
3.141592653589793 (not sympy's 15-digit 3.14159265358979, 7 ulp low), every sympy Float as the double it denotes, in
the C++ (CarpetX) and Fortran printers, in valid syntax (Fortran exponents as D). Run as a plain script.
"""

import math
import re
from decimal import Decimal

import sympy as sy

from EinsteinEngine import *
from EinsteinEngine.emit.code.common.code_tree import FloatLiteralExpr
from EinsteinEngine.emit.code.common.literals import real_literal
from EinsteinEngine.emit.code.cpp_carpetx.cpp_carpetx_visitor import CppVisitor, cpp_real_literal
from EinsteinEngine.emit.code.f90.f90_visitor import F90Visitor, f90_real_literal
from EinsteinEngine.emit.code.sympy_visitor import BaseSympyExprVisitor
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ThornDef
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.generators.cpp_carpetx_generator import CppCarpetXGenerator
from EinsteinEngine.generators.vanilla_f90_generator import VanillaF90Generator


def Float(*args: object) -> sy.Float:
    """sympy.Float, typed (sympy's constructor is untyped)."""
    return sy.Float(*args)  # type: ignore[no-untyped-call]


# (value, expected C++ literal, expected Fortran literal), written out independently of the implementation.
EXPECTED = [
    (math.pi, "3.141592653589793", "3.141592653589793D0"),
    (0.1, "0.1", "0.1D0"),
    (1 / 3, "0.3333333333333333", "0.3333333333333333D0"),
    (2 / 3, "0.6666666666666666", "0.6666666666666666D0"),
    (4 / 3, "1.3333333333333333", "1.3333333333333333D0"),
    (1e-10, "1e-10", "1D-10"),
    (5e-324, "5e-324", "5D-324"),
    (1.7976931348623157e308, "1.7976931348623157e+308", "1.7976931348623157D308"),
    (2.0, "2.0", "2.0D0"),
    (1e22, "1e+22", "1D22"),
    (123.456, "123.456", "123.456D0"),
    (-0.1, "-0.1", "-0.1D0"),
    (-2.0, "-2.0", "-2.0D0"),
]

F90_REAL = re.compile(r"^-?[0-9]+(\.[0-9]*)?D-?[0-9]+$")


def reads_back(text: str, value: float) -> bool:
    """Whether the literal's exact decimal value rounds to `value` (Decimal parsing, independent of repr)."""
    return float(Decimal(text.replace("D", "e"))) == value


def test_sympy_visitor() -> None:
    v = BaseSympyExprVisitor()  # type: ignore[var-annotated]
    pi_node = v.visit(sy.pi)
    assert isinstance(pi_node, FloatLiteralExpr) and type(pi_node.fl) is float, pi_node
    assert pi_node.fl == 3.141592653589793, pi_node.fl
    # Floats whose 15-digit decimal string is a different double than the Float denotes.
    for f, expected in [(Float(1 / 3), 0.3333333333333333), (Float(4 / 3), 1.3333333333333333),
                        (Float("0.1", 30), 0.1), (Float("1e-300"), 1e-300), (Float(2.0), 2.0)]:
        node = v.visit(f)
        assert isinstance(node, FloatLiteralExpr) and type(node.fl) is float, (f, node)
        assert node.fl == expected, (f, node.fl, expected)
    try:
        v.visit(Float("1e400"))
    except DslException:
        pass
    else:
        raise AssertionError("a Float beyond the double range was accepted")


def test_literal_functions() -> None:
    for value, cpp, f90 in EXPECTED:
        assert cpp_real_literal(value) == cpp, (value, cpp_real_literal(value), cpp)
        assert f90_real_literal(value) == f90, (value, f90_real_literal(value), f90)
        assert real_literal(value) == cpp
        assert reads_back(cpp, value) and reads_back(f90, value), (value, cpp, f90)
        assert F90_REAL.match(f90), f90
    assert f90_real_literal(3) == "3.0D0"  # an int (e.g. a parameter default) is printed as a double
    for bad in (math.inf, -math.inf, math.nan):
        for fn in (cpp_real_literal, f90_real_literal):
            try:
                fn(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{fn.__name__}({bad}) did not raise")


def test_f90_visitor() -> None:
    v = F90Visitor()
    for value, _, f90 in EXPECTED:
        assert v.visit(FloatLiteralExpr(value)) == f90


def test_cpp_end_to_end() -> None:
    gf = ThornDef("TestFloatLiterals", "FloatLits", derivative_stencil_width=5)
    u_t = gf.decl("u_t", [], centering=Centering.VVV)
    u = gf.decl("u", [], centering=Centering.VVV, rhs=u_t)
    fn = gf.create_function("rhs", ScheduleBin.Evolve)
    fn.add_eqn(u_t, sy.pi * u + Float(1 / 3) * u ** 2)
    gf.bake()
    gen = CppCarpetXGenerator(gf)
    code = CppVisitor(gen).visit(gen.generate_function_code("rhs"))
    assert "3.141592653589793" in code, code
    assert "0.3333333333333333" in code, code
    # the old printer's 15-digit forms
    assert not re.search(r"3\.14159265358979(?!3)", code), code
    assert not re.search(r"0\.333333333333333(?!3)", code), code


def test_f90_end_to_end() -> None:
    m = VanillaF90Module("float_lits")
    fn = m.create_function("rhs")
    a = m.decl("a", [])
    b = m.decl("b", [])
    fn.add_eqn(b, sy.pi * a + Float("1e-10") * a ** 2 + Float(1 / 3) * a ** 3)
    m.bake()
    gen = VanillaF90Generator(m)
    code = F90Visitor().visit(gen.generate_submodule_code("rhs"))
    assert "3.141592653589793D0" in code, code
    assert "1D-10" in code, code
    assert "0.3333333333333333D0" in code, code
    # the old printer's forms: 15 digits, and an E exponent followed by D0 (invalid Fortran)
    assert "3.14159265358979D0" not in code and not re.search(r"[0-9]E-?[0-9]+D0", code), code


if __name__ == "__main__":
    test_sympy_visitor()
    test_literal_functions()
    test_f90_visitor()
    test_cpp_end_to_end()
    test_f90_end_to_end()
    print("test_float_literal_printing: all tests passed")
