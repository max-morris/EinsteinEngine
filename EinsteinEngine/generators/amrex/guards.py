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
Guards (docs/AMREX-BACKEND-DESIGN.md, section 7): a product `Piecewise((a, c), (0, True)) * S` would compute `0 * S`,
which is NaN when S is not finite. Before CSE, `fold_guards` moves the other factors into the non-zero branch,
`Piecewise((a*S, c), (0, True))`. After CSE, `check_guards` raises on any surviving product of a guard with a
non-constant factor (a hard error in both modes), and reports, as a warning, arithmetic temporaries used only inside
guarded branches (work that a false guard does not skip).
"""

from typing import Collection, Mapping, cast

import sympy as sy
from sympy import Expr, Mul, Piecewise, Symbol

from EinsteinEngine.common.sympywrap import CseOptimization, free_symbols
from EinsteinEngine.common.util import wprint
from EinsteinEngine.generators.generator_exception import GeneratorException


def _pieces(expr: sy.Basic) -> list[tuple[sy.Basic, sy.Basic]]:
    """A Piecewise's (expression, condition) pairs."""
    return [(piece.args[0], piece.args[1]) for piece in expr.args]


def _is_zero(e: sy.Basic) -> bool:
    """Exactly zero, as an Integer or a Float (sympy's Float(0.0) == 0 is False)."""
    return e.is_Number is True and e.is_zero is True


def _is_zero_like(e: sy.Basic) -> bool:
    """Zero, or 0**n with n symbolic (zero for n > 0; sympy writes a guard raised to a symbol that way)."""
    return _is_zero(e) or (isinstance(e, sy.Pow) and e.base.is_zero is True and not e.exp.is_number)


def is_guard(expr: sy.Basic) -> bool:
    """A Piecewise with a branch that is zero (or 0**n, conservatively)."""
    return isinstance(expr, Piecewise) and any(_is_zero_like(e) for e, _ in _pieces(expr))


def _foldable(expr: sy.Basic) -> bool:
    """A two-piece guard with a literal zero branch. Other guards are checked, not folded."""
    return isinstance(expr, Piecewise) and len(expr.args) == 2 and any(_is_zero(e) for e, _ in _pieces(expr))


def _product(a: Expr, b: Expr) -> Expr:
    """a*b, without sympy's distribution of a number over a sum (2*(x + y) would become 2*x + 2*y, another roundoff
    than the recipe's)."""
    if (a.is_Number and isinstance(b, sy.Add)) or (b.is_Number and isinstance(a, sy.Add)):
        return Mul(a, b, evaluate=False)
    return cast(Expr, a * b)


def fold_guards(expr: Expr) -> Expr:
    """
    Moves the other factors of every product with a two-piece guard factor into its non-zero branch, bottom up and
    to a fixpoint (a product with several guards folds completely). An expression without a foldable product is
    returned as the identical object.
    """
    if not isinstance(expr, sy.Basic) or not expr.args:
        return expr
    args = tuple(fold_guards(cast(Expr, a)) for a in expr.args)
    node = expr if all(a is b for a, b in zip(args, expr.args)) else expr.func(*args)
    if isinstance(node, Mul):
        factors = list(node.args)
        if (pos := next((n for n, f in enumerate(factors) if _foldable(f)), None)) is not None:
            guard = factors[pos]
            others = factors[:pos] + factors[pos + 1:]
            # Without sympy's distribution of a number over a sum (Rational(1, 3)*(a + b) -> a/3 + b/3).
            distributes = any(f.is_Number for f in others) and any(isinstance(f, sy.Add) for f in others)
            rest = Mul(*others, evaluate=False) if distributes else Mul(*others)
            pieces = tuple((e if _is_zero(e) else fold_guards(_product(cast(Expr, e), rest)), c)
                           for e, c in _pieces(guard))
            return Piecewise(*pieces)  # type: ignore[no-untyped-call]
    return node


def fold_guards_optimization() -> CseOptimization:
    """`fold_guards` as a CSE preprocessor (the `cse_optimizations` bake option); the module inserts it first."""
    return fold_guards, None


def _is_load(expr: sy.Basic, temps: Collection[Symbol]) -> bool:
    """A temporary that is only a read: a symbol that is not a temporary, or a stencil call."""
    return (isinstance(expr, Symbol) and expr not in temps) or (isinstance(expr, sy.Function) and
                                                                 getattr(expr.func, 'name', '') == 'stencil')


def check_guards(eqns: Mapping[Symbol, Expr], temps: set[Symbol], where: str,
                 external: Collection[str] = ()) -> tuple[set[str], set[str]]:
    """
    `eqns`: one loop's equations after CSE (temporaries included); `temps`: its temporaries; `external`: the names of
    values that earlier loops or functions set to a guard (tile temporaries, outputs), read here. Raises if a guard
    multiplies a non-constant factor. A guard is any expression that contains a guarded value (a Piecewise with a zero
    branch, or a value holding a guard) and is exactly zero once every guarded value is replaced by 0: a constant
    multiple, a sum or power of guards, Abs, Max, sin of one, a choice between guards. A power with a symbolic exponent
    (0**n) counts as a guard. A sum with any other term (A + P) is not a guard: it is A where P is false. Warns about
    arithmetic temporaries used only inside guarded branches. Returns the names (unprimed) of the values this loop sets
    to a guard and of those it sets to anything else, for the loops that read them.
    """
    guard_temps: set[Symbol] = set()

    def held(e: sy.Basic) -> bool:
        return isinstance(e, Symbol) and (e in guard_temps or str(e).replace("'", "") in external)

    def guard_like(e: sy.Basic) -> bool:
        if held(e) or is_guard(e):
            return True
        guarded = [x for x in sy.preorder_traversal(e) if is_guard(x) or held(x)]  # type: ignore[no-untyped-call]
        if not guarded:
            return False
        zero = e.xreplace({x: sy.Integer(0) for x in guarded})  # type: ignore[no-untyped-call]
        if zero.is_zero is True:
            return True
        # 0**n with n symbolic is 0 for n > 0: a guard, conservatively.
        return any(isinstance(x, sy.Pow) and x.base.is_zero is True and not x.exp.is_number
                   for x in sy.preorder_traversal(zero))  # type: ignore[no-untyped-call]

    # Every value first, in definition order, so that a guard held in a temporary or an output is known where it is
    # used.
    for lhs, rhs in eqns.items():
        if guard_like(rhs):
            guard_temps.add(lhs)
    for lhs, rhs in eqns.items():
        for node in sy.preorder_traversal(rhs):  # type: ignore[no-untyped-call]
            if isinstance(node, Mul):
                g = [f for f in node.args if guard_like(f)]
                others = [f for f in node.args if not guard_like(f)]
                if g and (len(g) > 1 or any(not f.is_number for f in others)):
                    raise GeneratorException(
                        f"{where}: {lhs} multiplies a guard (a Piecewise with a zero branch) by a non-constant factor; "
                        f"0 * Inf would be NaN. Write the guarded product directly, Piecewise((f*S, c), (0, True)), "
                        f"or Piecewise((f, c), (0, True))*S, which the backend folds (docs, section 7).")

    # Hoisted work: arithmetic temporaries read only inside guarded branches, directly or through other such
    # temporaries (transitively).
    uses: dict[Symbol, list[tuple[Symbol, bool]]] = {}  # symbol -> (the equation reading it, inside a guarded branch)

    def collect(e: sy.Basic, lhs: Symbol, guarded: bool) -> None:
        if isinstance(e, Symbol):
            uses.setdefault(e, []).append((lhs, guarded))
            return
        if is_guard(e):
            for pe, pc in _pieces(e):
                collect(pe, lhs, guarded or not _is_zero(pe))
                collect(pc, lhs, guarded)
            return
        for a in e.args:
            collect(a, lhs, guarded)

    for lhs, rhs in eqns.items():
        collect(rhs, lhs, False)
    arithmetic = {t for t in temps if t in eqns and not _is_load(eqns[t], temps)}
    only_guarded: set[Symbol] = set()
    changed = True
    while changed:
        changed = False
        for t in arithmetic - only_guarded:
            if t in uses and all(g or reader in only_guarded for reader, g in uses[t]):
                only_guarded.add(t)
                changed = True
    hoisted = sorted(only_guarded, key=str)
    if hoisted:
        wprint(f"{where}: {len(hoisted)} temporaries computed unconditionally but used only inside guarded branches "
               f"(work a false guard does not skip): {', '.join(map(str, hoisted[:10]))}"
               f"{', ...' if len(hoisted) > 10 else ''}")
    return ({str(lhs).replace("'", "") for lhs in guard_temps},
            {str(lhs).replace("'", "") for lhs in eqns if lhs not in guard_temps})
