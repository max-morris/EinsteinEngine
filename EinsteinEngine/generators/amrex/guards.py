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
                 external: Mapping[str, Expr] | None = None) -> dict[str, Expr]:
    """
    `eqns`: one loop's equations after CSE (temporaries included); `temps`: its temporaries; `external`: the zero forms
    (below) of the values that earlier loops or functions computed from a guard (outputs, tile temporaries), by name.

    A value's zero form is its expression with every guard (a Piecewise with a zero branch) replaced by 0 and every
    value read replaced by its own zero form. A guard is anything whose zero form is exactly 0, is still a guard, or is
    0**n with n symbolic (conservatively): a constant multiple, a sum, a power, Abs, Max or sin of guards, a choice
    between guards or between a guard and anything else (zero in some region), and values computed from such through
    temporaries, including cancellations ((exp(P) - 1)*S). A sum with any other term (A + P) is not a guard: it is A
    where P is false. Raises if a guard multiplies a non-constant factor (0 * Inf would be NaN). Warns about arithmetic
    temporaries used only inside guarded branches. Returns the zero forms of this loop's values that differ from the
    values themselves, by unprimed name, with the loop's own symbols replaced by placeholders.
    """
    ext = dict(external or {})
    zero_form: dict[Symbol, Expr] = {}

    def lookup(x: sy.Basic) -> Expr | None:
        if not isinstance(x, Symbol):
            return None
        if x in zero_form:
            return zero_form[x]
        return ext.get(str(x).replace("'", ""))

    def subst(e: sy.Basic) -> sy.Basic | None:
        """The zero form of e, or None if e involves no guard."""
        repl: dict[sy.Basic, sy.Basic] = {}
        for x in sy.preorder_traversal(e):  # type: ignore[no-untyped-call]
            if is_guard(x):
                repl[x] = sy.Integer(0)
            elif isinstance(x, Symbol) and (z := lookup(x)) is not None:
                repl[x] = z
        return e.xreplace(repl) if repl else None  # type: ignore[no-untyped-call]

    def guard_like(e: sy.Basic, depth: int = 0) -> bool:
        z = subst(e)
        if z is None:
            return False
        if z.is_zero is True:
            return True
        if any(isinstance(x, sy.Pow) and x.base.is_zero is True and not x.exp.is_number
               for x in sy.preorder_traversal(z)):  # type: ignore[no-untyped-call]
            return True  # 0**n with n symbolic: 0 for n > 0
        # Still a guard after the substitution (a choice with a zero branch now): zero in some region.
        return depth < 16 and z != e and guard_like(z, depth + 1)

    for lhs, rhs in eqns.items():
        for node in sy.preorder_traversal(rhs):  # type: ignore[no-untyped-call]
            if isinstance(node, Mul):
                g = [f for f in node.args if guard_like(f)]
                if g and (len(g) > 1 or any(not f.is_number for f in node.args if f not in g)):
                    raise GeneratorException(
                        f"{where}: {lhs} multiplies a guard (a value that is 0 where a Piecewise with a zero branch is "
                        f"false) by a non-constant factor; 0 * Inf would be NaN. Write the guarded product directly, "
                        f"Piecewise((f*S, c), (0, True)), or Piecewise((f, c), (0, True))*S, which the backend folds "
                        f"(docs, section 7).")
        if guard_like(rhs):
            zero_form[lhs] = sy.Integer(0)
        elif (z := subst(rhs)) is not None:
            zero_form[lhs] = cast(Expr, z)
    local = set(eqns)
    out: dict[str, Expr] = {}
    for lhs, z in zero_form.items():
        placeholders = {s: sy.Dummy(str(s)) for s in z.free_symbols if s in local}  # type: ignore[no-untyped-call]
        out[str(lhs).replace("'", "")] = cast(Expr, z.xreplace(placeholders))  # type: ignore[no-untyped-call]
    for lhs in eqns:  # recomputed without a guard: no longer a zero form
        out.setdefault(str(lhs).replace("'", ""), cast(Expr, lhs))

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
    return out
