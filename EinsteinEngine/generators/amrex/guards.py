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


def is_guard(expr: sy.Basic) -> bool:
    """A Piecewise with a branch that is exactly zero."""
    return isinstance(expr, Piecewise) and any(e == 0 for e, _ in _pieces(expr))


def _foldable(expr: sy.Basic) -> bool:
    """A two-piece guard: one zero branch, one other. Guards with more pieces are checked, not folded."""
    return is_guard(expr) and len(expr.args) == 2


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
            rest = Mul(*(factors[:pos] + factors[pos + 1:]))
            pieces = tuple((e if e == 0 else fold_guards(cast(Expr, e) * rest), c) for e, c in _pieces(guard))
            return Piecewise(*pieces)  # type: ignore[no-untyped-call]
    return node


def fold_guards_optimization() -> CseOptimization:
    """`fold_guards` as a CSE preprocessor (the `cse_optimizations` bake option); the module inserts it first."""
    return fold_guards, None


def _is_load(expr: sy.Basic, temps: Collection[Symbol]) -> bool:
    """A temporary that is only a read: a symbol that is not a temporary, or a stencil call."""
    return (isinstance(expr, Symbol) and expr not in temps) or (isinstance(expr, sy.Function) and
                                                                 getattr(expr.func, 'name', '') == 'stencil')


def check_guards(eqns: Mapping[Symbol, Expr], temps: set[Symbol], where: str) -> None:
    """
    `eqns`: one loop's equations after CSE (temporaries included); `temps`: its temporaries. Raises if a guard (a
    Piecewise with a zero branch, a constant multiple of one, or a temporary holding either) multiplies a non-constant
    factor. Warns about arithmetic temporaries used only inside guarded branches.
    """
    guard_temps: set[Symbol] = set()

    def guard_like(e: sy.Basic) -> bool:
        if is_guard(e) or (isinstance(e, Symbol) and e in guard_temps):
            return True
        if isinstance(e, Mul):
            g = [f for f in e.args if guard_like(f)]
            return len(g) == 1 and all(f.is_number for f in e.args if f is not g[0])
        return False

    # Temporaries first, in definition order, so that a guard held in a temporary is known where it is used.
    for lhs, rhs in eqns.items():
        if lhs in temps and guard_like(rhs):
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

    # Hoisted work: arithmetic temporaries read inside some guarded branch and nowhere else.
    inside: set[Symbol] = set()
    outside: set[Symbol] = set()

    def collect(e: sy.Basic, guarded: bool) -> None:
        if isinstance(e, Symbol):
            (inside if guarded else outside).add(e)
            return
        if is_guard(e):
            for pe, pc in _pieces(e):
                collect(pe, guarded or pe != 0)
                collect(pc, guarded)
            return
        for a in e.args:
            collect(a, guarded)

    for rhs in eqns.values():
        collect(rhs, False)
    hoisted = sorted((t for t in inside - outside if t in temps and not _is_load(eqns[t], temps)), key=str)
    if hoisted:
        wprint(f"{where}: {len(hoisted)} temporaries computed unconditionally but used only inside guarded branches "
               f"(work a false guard does not skip): {', '.join(map(str, hoisted[:10]))}"
               f"{', ...' if len(hoisted) > 10 else ''}")
