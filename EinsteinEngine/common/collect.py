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

"""Greedy subset-factoring search (FORM-style partial Horner steps).

For each Add node, try collecting symbols shared by two or more terms
(e.g. ``x*a + y + x*b`` -> ``x*(a + b) + y``), alongside whole-expression
factor / factor_terms / together / expand moves. Steepest descent on a
cost function (default: operation count) until no move improves.

Only exact rewrites are used, so the result is mathematically equal to
the input by construction. Moves that raise on a given input are not
valid candidates and are skipped; the search itself never fails and
never returns a worse form than its input.
"""

from typing import Callable, Iterator, List, Tuple

from sympy import Add, Expr, Mul, Pow, Symbol, cancel, collect, count_ops, expand, factor, factor_terms, together
from sympy.core.traversal import preorder_traversal

__all__ = ["op_count", "collect_greedy", "collect_neighbors"]

CostFn = Callable[[Expr], int]

_MAX_SUBNODES = 40
# Whole-expression factor/together invoke sympy's heuristic polynomial GCD,
# which blows up on large inputs (observed hang). Gate them by size;
# factor_terms/collect/expand stay always-on (cheap and local).
_WHOLE_SLOW_MOVE_OP_LIMIT = 150


def op_count(expr: Expr) -> int:
    """Number of operations in expr (lower is simpler)."""
    return int(count_ops(expr))


def _shared_symbols(node: Add) -> List[Symbol]:
    """Symbols occurring in two or more terms of an Add node."""
    counts: dict[Symbol, int] = {}
    for term in node.args:
        for sym in term.free_symbols:
            if isinstance(sym, Symbol):
                counts[sym] = counts.get(sym, 0) + 1
    return [sym for sym, count in counts.items() if count >= 2]


def _abstract_atoms(expr: Expr) -> Tuple[Expr, dict[Expr, Expr]]:
    """Replace Indexed/undefined-function atoms with plain Symbols and back.

    Lets Horner-style moves work on tensor expressions; the mapping is
    exact, so equality is preserved.
    """
    from sympy import Dummy, Indexed, Function
    from sympy.core.function import UndefinedFunction
    reps: dict[Expr, Expr] = {}
    for atom in expr.atoms(Indexed):
        reps[atom] = Dummy()
    for atom in list(expr.atoms(Function)):
        if isinstance(atom.func, UndefinedFunction) and atom.args:
            reps.setdefault(atom, Dummy())
    if not reps:
        return expr, {}
    relocated = expr.xreplace(reps)
    return relocated, {v: k for k, v in reps.items()}


def collect_neighbors(expr: Expr, max_subnodes: int = _MAX_SUBNODES) -> Iterator[Tuple[str, Expr]]:
    """Yield (description, new_expr) single-move neighbors of expr."""
    seen = {hash(expr)}
    allow_slow = op_count(expr) <= _WHOLE_SLOW_MOVE_OP_LIMIT

    def emit(desc: str, new: Expr) -> Iterator[Tuple[str, Expr]]:
        if hash(new) not in seen:
            seen.add(hash(new))
            yield desc, new

    whole: list[tuple[str, Callable[[], Expr]]] = [
        ("factor_terms", lambda: factor_terms(expr)),
        ("expand", lambda: expand(expr)),
    ]
    if allow_slow:
        whole += [("factor", lambda: factor(expr)),
                  ("together", lambda: together(expr))]
    for desc, thunk in whole:
        try:
            yield from emit(desc, thunk())
        except Exception:
            continue
    count = 0
    for node in preorder_traversal(expr):
        if count >= max_subnodes:
            break
        if isinstance(node, Add):
            jobs = ([(f"collect({sym})", lambda sym=sym: collect(node, sym)) for sym in _shared_symbols(node)]
                    + [("factor_terms", lambda: factor_terms(node))])
            if op_count(node) <= _WHOLE_SLOW_MOVE_OP_LIMIT:
                jobs.append(("factor", lambda: factor(node)))
        elif isinstance(node, Pow) and isinstance(node.args[0], Add):
            jobs = [("expand-pow", lambda: expand(node))]
        elif isinstance(node, Mul):
            jobs = [("factor_terms", lambda: factor_terms(node))]
        else:
            continue
        for desc, thunk in jobs:
            try:
                grown = expr.xreplace({node: thunk()})
            except Exception:
                continue
            yield from emit(f"{desc}@{type(node).__name__}", grown)
        count += 1


def collect_greedy(expr: Expr, cost: CostFn | None = None,
                   max_iter: int = 25, max_subnodes: int = _MAX_SUBNODES,
                   max_seconds: float | None = None) -> Expr:
    """Steepest descent to a local cost minimum. Never worse than input.

    max_seconds bounds the search (checked per iteration); on expiry the
    current best — never worse than the input — is returned.
    """
    import time as _time
    if cost is None:
        cost = op_count
    current = expr
    current_cost = cost(expr)
    deadline = None if max_seconds is None else _time.monotonic() + max_seconds
    for _ in range(max_iter):
        if deadline is not None and _time.monotonic() >= deadline:
            break
        best, best_cost = current, current_cost
        for _, cand in collect_neighbors(current, max_subnodes=max_subnodes):
            try:
                cand_cost = cost(cand)
            except Exception:
                continue
            if cand_cost < best_cost:
                best, best_cost = cand, cand_cost
        if best_cost >= current_cost:
            break
        current, current_cost = best, best_cost
    return current


if __name__ == "__main__":
    from EinsteinEngine.common.sympywrap import mk_symbol

    x206 = mk_symbol("x206")
    x97 = mk_symbol("x97")
    x544 = mk_symbol("x544")
    x88 = mk_symbol("x88")
    x546 = mk_symbol("x546")
    x92 = mk_symbol("x92")
    x530 = mk_symbol("x530")
    x = mk_symbol("x")

    # The shared-factor pattern: x97 in two of three terms.
    kernel = x530 * (x206 * x97 + x544 * x88 + x546 * x97)
    assert op_count(kernel) == 6
    improved = collect_greedy(kernel)
    assert op_count(improved) == 5, improved
    assert cancel(improved - kernel) == 0

    # Factored square, never-worse battery, termination.
    assert collect_greedy(x**2 + 2 * x + 1) == (x + 1) ** 2
    for probe in [x + mk_symbol("y"), (x**2 - 1) / (x - 1),
                  x**4 + 4 * x**3 + 6 * x**2 + 4 * x + 1]:
        out = collect_greedy(probe)
        assert op_count(out) <= op_count(probe), probe
        assert cancel(out - probe) == 0, probe
    assert collect_greedy(x + mk_symbol("y"), max_iter=100) == x + mk_symbol("y")
    assert collect_greedy(x**2 + 2 * x + 1, max_iter=0) == x**2 + 2 * x + 1
    # Time budget: zero budget returns the input untouched (never worse).
    assert collect_greedy(x**2 + 2 * x + 1, max_seconds=0.0) == x**2 + 2 * x + 1
    # Hang guard: large inputs skip slow whole-expression moves (sympy
    # heuristic GCD); cheap moves alone must still collapse this.
    import time as _time
    big = expand((x + mk_symbol("y") + mk_symbol("z") + mk_symbol("w")) ** 6)
    _t0 = _time.monotonic()
    _out = collect_greedy(big)
    assert _time.monotonic() - _t0 < 30.0, "slow-move gate failed"
    assert op_count(_out) <= op_count(big)
    print("collect.py self-test ok")
