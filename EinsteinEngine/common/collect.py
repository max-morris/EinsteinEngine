#  Copyright (C) 2026 Steven R. Brandt and other Einstein Engine contributors.
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

The search is deterministic: it is bounded by iteration counts rather
than by time, and candidates are generated in an order that does not
depend on hashing, so the same input gives the same output on every
machine and in every process.
"""

from typing import Callable, Iterator, List, Tuple

from sympy import Add, Expr, Mul, Pow, Symbol, cancel, collect, count_ops, expand, factor, factor_terms, together
from sympy.core.sorting import default_sort_key
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
    return int(count_ops(expr))  # type: ignore[no-untyped-call]


def _shared_symbols(node: Add) -> List[Symbol]:
    """Symbols occurring in two or more terms of an Add node, in sorted order."""
    counts: dict[Symbol, int] = {}
    for term in node.args:
        for sym in term.free_symbols:
            if isinstance(sym, Symbol):
                counts[sym] = counts.get(sym, 0) + 1
    return sorted((sym for sym, count in counts.items() if count >= 2), key=default_sort_key)


def _abstract_atoms(expr: Expr) -> Tuple[Expr, dict[Expr, Expr]]:
    """Replace Indexed/undefined-function atoms with plain Symbols and back.

    Lets Horner-style moves work on tensor expressions, and on grid
    variables, which CSE wraps in an undefined function to isolate them
    (see cse_isolate); collect() only factors out plain Symbols. The
    mapping is exact, so equality is preserved.

    The replacements are named in the sorted order of the atoms they
    replace. A bare Dummy() is named by a global counter, which would make
    the search depend on how many Dummies were made before it.
    """
    from sympy import Dummy, Indexed, Function
    from sympy.core.function import UndefinedFunction
    atoms: set[Expr] = set(expr.atoms(Indexed))
    atoms |= {atom for atom in expr.atoms(Function) if isinstance(atom.func, UndefinedFunction) and atom.args}
    if not atoms:
        return expr, {}
    reps: dict[Expr, Expr] = {atom: Dummy(f"collect_atom_{i:06d}")  # type: ignore[no-untyped-call]
                              for i, atom in enumerate(sorted(atoms, key=default_sort_key))}
    relocated = expr.xreplace(reps)  # type: ignore[no-untyped-call]
    return relocated, {v: k for k, v in reps.items()}


def collect_neighbors(expr: Expr, max_subnodes: int = _MAX_SUBNODES) -> Iterator[Tuple[str, Expr]]:
    """Yield (description, new_expr) single-move neighbors of expr."""
    seen = {expr}
    allow_slow = op_count(expr) <= _WHOLE_SLOW_MOVE_OP_LIMIT

    def emit(desc: str, new: Expr) -> Iterator[Tuple[str, Expr]]:
        if new not in seen:
            seen.add(new)
            yield desc, new

    whole: list[tuple[str, Callable[[], Expr]]] = [
        ("factor_terms", lambda: factor_terms(expr)),
        ("expand", lambda: expand(expr)),  # type: ignore[no-untyped-call]
    ]
    if allow_slow:
        whole += [("factor", lambda: factor(expr)),  # type: ignore[no-untyped-call]
                  ("together", lambda: together(expr))]  # type: ignore[no-untyped-call]
    for desc, thunk in whole:
        try:
            yield from emit(desc, thunk())
        except Exception:
            continue
    count = 0
    for node in preorder_traversal(expr):  # type: ignore[no-untyped-call]
        if count >= max_subnodes:
            break
        if isinstance(node, Add):
            jobs = ([(f"collect({sym})", lambda sym=sym: collect(node, sym)) for sym in _shared_symbols(node)]  # type: ignore[no-untyped-call]
                    + [("factor_terms", lambda: factor_terms(node))])
            if op_count(node) <= _WHOLE_SLOW_MOVE_OP_LIMIT:
                jobs.append(("factor", lambda: factor(node)))  # type: ignore[no-untyped-call]
        elif isinstance(node, Pow) and isinstance(node.args[0], Add):
            jobs = [("expand-pow", lambda: expand(node))]  # type: ignore[no-untyped-call]
        elif isinstance(node, Mul):
            jobs = [("factor_terms", lambda: factor_terms(node))]
        else:
            continue
        for desc, thunk in jobs:
            try:
                grown = expr.xreplace({node: thunk()})  # type: ignore[no-untyped-call]
            except Exception:
                continue
            yield from emit(f"{desc}@{type(node).__name__}", grown)
        count += 1


def collect_greedy(expr: Expr, cost: CostFn | None = None,
                   max_iter: int = 25, max_subnodes: int = _MAX_SUBNODES) -> Expr:
    """Steepest descent to a local cost minimum. Never worse than input.

    At most max_iter improving moves are made. On reaching the budget the
    current best (never worse than the input) is returned.
    """
    if cost is None:
        cost = op_count
    abstracted, restore = _abstract_atoms(expr)
    current = abstracted
    current_cost = cost(abstracted)
    for _ in range(max_iter):
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
    if current is abstracted:
        return expr
    result: Expr = current.xreplace(restore)  # type: ignore[no-untyped-call]
    # The search compared costs with each atom standing in as a Symbol. An atom can cost more than that as written (an
    # undefined function call counts as an operation), so a move that repeats an atom could cost more once restored.
    return result if cost(result) <= cost(expr) else expr


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
    assert cancel(improved - kernel) == 0  # type: ignore[no-untyped-call]

    # Factored square, never-worse battery, termination.
    assert collect_greedy(x**2 + 2 * x + 1) == (x + 1) ** 2
    for probe in [x + mk_symbol("y"), (x**2 - 1) / (x - 1),
                  x**4 + 4 * x**3 + 6 * x**2 + 4 * x + 1]:
        out = collect_greedy(probe)
        assert op_count(out) <= op_count(probe), probe
        assert cancel(out - probe) == 0, probe  # type: ignore[no-untyped-call]
    assert collect_greedy(x + mk_symbol("y"), max_iter=100) == x + mk_symbol("y")
    assert collect_greedy(x**2 + 2 * x + 1, max_iter=0) == x**2 + 2 * x + 1
    # Hang guard: large inputs skip slow whole-expression moves (sympy
    # heuristic GCD); cheap moves alone must still collapse this.
    import time as _time
    big = expand((x + mk_symbol("y") + mk_symbol("z") + mk_symbol("w")) ** 6)  # type: ignore[no-untyped-call]
    _t0 = _time.monotonic()
    _out = collect_greedy(big)
    assert _time.monotonic() - _t0 < 30.0, "slow-move gate failed"
    assert op_count(_out) <= op_count(big)
    print("collect.py self-test ok")
