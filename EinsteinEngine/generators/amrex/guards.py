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

import itertools
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


MAX_INLINE = 400  # tree nodes; a larger value stays a symbol (an opaque value; review round 4, A B7)
MAX_EXPANDED = 2000  # nodes of the expanded form, estimated; a larger zero form is not expanded (round 5, A B10)
MAX_CONDS = 8  # distinct conditions of a zero form decided by enumeration (2**k assignments; round 8)
MAX_BRANCHES = 256  # past MAX_CONDS: sympy's fold only under this nested count; else refused (round 5, B B3)
MAX_DEPTH = 16


def unique_nodes(e: sy.Basic) -> list[sy.Basic]:
    """Every distinct subexpression of e, once each (a DAG walk: linear in the shared size, not the tree size)."""
    seen: set[sy.Basic] = set()
    out: list[sy.Basic] = []
    stack = [e]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        stack.extend(x.args)
    return out


def _zero_power_free(z: sy.Basic) -> sy.Basic:
    """0**n with n symbolic is 0 for n > 0: replaced by 0 (conservatively) before testing for zero. On the DAG (once
    per distinct node), not the tree."""
    memo: dict[sy.Basic, sy.Basic] = {}

    def go(x: sy.Basic) -> sy.Basic:
        if x not in memo:
            args = [go(a) for a in x.args]
            y = x if all(a is b for a, b in zip(args, x.args)) else x.func(*args)
            # Tested after the rebuild: a base can become zero below (Abs(0**k)**k).
            memo[x] = sy.Integer(0) if isinstance(y, sy.Pow) and y.base.is_zero is True and not y.exp.is_number else y
        return memo[x]
    return go(z)


def expanded_size(e: sy.Basic, memo: dict[sy.Basic, tuple[float, float]] | None = None) -> float:
    """An estimate of the tree size of sy.expand(e), on the DAG (linear in the shared size): (terms, nodes) per
    subexpression, where a sum adds them, a product multiplies the terms, and a positive integer power raises them.
    Capped at 10 MAX_EXPANDED."""
    memo = {} if memo is None else memo
    cap = 10.0 * MAX_EXPANDED

    def size(x: sy.Basic) -> tuple[float, float]:
        if x in memo:
            return memo[x]
        parts = [size(a) for a in x.args]
        if isinstance(x, sy.Add):
            r = (min(cap, sum(p[0] for p in parts)), min(cap, sum(p[1] for p in parts)))
        elif isinstance(x, Mul):
            terms = 1.0
            for p in parts:
                terms = min(cap, terms * p[0])
            r = (terms, min(cap, terms * sum(p[1] / p[0] for p in parts)))
        elif isinstance(x, sy.Pow) and x.exp.is_Rational and abs(x.exp) >= 1:
            # sy.expand also expands negative and fractional powers of sums ((x + y)**-2 -> 1/(x**2 + 2*x*y + y**2),
            # (x + y)**(5/2) -> (x**2 + 2*x*y + y**2)*sqrt(x + y)): estimated as the power floor(|n|) (round 6, B B2).
            bt, bs = parts[0]
            n = int(abs(x.exp))
            terms = min(cap, bt ** min(n, 64))
            r = (terms, min(cap, terms * n * bs / bt))
        else:
            r = (1.0, min(cap, 1.0 + sum(p[1] for p in parts)))
        memo[x] = r
        return r
    return size(e)[1]


def _nested_branches(e: sy.Basic) -> int:
    """An upper bound on the products piecewise_fold builds for e: a choice adds its pieces' counts, any other node
    multiplies its arguments' (on the DAG), capped at 10**9."""
    memo: dict[sy.Basic, int] = {}

    def nested(x: sy.Basic) -> int:
        if x not in memo:
            if isinstance(x, Piecewise):
                r = sum(nested(b) for b, _ in _pieces(x))
            else:
                r = 1
                for a in x.args:
                    r = min(r * nested(a), 10 ** 9)
            memo[x] = min(r, 10 ** 9)
        return memo[x]
    return nested(e)


def _minmax_choices(z: sy.Basic) -> sy.Basic:
    """Every Max and Min as a choice between its arguments, on opaque conditions (bottom up, once per distinct node):
    linear, where rewrite(Piecewise) writes every pairwise comparison (round 5, B B3)."""
    memo: dict[sy.Basic, sy.Basic] = {}

    def go(x: sy.Basic) -> sy.Basic:
        if x not in memo:
            if not x.args:
                memo[x] = x
            else:
                args = [go(a) for a in x.args]
                if isinstance(x, (sy.Max, sy.Min)):
                    pieces = [(a, sy.Dummy('mm')) for a in args[:-1]]  # type: ignore[no-untyped-call]
                    memo[x] = Piecewise(*pieces, (args[-1], True))  # type: ignore[no-untyped-call]
                else:
                    memo[x] = x if all(a is b for a, b in zip(args, x.args)) else x.func(*args)
        return memo[x]
    return go(z)


def _fold(z: sy.Basic, where: str) -> sy.Basic:
    """z's choices folded into one choice between its distinct values, on opaque conditions (round 8, B B1, A B11).
    With k <= MAX_CONDS distinct conditions, by enumeration: under each of the 2**k truth assignments every choice
    collapses to its first branch whose condition holds (a DAG rebuild), and the distinct results are the values.
    The conditions are opaque: two that are logically related count as independent, so the values may include
    combinations that never occur (conservative). Past MAX_CONDS, sympy's piecewise_fold on opaque conditions, only
    when the nested count bounds its work at MAX_BRANCHES; otherwise refused. (piecewise_fold builds the cartesian
    product of a node's choices before it drops duplicates, and simplifies the conditions it combines, which can run
    trigsimp without end; no estimate of its work held up in review.)"""
    conds: list[sy.Basic] = []
    for x in unique_nodes(z):
        if isinstance(x, Piecewise):
            conds += [c for _, c in _pieces(x) if c is not sy.true and c is not sy.false and c not in conds]
    if len(conds) > MAX_CONDS:
        if _nested_branches(z) > MAX_BRANCHES:
            raise GeneratorException(f"{where}: more than {MAX_CONDS} conditions and {MAX_BRANCHES} combinations of "
                                     f"choices around a guard; the guard check cannot decide whether it multiplies a "
                                     f"value by 0 (docs, section 7).")
        dummies = {c: sy.Dummy('cond') for c in conds}  # type: ignore[no-untyped-call]
        folded = sy.piecewise_fold(z.xreplace(dummies))  # type: ignore[no-untyped-call]
        return cast(sy.Basic, folded.xreplace({d: c for c, d in dummies.items()}))
    values: list[sy.Basic] = []
    for bits in itertools.product((True, False), repeat=len(conds)):
        assign = dict(zip(conds, bits))
        memo: dict[sy.Basic, sy.Basic] = {}

        def go(x: sy.Basic) -> sy.Basic:
            if x not in memo:
                if isinstance(x, Piecewise):
                    # The first branch whose condition holds; an incomplete choice with none is NaN (no value).
                    memo[x] = next((go(e) for e, c in _pieces(x) if c is sy.true or assign.get(c) is True), sy.nan)
                elif not x.args:
                    memo[x] = x
                else:
                    args = [go(a) for a in x.args]
                    memo[x] = x if all(a is b for a, b in zip(args, x.args)) else x.func(*args)
            return memo[x]
        v = go(z)
        if v not in values:
            values.append(v)
    if len(values) == 1:
        return values[0]
    return Piecewise(*[(v, sy.Dummy('cond')) for v in values[:-1]], (values[-1], True))  # type: ignore[no-untyped-call]


def tree_size(e: sy.Basic, memo: dict[sy.Basic, int] | None = None) -> int:
    """The size of e as a tree (what xreplace and printing walk), computed on the DAG."""
    memo = {} if memo is None else memo
    if e not in memo:
        memo[e] = 1 + sum(tree_size(a, memo) for a in e.args)
    return memo[e]


def check_guards(eqns: Mapping[Symbol, Expr], temps: set[Symbol], where: str,
                 external: Mapping[str, Expr] | None = None, carried: Collection[str] = (),
                 old: Mapping[str, sy.Basic] | None = None) -> dict[str, Expr]:
    """
    `eqns`: one loop's equations after CSE (temporaries included), in order; `temps`: its temporaries; `external`: the
    value forms (below) of values earlier loops or functions computed (outputs, tile temporaries), by name; `carried`:
    the names (unprimed) of this loop's values that later loops read by name; `old`: for each name this loop assigns
    (unprimed), the placeholder that stands for its value before the loop (analysis.py), shared by every form.

    A value's value form is its expression with every value it reads replaced by that value's own value form
    (inlined), as long as the result stays small (MAX_INLINE tree nodes; a larger value stays a symbol, and a larger
    guard stays a guard: Piecewise((name, c), (0, True)) with a fresh condition c). Its zero
    form is its value form with every guard (a Piecewise with a zero branch) replaced by 0. A guard is anything whose
    zero form is exactly 0 (after 0**n -> 0, Max/Min written as choices, piecewise_fold, and expansion while the
    expanded form stays under MAX_EXPANDED nodes), or is still a
    guard (recursively: a choice with a guarded branch is zero in some region). So constant multiples, sums, powers,
    Abs, Max, Min, sin of guards, choices with a guarded branch, and cancellations through temporaries, copies and
    constants ((exp(P) - 1)*S, (t + P - t1)*S with t1 = t) are guards. A sum with any other term (A + P) is not: it is A
    where P is false, and a zero of A itself (data) is outside R10 (docs, section 7). Raises if a guard multiplies a
    non-constant factor (0 * Inf would be NaN), and at nesting depth MAX_DEPTH (refused, not passed). Warns about
    arithmetic temporaries used only inside guarded branches.

    Returns, by unprimed name, the value form of every value of this loop, ready to carry: symbols of this loop that
    stayed symbols are replaced by one placeholder each (shared within the loop, so that identities survive a split),
    except carried names, which stay names; a name the loop assigns, read for its value before the loop, becomes
    `old[name]`. A value that stayed a symbol is returned as its own name.
    """
    ext = dict(external or {})
    value: dict[sy.Basic, sy.Basic] = {}
    memo: dict[sy.Basic, int] = {}
    expanded: dict[sy.Basic, tuple[float, float]] = {}

    def inline(e: sy.Basic) -> sy.Basic:
        repl: dict[sy.Basic, sy.Basic] = {}
        for x in e.free_symbols:
            if x in value:
                if value[x] != x:
                    repl[x] = value[x]
            elif (v := ext.get(str(x).replace("'", ""))) is not None and v != x:
                repl[x] = v
        return e.xreplace(repl) if repl else e  # type: ignore[no-untyped-call]

    def subst(e: sy.Basic) -> sy.Basic | None:
        """The zero form of e (piecewise-folded), or None if e involves no guard."""
        v = inline(e)
        guards = [x for x in unique_nodes(v) if is_guard(x)]
        if not guards:
            return None
        # 0**n -> 0 before Max/Min are rewritten and the fold, and again after the fold, which can move a 0 under a
        # symbolic power (round 5, B B2).
        z: sy.Basic = _zero_power_free(v.xreplace({g: sy.Integer(0) for g in guards}))  # type: ignore[no-untyped-call]
        # Max and Min of a zero are choices too (Max(0, b) is 0 where b <= 0); a choice whose branches became equal or
        # zero (Piecewise((1, c), (b, True)) - 1) is a guard again once folded.
        if any(isinstance(x, (sy.Max, sy.Min)) for x in unique_nodes(z)):
            z = _minmax_choices(z)
        if any(isinstance(x, Piecewise) for x in unique_nodes(z)):
            z = _zero_power_free(_fold(z, where))
        return z

    def is_zero(z: sy.Basic) -> bool:
        """z is a zero form from subst (folded, without 0**n)."""
        if z.is_zero is True:
            return True
        # Cancellation after expansion (u*(w + 1) - u*w - u), for zero forms small enough to expand.
        return (tree_size(z, memo) <= MAX_INLINE and expanded_size(z, expanded) <= MAX_EXPANDED and
                sy.expand(z).is_zero is True)  # type: ignore[no-untyped-call]

    def guard_like(e: sy.Basic, depth: int = 0) -> bool:
        z = subst(e)
        if z is None:
            return False
        if is_zero(z):
            return True
        if any(is_guard(x) for x in unique_nodes(z)):
            if depth >= MAX_DEPTH:
                raise GeneratorException(f"{where}: guards nested more than {MAX_DEPTH} deep; the guard check cannot "
                                         f"decide whether they multiply a value by 0 (docs, section 7).")
            return guard_like(z, depth + 1)
        return False

    for lhs, rhs in eqns.items():
        for node in unique_nodes(rhs):
            if isinstance(node, Mul):
                g = [f for f in node.args if guard_like(f)]
                if g and (len(g) > 1 or any(not f.is_number for f in node.args if f not in g)):
                    raise GeneratorException(
                        f"{where}: {lhs} multiplies a guard (a value that is 0 where a Piecewise with a zero branch is "
                        f"false) by a non-constant factor; 0 * Inf would be NaN. Write the guarded product directly, "
                        f"Piecewise((f*S, c), (0, True)), or Piecewise((f, c), (0, True))*S, which the backend folds "
                        f"(docs, section 7).")
        v = inline(rhs)
        if tree_size(v, memo) <= MAX_INLINE:
            value[lhs] = v
        elif guard_like(rhs):  # opaque, but still a guard (round 5, A B8)
            value[lhs] = Piecewise((lhs, sy.Dummy(f'{lhs}_c') > 0), (0, True))  # type: ignore[no-untyped-call]
        else:
            value[lhs] = lhs

    # Carry: the loop's symbols that stayed symbols become one placeholder each, carried names stay names, and the old
    # value of a name the loop assigns becomes its shared placeholder (round 5, A B9).
    keep = {str(c) for c in carried}
    placeholder: dict[sy.Basic, sy.Basic] = {
        x: (Symbol(str(x).replace("'", "")) if str(x).replace("'", "") in keep  # type: ignore[no-untyped-call]
            else sy.Dummy(str(x)))  # type: ignore[no-untyped-call]
        for x in eqns}
    for x in {x for v in value.values() for x in v.free_symbols}:
        if x not in placeholder and not isinstance(x, sy.Dummy) and str(x) in (old or {}):
            placeholder[x] = (old or {})[str(x)]
    out: dict[str, Expr] = {}
    for key, v in value.items():
        name = str(key).replace("'", "")
        if v == key:
            out[name] = cast(Expr, Symbol(name))  # type: ignore[no-untyped-call]
        else:
            out[name] = cast(Expr, v.xreplace(  # type: ignore[no-untyped-call]
                {x: placeholder[x] for x in v.free_symbols if x in placeholder}))

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
