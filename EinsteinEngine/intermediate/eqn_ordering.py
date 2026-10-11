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

from __future__ import annotations

import re
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from functools import cache
from itertools import chain
from math import sqrt
from typing import Iterator, Callable, TYPE_CHECKING, NamedTuple, cast, Optional, Any, Sequence

from bayes_opt import BayesianOptimization
from sympy import Symbol, Expr, Basic, preorder_traversal

from EinsteinEngine.intermediate.dependencies import Dependencies
from EinsteinEngine.frontend.definitions import stencil
from EinsteinEngine.common.util import pprint
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.common.describe_param import describe_param

if TYPE_CHECKING:
    from EinsteinEngine.intermediate.eqnlist import EqnList

EqnOrderingFn = Callable[[dict[Symbol, Expr], 'EqnList'], Iterator[Symbol | tuple[Symbol, str]]]

_NON_C_IDENTIFIER_RE = re.compile(r'[^A-Za-z0-9_]')

@dataclass(frozen=True)
class _LifetimesData:
    lifetimes: dict[Symbol, tuple[int, int]]
    births: dict[int, set[Symbol]]
    deaths: dict[int, set[Symbol]]

def _get_lifetimes(eqns: dict[Symbol, Expr], order: list[Symbol]) -> _LifetimesData:
    if len(eqns) == 0 or len(order) == 0:
        return _LifetimesData(dict(), dict(), dict())

    assert len(eqns) == len(order), f"eqns and order must have the same length, but got {len(eqns)} and {len(order)}"
    assert set(eqns.keys()) == set(order), "order must contain exactly the equation symbols"

    first_occurrence: dict[Symbol, int] = dict()
    last_occurrence: dict[Symbol, int] = dict()

    for idx, lhs in enumerate(order):
        if lhs not in first_occurrence:
            first_occurrence[lhs] = idx

        for sym in _free_symbols_with_dummies(eqns[lhs]):
            if sym not in first_occurrence:
                first_occurrence[sym] = idx
            last_occurrence[sym] = idx

    lifetimes = {
        sym: (first_occurrence[sym], last_occurrence[sym] if sym in last_occurrence else first_occurrence[sym])
        for sym in first_occurrence
    }
    births: dict[int, set[Symbol]] = defaultdict(set)
    deaths: dict[int, set[Symbol]] = defaultdict(set)

    for sym, (birth, death) in lifetimes.items():
        births[birth].add(sym)
        deaths[death].add(sym)

    return _LifetimesData(lifetimes, births, deaths)

def _score_memory_pressure(lifetimes: dict[Symbol, tuple[int, int]]) -> dict[Symbol, int]:
    return {sym: 1 + lifetime[1] - lifetime[0] for (sym, lifetime) in lifetimes.items()}

def score_memory_pressure(eqns: dict[Symbol, Expr], order: list[Symbol]) -> dict[Symbol, int]:
    return _score_memory_pressure(_get_lifetimes(eqns, order).lifetimes)

def _score_memory_pressure_fast(lifetimes: dict[Symbol, tuple[int, int]]) -> int:
    total = 0
    for (_, (birth, death)) in lifetimes.items():
        total += death - birth + 1
    return total

def _score_peak_liveness(lifetimes: dict[Symbol, tuple[int, int]], n_eqns: int) -> int:
    births: dict[int, int] = defaultdict(int)
    deaths: dict[int, int] = defaultdict(int)

    for (birth, death) in lifetimes.values():
        births[birth] += 1
        deaths[death] += 1

    peak = 0
    liveness = 0

    for idx in range(n_eqns):
        liveness += births[idx] - deaths[idx - 1]
        peak = max(peak, liveness)

    return peak

def _score_all_liveness(lifetimes: _LifetimesData, order: OrderedDict[Symbol, Expr]) -> dict[Symbol, set[Symbol]]:
    liveness: set[Symbol] = set()
    liveness_dict: dict[Symbol, set[Symbol]] = dict()

    for idx, (lhs, rhs) in enumerate(order.items()):
        if births := lifetimes.births.get(idx):
            liveness.update(births)
        if deaths := lifetimes.deaths.get(idx - 1):
            liveness.difference_update(deaths)

        liveness_dict[lhs] = liveness.copy()

    return liveness_dict

def score_peak_liveness(eqns: dict[Symbol, Expr], order: list[Symbol]) -> int:
    return _score_peak_liveness(_get_lifetimes(eqns, order).lifetimes, len(order))

def _score_symbol_reuse(lifetimes_data: _LifetimesData, eqns: dict[Symbol, Expr], order: OrderedDict[Symbol, Expr]) -> int:
    if len(eqns) == 0:
        return 0

    births, deaths = lifetimes_data.births, lifetimes_data.deaths

    in_memory: set[Symbol] = set()
    score = 0

    for idx, (_, rhs) in enumerate(order.items()):
        in_memory.update(births[idx])
        if idx > 0:
            in_memory.difference_update(deaths[idx - 1])
        score += len(in_memory.intersection(_free_symbols_with_dummies(rhs)))

    return score


def maximize_symbol_reuse(eqns: dict[Symbol, Expr], eqn_list: EqnList) -> Iterator[Symbol]:
    """
    Orders equations based on symbol reuse, prioritizing equations that use symbols already present in previous equations.
    Equations with higher complexity are given higher priority. The first equation is always the most complex.
    """

    if len(eqns) == 0:
        return

    eqns_remaining = eqns.copy()
    in_memory: set[Symbol] = set()

    disambiguation = sorted(eqns_remaining.keys(), key=str, reverse=True)

    lhs, rhs = max(eqns_remaining.items(), key=lambda kv: (eqn_list.complexity[kv[0]], disambiguation.index(kv[0])))
    del eqns_remaining[lhs]
    in_memory.update(_free_symbols_with_dummies(rhs))
    yield lhs

    while len(eqns_remaining) > 0:
        lhs, rhs = max(eqns_remaining.items(),
                       key=lambda kv: (len(_free_symbols_with_dummies(kv[1]).intersection(in_memory)), eqn_list.complexity[kv[0]],
                                       disambiguation.index(kv[0])))
        del eqns_remaining[lhs]
        in_memory.update(_free_symbols_with_dummies(rhs))
        yield lhs

class _EqnScoreByRarity(NamedTuple):
    linear: Callable[[Symbol], float]
    sqrt: Callable[[Symbol], float]

    @classmethod
    def get_unit(cls) -> '_EqnScoreByRarity':
        return cls(lambda _: 0.0, lambda _: 0.0)

def _get_eqn_score_fn_by_rarity(eqns: dict[Symbol, Expr], consider_frequency: bool = True) -> _EqnScoreByRarity:
    if len(eqns) == 0:
        return _EqnScoreByRarity.get_unit()

    reciprocal_rarity: dict[Symbol, float] = defaultdict(int)
    free_symbols_by_eqn: dict[Symbol, set[Symbol]] = dict()
    frequency_by_eqn: dict[Symbol, dict[Symbol, int]] = dict()  # {lhs: {sym: freq}}

    for lhs, rhs in eqns.items():
        free_symbols = _free_symbols_with_dummies(rhs)
        free_symbols_by_eqn[lhs] = free_symbols
        for sym in free_symbols:
            reciprocal_rarity[sym] += 1

        if consider_frequency:
            symbol_frequency = _symbol_frequency(rhs)
            frequency_by_eqn[lhs] = {sym: symbol_frequency[sym] for sym in free_symbols}

    symbol_rarity: dict[Symbol, float] = {sym: (1.0 / reciprocal) for sym, reciprocal in reciprocal_rarity.items()}
    symbol_rarity_sqrt: dict[Symbol, float] = {sym: 1.0 / sqrt(reciprocal) for sym, reciprocal in reciprocal_rarity.items()}

    if consider_frequency:
        scores = {
            lhs: sum(frequency_by_eqn[lhs][sym] * symbol_rarity[sym] for sym in free_symbols_by_eqn[lhs])
            for lhs in eqns.keys()
        }

        sqrt_scores = {
            lhs: sum(frequency_by_eqn[lhs][sym] * symbol_rarity_sqrt[sym] for sym in free_symbols_by_eqn[lhs])
            for lhs in eqns.keys()
        }
    else:
        scores = {
            lhs: sum(symbol_rarity[sym] for sym in free_symbols_by_eqn[lhs])
            for lhs in eqns.keys()
        }

        sqrt_scores = {
            lhs: sum(symbol_rarity_sqrt[sym] for sym in free_symbols_by_eqn[lhs])
            for lhs in eqns.keys()
        }

    return _EqnScoreByRarity(lambda lhs: scores[lhs], lambda lhs: sqrt_scores[lhs])


def _get_eqn_score_fn_by_complexity(eqn_list: EqnList) -> Callable[[Symbol], float]:
    return lambda lhs: eqn_list.complexity[lhs]


def _score_eqn_by_symbol_reuse(eqn_rhs: Expr, previous_rhses: set[Expr]) -> int:
    if len(previous_rhses) == 0:
        return 0

    return len(_free_symbols_with_dummies(eqn_rhs).intersection(chain(*(_free_symbols_with_dummies(rhs) for rhs in previous_rhses))))


def _get_reused_symbol_distances(eqn_rhs: Expr, in_memory: dict[Symbol, int], my_pos: int) -> dict[Symbol, int]:
    if len(in_memory) == 0:
        return dict()

    return {sym: my_pos - in_memory[sym] for sym in _free_symbols_with_dummies(eqn_rhs).intersection(in_memory.keys())}


class _SymbolReuseStats(NamedTuple):
    n_reused: int
    peak_distance: int
    avg_distance: float


def _get_symbol_reuse_stats(free_symbols: set[Symbol], in_memory: dict[Symbol, int], my_pos: int) -> _SymbolReuseStats:
    if len(in_memory) == 0:
        return _SymbolReuseStats(0, 0, 0.0)

    count = 0
    peak = 0
    total = 0

    for sym in free_symbols:
        if (last_pos := in_memory.get(sym)) is not None:
            count += 1
            distance = my_pos - last_pos
            total += distance
            if distance > peak:
                peak = distance

    return _SymbolReuseStats(count, peak, (total / count if count > 0 else 0.0))


def cartesian_product(*ordering_fns: EqnOrderingFn) -> EqnOrderingFn:
    def fn(eqns: dict[Symbol, Expr], eqn_list: EqnList) -> Iterator[tuple[Symbol, str]]:
        order: OrderedDict[Symbol, str] = OrderedDict()

        for ordering_fn in ordering_fns:
            for item in ordering_fn(eqns, eqn_list):
                if isinstance(item, tuple):
                    sym, msg = item
                else:
                    sym, msg = item, ''
                if sym not in order:
                    order[sym] = msg

        yield from order.items()

    return fn


def prioritize_rare_symbols(eqns: dict[Symbol, Expr],
                            eqn_list: EqnList,
                            consider_frequency: bool = True,
                            complexity_factor: float = 0.0) -> Iterator[tuple[Symbol, str]]:
    """
    Orders equations based on symbol rarity.
    Equations which use symbols that are less common in other equations are given higher priority.

    To determine the rarity of a symbol, multiple occurrences of the same symbol in an equation are treated as one.
    If `consider_frequency` is true, when evaluating the overall priority of an equation, the rarity of each symbol is weighted positively by the frequency of that symbol in the equation.

    The complexity score of an equation, scaled by `complexity_factor`, is added to the priority.
    """

    if len(eqns) == 0:
        return

    raw_eqn_score = _get_eqn_score_fn_by_rarity(eqns, consider_frequency).linear
    def eqn_score(lhs: Symbol) -> float: return raw_eqn_score(lhs) + (complexity_factor * eqn_list.complexity[lhs])

    scores = {lhs: eqn_score(lhs) for lhs in eqns.keys()}

    disambiguation_rank: dict[Symbol, int] = {
        lhs: idx for idx, lhs in enumerate(sorted(eqns.keys(), key=str, reverse=True))
    }

    ordered = sorted(eqns.keys(), key=lambda lhs: (scores[lhs], eqn_list.complexity[lhs], disambiguation_rank[lhs]), reverse=True)
    ordered_dict: OrderedDict[Symbol, Expr] = OrderedDict()
    for lhs in ordered:
        ordered_dict[lhs] = eqns[lhs]

    ordered_liveness = _score_all_liveness(_get_lifetimes(eqns, ordered), ordered_dict)

    yield from (
        (lhs, f'Liveness = {len(ordered_liveness[lhs])}; {sorted(ordered_liveness[lhs], key=str)}')
        for lhs in ordered.__iter__()
    )
    
def lexicographical_order(eqns: dict[Symbol, Expr], _eqn_list: EqnList) -> Iterator[Symbol]:
    """
    Orders equations lexicographically based on their LHS names.
    """

    yield from sorted(eqns.keys(), key=str)

def _order_by_keys(order: dict[Symbol, int], eqns: dict[Symbol, Expr], eqn_list: EqnList, exclude_synthetic_symbols: bool, what: str) -> Iterator[Symbol]:
    """The equations in the key order of `order`, optionally without the list's synthetic symbols."""

    assert (keyset := set(eqns.keys())).intersection(order.keys()) == keyset, f"EqnList {what} order dict is missing keys"

    for lhs in order.keys():
        if lhs in eqns and not (exclude_synthetic_symbols and lhs in eqn_list.synthetic_symbols):
            yield lhs


def recipe_order(eqns: dict[Symbol, Expr], eqn_list: EqnList, exclude_synthetic_symbols: bool = False) -> Iterator[Symbol]:
    """
    Orders equations based on their recipe order, i.e., the order in which they were added to the function.
    Temporaries (pull-out, CSE, etc.) come after the recipe equations, in the order they were created.
    The recipe order is never rewritten; see `pre_population_order` for the order the ordering loci act on.

    If `exclude_synthetic_symbols` is true, synthetic symbols are not yielded; dependency repair places them.
    """

    yield from _order_by_keys(eqn_list.eqn_recipe_order, eqns, eqn_list, exclude_synthetic_symbols, "recipe")


def pre_population_order(eqns: dict[Symbol, Expr], eqn_list: EqnList, exclude_synthetic_symbols: bool = False) -> Iterator[Symbol]:
    """
    Orders equations based on their pre-population order. This starts equal to the recipe order and is rewritten by
    the ``early_ordering_fn`` and ``pre_population_ordering_fn`` bake options; without them, it is the recipe order.
    Temporaries come after, in the order they were created, and equations a soft-split merge moves into a list are
    appended in the pre-population order of the list they come from.

    If `exclude_synthetic_symbols` is true, synthetic symbols are not yielded; dependency repair places them.
    """

    yield from _order_by_keys(eqn_list.eqn_pre_population_order, eqns, eqn_list, exclude_synthetic_symbols, "pre-population")


class _AddEqnKeyOrder:
    """The ordering function returned by `add_eqn_key_order`. Its repr names the key, so it is informative and stable
    across runs (tuners record it to detect a changed configuration)."""

    def __init__(self, key: Callable[[int], float]) -> None:
        self.key = key

    def __call__(self, eqns: dict[Symbol, Expr], eqn_list: EqnList) -> Iterator[Symbol]:
        origins = eqn_list.eqn_origin
        recipe_positions = eqn_list.eqn_recipe_order
        with_origin = sorted(
            (lhs for lhs in eqns.keys() if lhs in origins),
            key=lambda lhs: (self.key(origins[lhs]), recipe_positions[lhs])
        )
        yield from with_origin
        yield from (lhs for lhs in pre_population_order(eqns, eqn_list) if lhs not in origins)

    def __repr__(self) -> str:
        return f'add_eqn_key_order({describe_param(self.key)})'


class _AddEqnOrder(_AddEqnKeyOrder):
    """The ordering function returned by `add_eqn_order`; its repr is ``add_eqn_order([...])``."""

    def __init__(self, order: Sequence[int]) -> None:
        self.order = tuple(order)
        for idx in self.order:
            if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
                raise DslException(f"add_eqn_order: {idx!r} is not a valid 0-based add_eqn call index.")
        if len(duplicates := sorted({idx for idx in self.order if self.order.count(idx) > 1})) > 0:
            raise DslException(f"add_eqn_order: the call indices {duplicates} are listed more than once.")

        rank = {idx: position for position, idx in enumerate(self.order)}
        super().__init__(lambda origin: rank.get(origin, len(rank) + origin))

    def __call__(self, eqns: dict[Symbol, Expr], eqn_list: EqnList) -> Iterator[Symbol]:
        if len(self.order) > 0 and (max_idx := max(self.order)) >= (n_calls := eqn_list.parent.origin_count):
            raise DslException(f"add_eqn_order: call index {max_idx} is out of range; the function has {n_calls} add_eqn calls (valid indices are 0..{n_calls - 1}).")
        yield from super().__call__(eqns, eqn_list)

    def __repr__(self) -> str:
        return f'add_eqn_order({list(self.order)!r})'


def add_eqn_key_order(key: Callable[[int], float]) -> EqnOrderingFn:
    """
    Orders equations by `key` applied to the 0-based index of the add_eqn call that created them (their origin);
    ties go to recipe order. Equations without an origin (e.g., CSE temps) follow, in pre-population order;
    dependency repair pulls them earlier as needed.

    Used as the ``early_ordering_fn``, this orders the author-level add_eqn calls themselves.
    The result's repr is ``add_eqn_key_order(<key>)``, naming a function key by its qualified name.
    """

    return _AddEqnKeyOrder(key)


def add_eqn_order(order: Sequence[int]) -> EqnOrderingFn:
    """
    Orders equations by an explicit order of add_eqn calls, given as 0-based call indices: the equations of call
    ``order[0]`` come first, then those of ``order[1]``, and so on. Calls not listed keep their recipe order after the
    listed ones. Equations without an origin follow, as in `add_eqn_key_order`.

    The indices refer to one function's add_eqn calls, so pass this per function (``functions={name: {...}}``) when a
    thorn has several functions. The result's repr is ``add_eqn_order([...])``.
    """

    return _AddEqnOrder(order)


def fixed_order(order: Sequence[Symbol]) -> EqnOrderingFn:
    """
    Yields the equations in exactly the given order (equations not listed follow in dict order). It is not marked as
    respecting dependency order, so order_builder still repairs dependencies. Used to rebake refined EqnLists so that
    the order a cut was made in is the order the pieces keep.
    """

    fixed = tuple(order)

    def fn(eqns: dict[Symbol, Expr], _eqn_list: EqnList) -> Iterator[Symbol]:
        seen: set[Symbol] = set()
        for lhs in fixed:
            if lhs in eqns and lhs not in seen:
                seen.add(lhs)
                yield lhs
        yield from (lhs for lhs in eqns.keys() if lhs not in seen)

    return fn

@cache
def _dummy_stencil_symbol(call: Basic) -> Symbol:
    return Symbol(_NON_C_IDENTIFIER_RE.sub('_', f'__dummy_stencil_{call.args}'.replace('-', 'm')))  # type: ignore[no-untyped-call]

@cache
def _expr_with_stencil_dummies(expr: Expr) -> Expr:
    stencil_calls: set[Basic] = expr.find(stencil)  # type: ignore[no-untyped-call]
    if len(stencil_calls) == 0:
        return expr
    else:
        return expr.xreplace({call: _dummy_stencil_symbol(call) for call in stencil_calls})  # type: ignore[no-any-return,no-untyped-call]

@cache
def _symbol_frequency(expr: Expr) -> dict[Symbol, int]:
    freq: dict[Symbol, int] = defaultdict(int)
    for node in preorder_traversal(_expr_with_stencil_dummies(expr)):  # type: ignore[no-untyped-call]
        if isinstance(node, Symbol):
            freq[node] += 1
    return freq

@cache
def _free_symbols(expr: Expr) -> set[Symbol]:
    return cast(set[Symbol], expr.free_symbols)

@cache
def _free_symbols_with_dummies(expr: Expr) -> set[Symbol]:
    # When calculating liveness and symbol reuse, we want to consider unique stencil calls as their own symbols because
    # they are distinct quantities.
    # If we don't do this, e.g., `stencil(w, 1, 0, 0)` and `w` will be considered the same symbol.
    return _free_symbols(_expr_with_stencil_dummies(expr))

def bayesian_optimization(eqns: dict[Symbol, Expr],
                          eqn_list: EqnList,
                          exploration_iter: int = 5,
                          optimization_iter: int = 10,
                          memory_pressure_factor: float = 0.0,
                          peak_liveness_factor: float = -1.0,
                          symbol_reuse_factor: float = 0.0) -> Iterator[tuple[Symbol, str]]:
    linear_rarity_score, sqrt_rarity_score = _get_eqn_score_fn_by_rarity(eqns, consider_frequency=True)
    free_symbols_by_lhs = {lhs: _free_symbols_with_dummies(rhs) for lhs, rhs in eqns.items()}
    dependencies = Dependencies(eqns, free_symbols=_free_symbols)
    dummy_dependencies = Dependencies(eqns, free_symbols=_free_symbols_with_dummies)

    disambiguation_rank: dict[Symbol, int] = {
        lhs: idx for idx, lhs in enumerate(sorted(eqns.keys(), key=str, reverse=True))
    }

    order_cache: dict[tuple[tuple[str, float], ...], OrderedDict[Symbol, Expr]] = dict()
    def put_cache(val: OrderedDict[Symbol, Expr], **kwargs: float) -> None:
        key = tuple(sorted(kwargs.items()))
        order_cache[key] = val

    def get_cache(**kwargs: float) -> OrderedDict[Symbol, Expr]:
        key = tuple(sorted(kwargs.items()))
        return order_cache[key]

    def black_box_order(complexity_weight: float,
                        rarity_weight: float,
                        sqrt_rarity_weight: float,
                        peak_symbol_distance_weight: float,
                        avg_symbol_distance_weight: float,
                        symbol_reuse_weight: float,
                        in_memory_override: Optional[dict[Symbol, int]] = None,
                        eqns_remaining_override: Optional[dict[Symbol, Expr]] = None) -> OrderedDict[Symbol, Expr]:
        eqns_remaining = eqns_remaining_override.copy() if eqns_remaining_override is not None else eqns.copy()

        order: OrderedDict[Symbol, Expr] = OrderedDict()
        in_memory: dict[Symbol, int] = in_memory_override or dict()  # symbol -> index in order

        idx = len(in_memory)

        while len(eqns_remaining) > 0:
            required_deps: dict[Symbol, set[Symbol]] = dict()

            def score(lhs: Symbol) -> float:
                dependency_symbols: set[Symbol] = set()
                dummy_dependency_symbols: set[Symbol] = set()

                if len(deps := dependencies.get_transitive_dependencies(lhs)) > 0:
                    dummy_deps = dummy_dependencies.get_transitive_dependencies(lhs)

                    for dep in deps:
                        if dep not in in_memory:
                            dependency_symbols.add(dep)

                    for dep in dummy_deps:
                        if dep not in in_memory:
                            dummy_dependency_symbols.add(dep)

                required_deps[lhs] = dependency_symbols

                symbol_reuse_count, peak_symbol_distance, avg_symbol_distance = _get_symbol_reuse_stats(
                    free_symbols_by_lhs[lhs].union(dummy_dependency_symbols), in_memory, idx
                )

                return (
                    complexity_weight * eqn_list.complexity[lhs]
                    + rarity_weight * linear_rarity_score(lhs)
                    + sqrt_rarity_weight * sqrt_rarity_score(lhs)
                    + peak_symbol_distance_weight * peak_symbol_distance
                    + avg_symbol_distance_weight * avg_symbol_distance
                    + symbol_reuse_weight * symbol_reuse_count
                )

            lhs = max(eqns_remaining.keys(), key=lambda lhs: (score(lhs), disambiguation_rank[lhs]))

            if lhs in required_deps:
                partial_dep_ordering = dependencies.get_partial_order(required_deps[lhs], set(in_memory.keys()))

                for dep_set in partial_dep_ordering:
                    dep_dict = {lhs: eqns[lhs] for lhs in dep_set}
                    for dep in black_box_order(complexity_weight,
                                               rarity_weight,
                                               sqrt_rarity_weight,
                                               peak_symbol_distance_weight,
                                               avg_symbol_distance_weight,
                                               symbol_reuse_weight,
                                               in_memory,
                                               dep_dict):
                        del eqns_remaining[dep]
                        order[dep] = dep_dict[dep]
                        in_memory[dep] = idx
                        idx += 1

            rhs = eqns_remaining[lhs]
            del eqns_remaining[lhs]
            order[lhs] = rhs
            in_memory[lhs] = idx
            idx += 1

        put_cache(order, complexity_weight=complexity_weight, rarity_weight=rarity_weight, sqrt_rarity_weight=sqrt_rarity_weight,
                  peak_symbol_distance_weight=peak_symbol_distance_weight, avg_symbol_distance_weight=avg_symbol_distance_weight,
                  symbol_reuse_weight=symbol_reuse_weight)
        return order

    def black_box_score(complexity_weight: float,
                        rarity_weight: float,
                        sqrt_rarity_weight: float,
                        peak_symbol_distance_weight: float,
                        avg_symbol_distance_weight: float,
                        symbol_reuse_weight: float) -> float:
        order = black_box_order(complexity_weight, rarity_weight, sqrt_rarity_weight, peak_symbol_distance_weight, avg_symbol_distance_weight, symbol_reuse_weight)
        lifetime_data = _get_lifetimes(eqns, list(order.keys()))
        return (
            memory_pressure_factor * _score_memory_pressure_fast(lifetime_data.lifetimes)
            + peak_liveness_factor * _score_peak_liveness(lifetime_data.lifetimes, len(order))
            + symbol_reuse_factor * _score_symbol_reuse(lifetime_data, eqns, order)
        )

    optimizer = BayesianOptimization(
        f=black_box_score,
        pbounds={
            "complexity_weight": (-5.0, 5.0),
            "rarity_weight": (-0.0, 5.0),
            "sqrt_rarity_weight": (-5.0, 5.0),
            "peak_symbol_distance_weight": (-5.0, 0.0),
            "avg_symbol_distance_weight": (-5.0, 0.0),
            "symbol_reuse_weight": (0.0, 5.0)
        },
        #verbose=0
    )

    optimizer.maximize(init_points=exploration_iter, n_iter=optimization_iter)
    assert optimizer.max is not None
    pprint(f'Bayesian Optimization result: {optimizer.max}')

    opt_order = get_cache(**optimizer.max['params'])
    opt_liveness = _score_all_liveness(_get_lifetimes(eqns, list(opt_order.keys())), opt_order)

    yield from ((lhs, f'Liveness = {len(opt_liveness[lhs])}; {sorted(opt_liveness[lhs], key=str)}') for lhs in opt_order.keys())

bayesian_optimization.respects_dependency_order = True  # type: ignore[attr-defined]

def respects_dependency_order(fn: Any) -> bool:
    return (
        hasattr(fn, 'respects_dependency_order') and fn.respects_dependency_order
        or hasattr(fn, 'func') and respects_dependency_order(fn.func)
    )


def pre_cse_stand_in(fn: EqnOrderingFn) -> EqnOrderingFn:
    """
    The ordering function to run in place of `fn` before CSE. Bayesian optimization (a function, or a partial of one,
    whose name contains 'bayesian') is too expensive to run on the equations before CSE, whose order CSE rewrites
    anyway, so it is replaced by prioritize_rare_symbols; any other function is returned unchanged.
    """
    for candidate in (fn, getattr(fn, 'func', None)):
        if 'bayesian' in getattr(candidate, '__name__', ''):
            return prioritize_rare_symbols
    return fn
