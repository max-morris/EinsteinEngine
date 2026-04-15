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

from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from itertools import chain
from statistics import mean
from typing import Iterator, Callable, TYPE_CHECKING, NamedTuple

import line_profiler
from bayes_opt import BayesianOptimization
from sympy import Symbol, Expr

from util import pprint

if TYPE_CHECKING:
    from EinsteinEngine.dsl.eqnlist import EqnList

EqnOrderingFn = Callable[[dict[Symbol, Expr], 'EqnList'], Iterator[Symbol]]

@dataclass(frozen=True)
class _LifetimesData:
    lifetimes: dict[Symbol, tuple[int, int]]
    births: dict[int, set[Symbol]]
    deaths: dict[int, set[Symbol]]

def _get_lifetimes(eqns: dict[Symbol, Expr], order: list[Symbol]) -> _LifetimesData:
    if len(eqns) == 0 or len(order) == 0:
        return _LifetimesData(dict(), dict(), dict())

    assert len(eqns) == len(order), f"eqns and order must have the same length, but got {len(eqns)} and {len(order)}"

    first_read: dict[Symbol, int] = dict()
    last_read: dict[Symbol, int] = dict()

    for idx, (_, rhs) in enumerate(sorted(eqns.items(), key=lambda eqn: order.index(eqn[0]))):
        for sym in rhs.free_symbols:
            if sym not in first_read:
                first_read[sym] = idx
            last_read[sym] = idx

    lifetimes = {sym: (first_read[sym], last_read[sym]) for sym in first_read}
    births = defaultdict(set)
    deaths = defaultdict(set)

    for sym, (birth, death) in lifetimes.items():
        births[birth].add(sym)
        deaths[death].add(sym)

    return _LifetimesData(lifetimes, births, deaths)

def _score_memory_pressure(lifetimes: dict[Symbol, tuple[int, int]]) -> dict[Symbol, int]:
    return {sym: 1 + lifetime[1] - lifetime[0] for (sym, lifetime) in lifetimes.items()}

def score_memory_pressure(eqns: dict[Symbol, Expr], order: list[Symbol]) -> dict[Symbol, int]:
    return _score_memory_pressure(_get_lifetimes(eqns, order).lifetimes)

def _score_peak_liveness(lifetimes: dict[Symbol, tuple[int, int]], n_eqns: int) -> int:
    peak = 0
    liveness = 0

    for idx in range(n_eqns):
        for (birth, death) in lifetimes.values():
            if birth == idx:
                liveness += 1
            elif death == idx - 1:
                liveness -= 1

        assert liveness >= 0, f"Negative liveness at index {idx}"
        peak = max(peak, liveness)

    return peak

def score_peak_liveness(eqns: dict[Symbol, Expr], order: list[Symbol]) -> int:
    return _score_peak_liveness(_get_lifetimes(eqns, order).lifetimes, len(order))

def _score_symbol_reuse(lifetimes_data: _LifetimesData, eqns: dict[Symbol, Expr], order: list[Symbol]) -> int:
    if len(eqns) == 0:
        return 0

    births, deaths = lifetimes_data.births, lifetimes_data.deaths

    in_memory: set[Symbol] = set()
    score = 0

    for idx, (_, rhs) in enumerate(sorted(eqns.items(), key=lambda eqn: order.index(eqn[0]))):
        in_memory.update(births[idx])
        if idx > 0:
            in_memory.difference_update(deaths[idx - 1])
        score += len(in_memory.intersection(rhs.free_symbols))

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
    in_memory.update(rhs.free_symbols)
    yield lhs

    while len(eqns_remaining) > 0:
        lhs, rhs = max(eqns_remaining.items(),
                       key=lambda kv: (len(kv[1].free_symbols.intersection(in_memory)), eqn_list.complexity[kv[0]],
                                       disambiguation.index(kv[0])))
        del eqns_remaining[lhs]
        in_memory.update(rhs.free_symbols)
        yield lhs


def _get_eqn_score_fn_by_rarity(eqns: dict[Symbol, Expr], consider_frequency: bool = True) -> Callable[[Symbol], float]:
    if len(eqns) == 0:
        return lambda _: 0.0

    reciprocal_rarity: dict[Symbol, float] = defaultdict(int)
    frequency_by_eqn: dict[Symbol, dict[Symbol, float]] = defaultdict(dict)  # {lhs: {sym: freq}}
    for lhs, rhs in eqns.items():
        for sym in rhs.free_symbols:
            reciprocal_rarity[sym] += 1
            frequency_by_eqn[lhs][sym] = rhs.count(sym)  # type: ignore[no-untyped-call]

    def symbol_rarity(sym: Symbol) -> float:
        return 1 / reciprocal_rarity[sym]

    def symbol_score(sym: Symbol, lhs: Symbol) -> float:
        return frequency_by_eqn[lhs][sym] * symbol_rarity(sym) if consider_frequency else symbol_rarity(sym)

    def eqn_score(lhs: Symbol) -> float:
        return sum(symbol_score(sym, lhs) for sym in eqns[lhs].free_symbols)

    return eqn_score


def _get_eqn_score_fn_by_complexity(eqn_list: EqnList) -> Callable[[Symbol], float]:
    return lambda lhs: eqn_list.complexity[lhs]


def _score_eqn_by_symbol_reuse(eqn_rhs: Expr, previous_rhses: set[Expr]) -> int:
    if len(previous_rhses) == 0:
        return 0

    return len(eqn_rhs.free_symbols.intersection(chain(*(rhs.free_symbols for rhs in previous_rhses))))


@line_profiler.profile
def _get_reused_symbol_distances(eqn_rhs: Expr, in_memory: dict[Symbol, int], my_pos: int) -> dict[Symbol, int]:
    if len(in_memory) == 0:
        return dict()

    return {sym: my_pos - in_memory[sym] for sym in eqn_rhs.free_symbols.intersection(in_memory.keys())}

class _SymbolDistanceData(NamedTuple):
    peak: float
    avg: float

def _score_eqn_by_symbol_distance(reused_symbols: dict[Symbol, int]) -> _SymbolDistanceData:
    return _SymbolDistanceData(
        max(reused_symbols.values(), default=0),
        mean(reused_symbols.values()) if len(reused_symbols) > 0 else 0
    )


def prioritize_rare_symbols(eqns: dict[Symbol, Expr],
                            eqn_list: EqnList,
                            consider_frequency: bool = True,
                            complexity_factor: float = 0.0) -> Iterator[Symbol]:
    """
    Orders equations based on symbol rarity.
    Equations which use symbols that are less common in other equations are given higher priority.

    To determine the rarity of a symbol, multiple occurrences of the same symbol in an equation are treated as one.
    If `consider_frequency` is true, when evaluating the overall priority of an equation, the rarity of each symbol is weighted positively by the frequency of that symbol in the equation.

    The complexity score of an equation, scaled by `complexity_factor`, is added to the priority.
    """

    if len(eqns) == 0:
        return

    raw_eqn_score = _get_eqn_score_fn_by_rarity(eqns, consider_frequency)
    eqn_score = lambda lhs: raw_eqn_score(lhs) + (complexity_factor * eqn_list.complexity[lhs])

    disambiguation = sorted(eqns.keys(), key=str, reverse=True)
    ordered = sorted(eqns.keys(), key=lambda lhs: (eqn_score(lhs), eqn_list.complexity[lhs], disambiguation.index(lhs)), reverse=True)

    yield from ordered.__iter__()


@line_profiler.profile
def bayesian_optimization(eqns: dict[Symbol, Expr],
                          eqn_list: EqnList,
                          memory_pressure_factor: float = 0.0,
                          peak_liveness_factor: float = -1.0,
                          symbol_reuse_factor: float = 0.0) -> Iterator[Symbol]:
    rarity_score = _get_eqn_score_fn_by_rarity(eqns, consider_frequency=True)

    order_cache: dict[tuple[tuple[str, float], ...], OrderedDict[Symbol, Expr]] = dict()
    def put_cache(val: OrderedDict[Symbol, Expr], **kwargs: float) -> None:
        key = tuple(sorted(kwargs.items()))
        order_cache[key] = val

    def get_cache(**kwargs: float) -> OrderedDict[Symbol, Expr]:
        key = tuple(sorted(kwargs.items()))
        return order_cache[key]

    @line_profiler.profile
    def black_box_order(complexity_weight: float,
                        rarity_weight: float,
                        peak_symbol_distance_weight: float,
                        avg_symbol_distance_weight: float,
                        symbol_reuse_weight: float) -> OrderedDict[Symbol, Expr]:
        eqns_remaining = eqns.copy()
        disambiguation = sorted(eqns_remaining.keys(), key=str, reverse=True)

        order: OrderedDict[Symbol, Expr] = OrderedDict()
        in_memory: dict[Symbol, int] = dict()  # symbol -> index in order

        for idx in range(len(eqns)):
            @line_profiler.profile
            def score(lhs: Symbol) -> float:
                symbol_distances = _get_reused_symbol_distances(eqns[lhs], in_memory, idx)
                peak_symbol_distance, avg_symbol_distance = _score_eqn_by_symbol_distance(symbol_distances)

                return (
                    complexity_weight * eqn_list.complexity[lhs]
                    + rarity_weight * rarity_score(lhs)
                    + peak_symbol_distance_weight * peak_symbol_distance
                    + avg_symbol_distance_weight * avg_symbol_distance
                    + symbol_reuse_weight * len(symbol_distances)
                )

            lhs = max(eqns_remaining.keys(), key=lambda lhs: (score(lhs), disambiguation.index(lhs)))
            rhs = eqns_remaining[lhs]
            del eqns_remaining[lhs]
            order[lhs] = rhs
            in_memory.update({sym: idx for sym in rhs.free_symbols})

        put_cache(order, complexity_weight=complexity_weight, rarity_weight=rarity_weight,
                  peak_symbol_distance_weight=peak_symbol_distance_weight, avg_symbol_distance_weight=avg_symbol_distance_weight,
                  symbol_reuse_weight=symbol_reuse_weight)
        return order

    @line_profiler.profile
    def black_box_score(complexity_weight: float,
                        rarity_weight: float,
                        peak_symbol_distance_weight: float,
                        avg_symbol_distance_weight: float,
                        symbol_reuse_weight: float) -> float:
        order = black_box_order(complexity_weight, rarity_weight, peak_symbol_distance_weight, avg_symbol_distance_weight, symbol_reuse_weight)
        lifetime_data = _get_lifetimes(eqns, list(order.keys()))
        return (
                peak_liveness_factor * _score_peak_liveness(lifetime_data.lifetimes, len(order))
            #memory_pressure_factor * sum(_score_memory_pressure(lifetime_data.lifetimes).values())
            #+ peak_liveness_factor * _score_peak_liveness(lifetime_data.lifetimes, len(order))
            #+ symbol_reuse_factor * _score_symbol_reuse(lifetime_data, eqns, list(order.keys()))
        )

    optimizer = BayesianOptimization(
        f=black_box_score,
        pbounds={
            "complexity_weight": (-5.0, 5.0),
            "rarity_weight": (-0.0, 5.0),
            "peak_symbol_distance_weight": (-5.0, 0.0),
            "avg_symbol_distance_weight": (-5.0, 0.0),
            "symbol_reuse_weight": (0.0, 5.0)
        },
        #verbose=0
    )

    optimizer.maximize(init_points=5, n_iter=10)
    assert optimizer.max is not None
    pprint(f'Bayesian Optimization result: {optimizer.max}')

    opt_order = get_cache(**optimizer.max['params'])
    yield from opt_order.keys()

