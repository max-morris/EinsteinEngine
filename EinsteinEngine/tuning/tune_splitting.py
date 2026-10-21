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
Tuners that search a recipe's auto split predicates. Besides the predicates, they provide `auto_split_locus` and the
ordering functions they were given, which the recipe must read through `get_optional_tuning_param`. Position i means
"split after element i at the locus"; the number of positions is found by probing the recipe.

A `prefix` is prepended to every name a split tuner uses: its in-params and all its out-params (the predicates, the
locus and the ordering functions). Several split tuners, one per function, then fit in one CompositeTuner study: with
prefix `rhs_`, the recipe passes `rhs_auto_hard_split_predicate` and the other `rhs_` params to the function `rhs`.
"""

import os
import warnings
from abc import abstractmethod
from collections.abc import Collection
from typing import Any, Callable

from EinsteinEngine.intermediate.eqn_ordering import EqnOrderingFn
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy, retain_percentile
from EinsteinEngine.intermediate.split_locus import SplitLocus

from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.tuning.probe import ProbeResult, RankByPostPopulation
from EinsteinEngine.tuning.tuning import Tuner

HardSplitPredicate = Callable[[int], bool]
SoftSplitPredicate = Callable[[int], bool | SoftSplitRetainmentStrategy]

#  The out-param whose predicate is probed for the number of positions.
PROBED_PREDICATE = 'auto_hard_split_predicate'

#  Above this many positions, CombinatorialSplitTuner (two params per position) is not practical.
MAX_COMBINATORIAL_POSITIONS = 64

#  Warnings are attributed to the first caller outside this package.
_TUNING_DIR = os.path.dirname(__file__) + os.sep


class _SplitTuner(Tuner):
    """The locus and ordering functions shared by the split tuners, and the out-params that carry them."""

    def __init__(self, *, locus: SplitLocus, early_ordering_fn: EqnOrderingFn | RankByPostPopulation | None,
                 pre_population_ordering_fn: EqnOrderingFn | None, ordering_fn: EqnOrderingFn | None,
                 prefix: str) -> None:
        self.locus = locus
        self.early_ordering_fn = early_ordering_fn
        self.pre_population_ordering_fn = pre_population_ordering_fn
        self.ordering_fn = ordering_fn
        self.prefix = prefix

    def param_name(self, base: str) -> str:
        """The name of one of this tuner's params, with its prefix."""
        return self.prefix + base

    def probe_targets(self) -> Collection[str]:
        return (self.param_name(PROBED_PREDICATE),)

    def probe_params(self) -> dict[str, Any]:
        """The locus and the ordering functions that were given, which change the number of positions."""
        params: dict[str, Any] = {self.param_name('auto_split_locus'): self.locus}
        for name, fn in (('early_ordering_fn', self.early_ordering_fn),
                         ('pre_population_ordering_fn', self.pre_population_ordering_fn),
                         ('ordering_fn', self.ordering_fn)):
            if fn is not None:
                params[self.param_name(name)] = fn
        return params

    def get_experiment(self) -> Experiment:
        raise RuntimeError(f"{type(self).__name__} probes the recipe for its number of split positions. Build its "
                           f"experiment with tuning.build_experiment, or call get_probed_experiment with a probe "
                           f"result.")

    def get_probed_experiment(self, probe: ProbeResult) -> Experiment:
        target = self.param_name(PROBED_PREDICATE)
        if target not in probe:
            raise RuntimeError(f"Expected a probe result for {target!r}, got {dict(probe)}. Build the "
                               f"experiment with tuning.build_experiment, which probes the recipe.")
        return self._build(probe[target])

    @abstractmethod
    def _build(self, n_positions: int) -> Experiment:
        """The Experiment for `n_positions` split positions at the locus."""
        ...

    def _add_out_params(self, e: Experiment, hard: Callable[[dict[str, Any]], HardSplitPredicate],
                        soft: Callable[[dict[str, Any]], SoftSplitPredicate]) -> None:
        e.add_out_param(self.param_name('auto_hard_split_predicate'), hard)
        e.add_out_param(self.param_name('auto_soft_split_predicate'), soft)
        for name, value in self.probe_params().items():
            e.add_out_param(name, lambda _params, value=value: value)


class CombinatorialSplitTuner(_SplitTuner):
    """
    Search every position independently: no split, soft split (with a retainment percentile), or hard split.
    Params: `split_i` in {0: none, 1: soft, 2: hard} and `soft_retain_percentile_i` (only when `split_i` is 1), for i
    below `n_vars` if given, otherwise the probed number of positions; each name starts with `prefix`. With two params per position, this is only
    practical at the early locus; use CutPositionSplitTuner elsewhere.
    """

    def __init__(self, n_vars: int | None = None, *, locus: SplitLocus = SplitLocus.Early,
                 early_ordering_fn: EqnOrderingFn | RankByPostPopulation | None = None,
                 pre_population_ordering_fn: EqnOrderingFn | None = None,
                 ordering_fn: EqnOrderingFn | None = None, prefix: str = '') -> None:
        super().__init__(locus=locus, early_ordering_fn=early_ordering_fn,
                         pre_population_ordering_fn=pre_population_ordering_fn, ordering_fn=ordering_fn,
                         prefix=prefix)
        self.n_vars = n_vars

    def probe_targets(self) -> Collection[str]:
        return super().probe_targets() if self.n_vars is None else ()

    def get_experiment(self) -> Experiment:
        return super().get_experiment() if self.n_vars is None else self._build(self.n_vars)

    def get_probed_experiment(self, probe: ProbeResult) -> Experiment:
        return super().get_probed_experiment(probe) if self.n_vars is None else self._build(self.n_vars)

    def _build(self, n_positions: int) -> Experiment:
        if n_positions > MAX_COMBINATORIAL_POSITIONS:
            warnings.warn(f"CombinatorialSplitTuner has {n_positions} positions at the {self.locus.name} locus, "
                          f"too many to search with two params each. Use CutPositionSplitTuner instead.",
                          skip_file_prefixes=(_TUNING_DIR,))
        e = Experiment()

        splits = [self.param_name(f'split_{i}') for i in range(n_positions)]
        retains = [self.param_name(f'soft_retain_percentile_{i}') for i in range(n_positions)]
        for split, retain in zip(splits, retains):
            e.add_in_param(split, (0, 2))
            #  Python has completely insane lexical scoping, so we have to early-bind split in the lambda to capture
            #  by value
            e.add_in_param(retain, (0.0, 1.0), lambda params, split=split: params[split] == 1)

        def get_hard_split_predicate(params: dict[str, Any]) -> HardSplitPredicate:
            return lambda i: 0 <= i < n_positions and int(params[splits[i]]) == 2

        def get_soft_split_predicate(params: dict[str, Any]) -> SoftSplitPredicate:
            def f(i: int) -> bool | SoftSplitRetainmentStrategy:
                if not 0 <= i < n_positions or int(params[splits[i]]) != 1:
                    return False
                return retain_percentile(params[retains[i]])
            return f

        self._add_out_params(e, get_hard_split_predicate, get_soft_split_predicate)
        return e


class CutPositionSplitTuner(_SplitTuner):
    """
    Search the positions of `n_cuts` cuts, and the kind of each, among the probed positions. Params, for j below
    `n_cuts`: `cut_position_j` (strictly increasing), `cut_kind_j` in {0: unused, 1: soft, 2: hard}, where 0 is only
    searched with `allow_unused_cuts`, and `cut_retain_percentile_j` (only when `cut_kind_j` is 1); each name starts
    with `prefix`. The number of
    params does not grow with the number of positions, so this suits the later loci.
    """

    def __init__(self, n_cuts: int, *, locus: SplitLocus,
                 early_ordering_fn: EqnOrderingFn | RankByPostPopulation | None = None,
                 pre_population_ordering_fn: EqnOrderingFn | None = None,
                 ordering_fn: EqnOrderingFn | None = None, allow_unused_cuts: bool = True,
                 prefix: str = '') -> None:
        if n_cuts < 1:
            raise ValueError(f"n_cuts must be at least 1, got {n_cuts}")
        super().__init__(locus=locus, early_ordering_fn=early_ordering_fn,
                         pre_population_ordering_fn=pre_population_ordering_fn, ordering_fn=ordering_fn,
                         prefix=prefix)
        self.n_cuts = n_cuts
        self.allow_unused_cuts = allow_unused_cuts

    def _build(self, n_positions: int) -> Experiment:
        if n_positions < self.n_cuts:
            raise RuntimeError(f"Cannot place {self.n_cuts} cuts at distinct positions: the recipe only has "
                               f"{n_positions} positions at the {self.locus.name} locus.")
        e = Experiment()

        positions = e.add_distinct_sorted(self.param_name('cut_position'), self.n_cuts, (0, n_positions - 1))
        kinds = [self.param_name(f'cut_kind_{j}') for j in range(self.n_cuts)]
        retains = [self.param_name(f'cut_retain_percentile_{j}') for j in range(self.n_cuts)]
        for kind, retain in zip(kinds, retains):
            e.add_in_param(kind, (0 if self.allow_unused_cuts else 1, 2))
            e.add_in_param(retain, (0.0, 1.0), lambda params, kind=kind: params[kind] == 1)

        #  The positions are distinct, so no position is both hard and soft.
        def get_hard_split_predicate(params: dict[str, Any]) -> HardSplitPredicate:
            hard = {int(params[p]) for p, k in zip(positions, kinds) if int(params[k]) == 2}
            return lambda i: i in hard

        def get_soft_split_predicate(params: dict[str, Any]) -> SoftSplitPredicate:
            soft = {int(params[p]): float(params[r])
                    for p, k, r in zip(positions, kinds, retains) if int(params[k]) == 1}

            def f(i: int) -> bool | SoftSplitRetainmentStrategy:
                return False if i not in soft else retain_percentile(soft[i])
            return f

        self._add_out_params(e, get_hard_split_predicate, get_soft_split_predicate)
        return e
