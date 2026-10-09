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

"""Tuners that search a recipe's auto split predicates.

Both tuners drive the ``auto_hard_split_predicate`` and ``auto_soft_split_predicate`` out-params, and also provide
``auto_split_locus`` and whichever of ``early_ordering_fn``, ``pre_population_ordering_fn`` and ``ordering_fn`` were
given. The recipe must read those through ``get_optional_tuning_param`` and pass them to ``create_function`` / the
bake options; probing checks that it reads every one. Positions are 0-based: position ``i`` means "split after element
``i`` at the locus", and a cut is a hard or soft split at one position. The number of positions is found by probing
the recipe (see ``EinsteinEngine.tuning.probe``), with the same locus and ordering functions that the trials use.
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
from EinsteinEngine.tuning.probe import ProbeResult
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

    def __init__(self, *, locus: SplitLocus, early_ordering_fn: EqnOrderingFn | None,
                 pre_population_ordering_fn: EqnOrderingFn | None, ordering_fn: EqnOrderingFn | None) -> None:
        self.locus = locus
        self.early_ordering_fn = early_ordering_fn
        self.pre_population_ordering_fn = pre_population_ordering_fn
        self.ordering_fn = ordering_fn

    def probe_targets(self) -> Collection[str]:
        return (PROBED_PREDICATE,)

    def probe_params(self) -> dict[str, Any]:
        """The out-params that are the same in every trial: the locus and the ordering functions that were given.

        They change the number of positions, so the recipe is probed with them too.
        """
        params: dict[str, Any] = {'auto_split_locus': self.locus}
        for name, fn in (('early_ordering_fn', self.early_ordering_fn),
                         ('pre_population_ordering_fn', self.pre_population_ordering_fn),
                         ('ordering_fn', self.ordering_fn)):
            if fn is not None:
                params[name] = fn
        return params

    def get_experiment(self) -> Experiment:
        raise RuntimeError(f"{type(self).__name__} probes the recipe for its number of split positions. Build its "
                           f"experiment with tuning.build_experiment (as remote_tuner, generate_best and plot_tuning "
                           f"do), or call get_probed_experiment with a probe result.")

    def get_probed_experiment(self, probe: ProbeResult) -> Experiment:
        if PROBED_PREDICATE not in probe:
            raise RuntimeError(f"Expected a probe result for {PROBED_PREDICATE!r}, got {dict(probe)}. Build the "
                               f"experiment with tuning.build_experiment, which probes the recipe.")
        return self._build(probe[PROBED_PREDICATE])

    @abstractmethod
    def _build(self, n_positions: int) -> Experiment:
        """The Experiment for ``n_positions`` split positions at the locus."""
        ...

    def _add_out_params(self, e: Experiment, hard: Callable[[dict[str, Any]], HardSplitPredicate],
                        soft: Callable[[dict[str, Any]], SoftSplitPredicate]) -> None:
        e.add_out_param('auto_hard_split_predicate', hard)
        e.add_out_param('auto_soft_split_predicate', soft)
        for name, value in self.probe_params().items():
            e.add_out_param(name, lambda _params, value=value: value)


class CombinatorialSplitTuner(_SplitTuner):
    """Search every position independently: no split, soft split (with a retainment percentile), or hard split.

    Params: ``split_i`` in {0: none, 1: soft, 2: hard} and ``soft_retain_percentile_i`` (only when ``split_i`` is
    1), for ``i`` in ``0..N-1``. ``N`` is ``n_vars`` if given, otherwise the probed number of positions. The
    predicates never split at ``i >= N``.

    With two params per position this is only practical at the early locus (about a dozen positions for Z4c); use
    CutPositionSplitTuner at the other loci. A warning is printed above MAX_COMBINATORIAL_POSITIONS positions.
    """

    def __init__(self, n_vars: int | None = None, *, locus: SplitLocus = SplitLocus.Early,
                 early_ordering_fn: EqnOrderingFn | None = None,
                 pre_population_ordering_fn: EqnOrderingFn | None = None,
                 ordering_fn: EqnOrderingFn | None = None) -> None:
        super().__init__(locus=locus, early_ordering_fn=early_ordering_fn,
                         pre_population_ordering_fn=pre_population_ordering_fn, ordering_fn=ordering_fn)
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
                          f"which is too many to search one by one (it declares two params per position). Use "
                          f"CutPositionSplitTuner, whose number of params does not grow with the number of positions.",
                          skip_file_prefixes=(_TUNING_DIR,))
        e = Experiment()

        for i in range(n_positions):
            e.add_in_param(f'split_{i}', (0, 2))
            #  Python has completely insane lexical scoping, so we have to early-bind i in the lambda to capture by value
            e.add_in_param(f'soft_retain_percentile_{i}', (0.0, 1.0), lambda params, i=i: params[f'split_{i}'] == 1)

        def get_hard_split_predicate(params: dict[str, Any]) -> HardSplitPredicate:
            return lambda i: 0 <= i < n_positions and int(params[f'split_{i}']) == 2

        def get_soft_split_predicate(params: dict[str, Any]) -> SoftSplitPredicate:
            def f(i: int) -> bool | SoftSplitRetainmentStrategy:
                if not 0 <= i < n_positions or int(params[f'split_{i}']) != 1:
                    return False
                return retain_percentile(params[f'soft_retain_percentile_{i}'])
            return f

        self._add_out_params(e, get_hard_split_predicate, get_soft_split_predicate)
        return e


class CutPositionSplitTuner(_SplitTuner):
    """Search the positions of ``n_cuts`` cuts, and the kind of each, among the probed positions.

    A cut is a hard or soft split at one position. Params, for ``j`` in ``0..n_cuts-1``:

    - ``cut_position_j``: strictly increasing positions in ``[0, N-1]`` (see ``Experiment.add_distinct_sorted``).
    - ``cut_kind_j`` in {0: unused, 1: soft, 2: hard}; 0 is only searched when ``allow_unused_cuts`` is True, which
      lets the search use fewer than ``n_cuts`` cuts.
    - ``cut_retain_percentile_j``: the soft split retainment percentile, only when ``cut_kind_j`` is 1.

    Unlike CombinatorialSplitTuner, the number of params does not grow with the number of positions, so this suits
    the pre-population and post-population loci.
    """

    def __init__(self, n_cuts: int, *, locus: SplitLocus, early_ordering_fn: EqnOrderingFn | None = None,
                 pre_population_ordering_fn: EqnOrderingFn | None = None,
                 ordering_fn: EqnOrderingFn | None = None, allow_unused_cuts: bool = True) -> None:
        if n_cuts < 1:
            raise ValueError(f"n_cuts must be at least 1, got {n_cuts}")
        super().__init__(locus=locus, early_ordering_fn=early_ordering_fn,
                         pre_population_ordering_fn=pre_population_ordering_fn, ordering_fn=ordering_fn)
        self.n_cuts = n_cuts
        self.allow_unused_cuts = allow_unused_cuts

    def _build(self, n_positions: int) -> Experiment:
        if n_positions < self.n_cuts:
            raise RuntimeError(f"Cannot place {self.n_cuts} cuts at distinct positions: the recipe only has "
                               f"{n_positions} positions at the {self.locus.name} locus.")
        e = Experiment()

        positions = e.add_distinct_sorted('cut_position', self.n_cuts, (0, n_positions - 1))
        kinds = [f'cut_kind_{j}' for j in range(self.n_cuts)]
        retains = [f'cut_retain_percentile_{j}' for j in range(self.n_cuts)]
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
