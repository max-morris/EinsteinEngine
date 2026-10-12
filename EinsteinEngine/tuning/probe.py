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

"""Probe a recipe for the number of auto split positions its functions expose.

A tuner that searches split points needs to know how many positions there are before it can declare its search
space. ``probe_recipe`` finds out by running the recipe with a placeholder predicate (which never splits) in place of
each probed out-param. When a function evaluates its auto split predicates at its locus, the engine calls
``report_split_positions`` with the predicate object and the number of positions; the probe maps the object back to
its out-param name. At the end of every ``DslFrontend.bake`` the engine calls ``bake_finished``.

``derive_add_eqn_order`` runs the recipe the same way to resolve a tuner's ``rank_by_post_population(fn)``: the
function baked with ``fn`` has its add_eqn calls ranked by where ``fn`` puts their equations after CSE. The engine calls
``bake_started`` and ``cse_finished`` so that the probe can snapshot the calls' equations and read the post-CSE order.

Limitations:

- The recipe is stopped at the end of the first bake after which every placeholder has been reported. Everything the
  recipe does before that point still happens. In particular, a recipe that bakes and generates several thorns emits
  the code of each thorn it generates before that point (into the working directory).
- If some placeholder is never reported, nothing stops the recipe: it runs to completion, emitting all of its code,
  and the probe then fails.
- The check that the recipe reads every provided name only sees the reads made before the recipe is stopped.
- While probing, ``get_tuning_param`` returns the recipe's default for every name the probe does not provide.

This module is imported by the engine, so it must stay lightweight: no optuna, and no import of
``EinsteinEngine.tuning.tuning`` at module import time.
"""

import functools
import runpy
import sys
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sympy import Symbol

# Re-exported: describe_param lives in common so that the intermediate layer can use it without importing tuning.
from EinsteinEngine.common.describe_param import describe_param as describe_param

if TYPE_CHECKING:
    from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend
    from EinsteinEngine.intermediate.eqn_ordering import EqnOrderingFn


class _ProbeComplete(BaseException):
    """Raised by ``bake_finished`` to stop the recipe once the probe has everything it needs.

    It derives from BaseException so that a recipe's ``except Exception`` cannot swallow it.
    """


class _PlaceholderPredicate:
    """A split predicate that never splits. One distinct instance stands in for each probed out-param."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, _position: int, /) -> bool:
        return False

    def __repr__(self) -> str:
        return f'_PlaceholderPredicate({self.name!r})'


@dataclass
class _ProbeState:
    recipe_path: str
    placeholders: dict[str, _PlaceholderPredicate]
    params: dict[str, Any]
    # out-param name -> function name -> the distinct counts that function reported
    reports: dict[str, dict[str, set[int]]] = field(default_factory=dict)
    # The names the recipe asked for (through get_tuning_param / get_optional_tuning_param)
    read: set[str] = field(default_factory=set)
    # While deriving: the ordering function the ranked function is baked with, each function's add_eqn origins at the
    # start of the bake, and the derived add_eqn order of each function baked with ranking_fn.
    ranking_fn: Any = None
    origins: dict[str, list[dict[Symbol, int]]] = field(default_factory=dict)
    derived: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def name_of(self, predicate: object) -> str | None:
        for name, placeholder in self.placeholders.items():
            if predicate is placeholder:
                return name
        return None

    def done(self) -> bool:
        return all(name in self.reports for name in self.placeholders) and (self.ranking_fn is None or bool(self.derived))


_state: _ProbeState | None = None


def active_params() -> Mapping[str, Any] | None:
    """The recipe-facing params provided while probing, or None when no probe is running.

    Used by ``get_tuning_param``: while probing, a provided name returns its value and any other name returns the
    recipe's default. The caller reports each name it looks up with ``note_read``.
    """
    return None if _state is None else _state.params


def note_read(name: str) -> None:
    """Record that the recipe read the tuning param ``name``. No-op unless probing."""
    if _state is not None:
        _state.read.add(name)


def report_split_positions(predicate: object, function_name: str, count: int) -> None:
    """Engine hook: ``function_name`` is about to evaluate ``predicate`` at positions ``0..count-1``.

    No-op unless probing, and for predicates that are not probe placeholders.
    """
    if _state is None:
        return
    name = _state.name_of(predicate)
    if name is None:
        return
    _state.reports.setdefault(name, {}).setdefault(function_name, set()).add(count)


def deriving() -> bool:
    """Engine hook: whether an add_eqn order is being derived, during which no function makes auto splits."""
    return _state is not None and _state.ranking_fn is not None


def bake_started(frontend: 'DslFrontend[Any, Any, Any]') -> None:
    """Engine hook, called at the start of ``DslFrontend.bake``. While deriving, snapshots each function's add_eqn
    origins, before pull-out temporaries inherit them."""
    if _state is None or _state.ranking_fn is None:
        return
    _state.origins = {name: [dict(el.eqn_origin) for el in tf.eqn_complex.eqn_lists]
                      for name, tf in frontend.functions.items()}


def cse_finished(frontend: 'DslFrontend[Any, Any, Any]') -> None:
    """Engine hook, called by ``DslFrontend.bake`` after global CSE (or where it would run). While deriving, ranks the
    add_eqn calls of each function baked with the ranking ordering function by their post-CSE positions."""
    if _state is None or _state.ranking_fn is None:
        return
    from EinsteinEngine.intermediate.eqn_grouping import rank_add_eqn_calls
    for name, tf in frontend.functions.items():
        eqn_lists = tf.eqn_complex.eqn_lists
        if name in _state.origins and any(el.ordering_fn is _state.ranking_fn for el in eqn_lists):
            orders = [list(el.order) for el in eqn_lists]
            _state.derived[name] = tuple(rank_add_eqn_calls(_state.origins[name], orders, tf.eqn_complex.origin_count))


def bake_finished() -> None:
    """Engine hook, called at the very end of ``DslFrontend.bake``.

    No-op unless probing. Once every placeholder predicate has been reported (and, while deriving, an order derived),
    stops the recipe, so that it does not go on to emit code. Until then it does nothing, and the recipe continues
    (see the module docstring).
    """
    if _state is not None and _state.done():
        raise _ProbeComplete()


class ProbeResult(Mapping[str, int]):
    """An immutable mapping from probed out-param name to the number of split positions at its locus."""

    def __init__(self, counts: Mapping[str, int] | None = None) -> None:
        self._counts: dict[str, int] = dict(sorted((counts or {}).items()))

    def __getitem__(self, name: str) -> int:
        return self._counts[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._counts)

    def __len__(self) -> int:
        return len(self._counts)

    def __repr__(self) -> str:
        return f'ProbeResult({self._counts!r})'

    @staticmethod
    def from_counts(obj: object) -> 'ProbeResult':
        """Validate and wrap a JSON-decoded ``{name: count}`` object."""
        if not isinstance(obj, dict):
            raise ValueError(f'Expected a JSON object of probe counts, got {obj!r}')
        for name, count in obj.items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f'Probe count for {name!r} must be a non-negative integer, got {count!r}')
        return ProbeResult(obj)


EMPTY_PROBE = ProbeResult()


def probe_recipe(recipe_path: str, predicate_names: Collection[str],
                 extra_params: Mapping[str, Any] | None = None) -> ProbeResult:
    """Run ``recipe_path`` and count the split positions seen by each out-param in ``predicate_names``.

    Each name is bound to a placeholder predicate that never splits. ``extra_params`` supplies fixed recipe-facing
    params that change the count (e.g. ``auto_split_locus`` or an ordering function); every other
    ``get_tuning_param`` call returns the recipe's default. The recipe runs as ``remote_feedback`` runs it (as
    ``__main__``, with ``sys.argv`` reset), and is stopped at the end of the first bake after which every placeholder
    has been reported (see the module docstring for what it may do before then).

    Raises RuntimeError if the recipe fails, never reads one of the provided names (an ordering function or locus it
    ignores would make the count wrong), never evaluates a placeholder, or evaluates one placeholder with different
    counts in different functions.
    """
    names = list(dict.fromkeys(predicate_names))
    if not names:
        return EMPTY_PROBE
    extra = dict(extra_params or {})
    if overlap := sorted(set(names) & set(extra)):
        raise ValueError(f'Out-params {overlap} are both probed and passed as extra_params.')

    placeholders = {name: _PlaceholderPredicate(name) for name in names}
    state = _ProbeState(recipe_path, placeholders, {**extra, **placeholders})
    _run(state, f'while being probed for {names}')
    _check_read(state)

    counts: dict[str, int] = {}
    for name in state.placeholders:
        by_function = state.reports.get(name)
        if by_function is None:
            raise RuntimeError(
                f'Recipe {state.recipe_path} never evaluated the predicate for out-param {name!r}. Does the recipe '
                f'pass get_tuning_param({name!r}, ...) to create_function, and is auto_split_locus set to a locus '
                f'the function reaches?')
        distinct = {c for cs in by_function.values() for c in cs}
        if len(distinct) != 1:
            detail = ', '.join(f'{fn} ({"/".join(str(c) for c in sorted(cs))})'
                               for fn, cs in sorted(by_function.items()))
            raise RuntimeError(
                f'The predicate for out-param {name!r} was evaluated with different numbers of split positions: '
                f'{detail}. Give each function its own out-param.')
        counts[name] = distinct.pop()
    return ProbeResult(counts)


class RankByPostPopulation:
    """A tuner's early ordering function that is derived from the recipe; see `rank_by_post_population`."""

    def __init__(self, ordering_fn: 'EqnOrderingFn') -> None:
        self.ordering_fn = ordering_fn

    def __repr__(self) -> str:
        return f'rank_by_post_population({describe_param(self.ordering_fn)})'


def rank_by_post_population(ordering_fn: 'EqnOrderingFn') -> RankByPostPopulation:
    """
    For a tuner's ``early_ordering_fn``: order the add_eqn calls of the tuned function by where `ordering_fn` puts
    their equations after CSE.

    ``tuning.build_experiment`` resolves it once per checkpoint with ``derive_add_eqn_order`` and records the order in
    the checkpoint's probe file. The probe and every trial then use ``add_eqn_order(<that order>)``, and so does
    ``generate_best``, which reads the recorded order instead of deriving it again.
    """
    return RankByPostPopulation(ordering_fn)


def derive_add_eqn_order(recipe_path: str, ordering_fn: 'EqnOrderingFn',
                         extra_params: Mapping[str, Any] | None = None) -> tuple[str, tuple[int, ...]]:
    """Run ``recipe_path`` without auto splits, with `ordering_fn` as the ``ordering_fn`` out-param and no early (or,
    if ``extra_params`` provides one, pre-population) ordering function, and rank the add_eqn calls of the function
    baked with it by the median post-CSE position of the equations each call produced
    (``eqn_grouping.rank_add_eqn_calls``).

    ``extra_params`` supplies other fixed recipe-facing params, as for ``probe_recipe``. Returns the function's name
    and its 0-based add_eqn order. Raises RuntimeError if the recipe fails, never reads one of the provided names, or
    does not bake exactly one function with `ordering_fn`.
    """
    # A distinct object, so the hooks can tell the ranked function by its ordering function; pre_cse_stand_in and
    # respects_dependency_order look through the partial, so it bakes exactly as ordering_fn would.
    ranking_fn = functools.partial(ordering_fn)
    params = {**(extra_params or {}), 'early_ordering_fn': None, 'ordering_fn': ranking_fn}
    if 'pre_population_ordering_fn' in params:
        params['pre_population_ordering_fn'] = None
    state = _ProbeState(recipe_path, {}, params, ranking_fn=ranking_fn)
    _run(state, 'while deriving an add_eqn order')
    _check_read(state)
    if len(state.derived) != 1:
        raise RuntimeError(f'Recipe {recipe_path} must bake exactly one function with the ordering_fn out-param to '
                           f'derive its add_eqn order, but baked {sorted(state.derived) or "none"}.')
    return next(iter(state.derived.items()))


def _run(state: _ProbeState, doing: str) -> None:
    global _state
    if _state is not None:
        raise RuntimeError('probe_recipe cannot be nested.')
    # Imported here: these import the engine, which imports this module. The recipe advances their automatic name
    # counters, which are restored so that the probe does not change the names a later run in this process generates.
    from EinsteinEngine.frontend.dsl.cactus.carpetx import ExplicitSyncBatch, NewRadXBoundaryBatch
    from EinsteinEngine.tuning.clear_caches import clear_caches
    saved_counters = ExplicitSyncBatch._name_counter, NewRadXBoundaryBatch._name_counter
    saved_argv = sys.argv
    _state = state
    try:
        sys.argv = [state.recipe_path]
        try:
            runpy.run_path(state.recipe_path, run_name='__main__')
        except _ProbeComplete:
            pass
        except Exception as e:
            raise RuntimeError(f'Recipe {state.recipe_path} failed {doing}.') from e
    finally:
        _state = None
        sys.argv = saved_argv
        ExplicitSyncBatch._name_counter, NewRadXBoundaryBatch._name_counter = saved_counters
        clear_caches()


def _check_read(state: _ProbeState) -> None:
    if unread := [name for name in state.params if name not in state.read]:
        raise RuntimeError(
            f'Recipe {state.recipe_path} never read the tuner out-param(s) {unread} while being probed. The tuner '
            f'provides them because they are the probed split predicates or change the number of split positions '
            f'(the locus and the ordering functions), so the probe would count the wrong positions. Read each one '
            f'with get_optional_tuning_param(name, default) (or get_tuning_param) and pass it to create_function or '
            f'bake; see docs/TUNING-QUICKSTART.md, section 1. Only reads made before the probe stopped the recipe '
            f'(at the end of the first bake after which every probed predicate was evaluated) are seen, so a name '
            f'read only later (e.g. for a second thorn) counts as never read.')
