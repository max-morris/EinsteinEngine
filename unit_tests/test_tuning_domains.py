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

from contextlib import contextmanager
from typing import Any, Iterator

from EinsteinEngine.tuning.experiment import (
    Discrete,
    Experiment,
    InfeasibleParamError,
    Interval,
    Union,
)


def _approx(x: float, y: float, tol: float = 1e-9) -> bool:
    return abs(x - y) <= tol


@contextmanager
def _raises(exc: type[BaseException]) -> Iterator[None]:
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")


class _ScriptedTrial:
    """A TrialObj that returns caller-chosen coordinates (or the low end)."""

    def __init__(self, picks: dict[str, int | float] | None = None) -> None:
        self._picks = picks or {}

    def suggest_int(self, name: str, lo: int, hi: int, /) -> int:
        v = int(self._picks.get(name, lo))
        assert lo <= v <= hi, (name, v, lo, hi)
        return v

    def suggest_float(self, name: str, lo: float, hi: float, /) -> float:
        v = float(self._picks.get(name, lo))
        assert lo <= v <= hi, (name, v, lo, hi)
        return v


def test_interval_coord_equals_value() -> None:
    dom = Interval(3, 9)
    assert dom.coord_distribution({}) == (int, 3, 9)
    assert dom.value_of(7, {}) == 7
    fdom = Interval(0.0, 1.0)
    assert fdom.coord_distribution({}).kind is float


def test_int_union_maps_index_and_skips_gap() -> None:
    dom = Union([(0, 3), (5, 6)])
    spec = dom.coord_distribution({})
    assert spec == (int, 0, 5)
    mapped = [dom.value_of(i, {}) for i in range(6)]
    assert mapped == [0, 1, 2, 3, 5, 6]
    assert 4 not in mapped  # the gap is never produced


def test_float_union_piecewise_inverse_cdf() -> None:
    dom = Union([(0.0, 3.0), (5.0, 6.0)])
    assert dom.coord_distribution({}) == (float, 0.0, 4.0)
    assert _approx(dom.value_of(0.0, {}), 0.0)
    assert _approx(dom.value_of(1.5, {}), 1.5)
    assert _approx(dom.value_of(3.0, {}), 3.0)
    assert _approx(dom.value_of(3.5, {}), 5.5)  # jumps across the gap
    assert _approx(dom.value_of(4.0, {}), 6.0)


def test_discrete_sorts_and_dedups() -> None:
    dom = Discrete([30, 10, 20, 10])
    assert dom.coord_distribution({}) == (int, 0, 2)
    assert [dom.value_of(i, {}) for i in range(3)] == [10, 20, 30]


def test_tuple_bounds_still_accepted() -> None:
    exp = Experiment()
    exp.add_in_param('x', (0, 5))
    exp.add_out_param('x', lambda r: r['x'])
    res = exp.suggest_params(_ScriptedTrial({'x': 4}))
    assert res.realized_params['x'] == 4  # coord == value for an Interval
    assert res.recipe_facing_args['x'] == 4


def test_distinct_sorted_is_strictly_increasing_without_rejection() -> None:
    exp = Experiment()
    names = exp.add_distinct_sorted('p', 3, (1, 6))
    exp.add_out_param('vals', lambda r: tuple(r[n] for n in names))
    # Push every coordinate to the top of its narrowed interval; still valid.
    res = exp.suggest_params(_ScriptedTrial({'p_0': 4, 'p_1': 5, 'p_2': 6}))
    vals = res.recipe_facing_args['vals']
    assert vals == (4, 5, 6)
    assert list(vals) == sorted(set(vals))  # strictly increasing => distinct


def test_distinct_choice_yields_a_permutation() -> None:
    exp = Experiment()
    names = exp.add_distinct_choice('c', 3, [10, 20, 30, 40])
    exp.add_out_param('vals', lambda r: tuple(r[n] for n in names))
    # Each param indexes into the remaining pool; index 1 each time.
    res = exp.suggest_params(_ScriptedTrial({'c_0': 1, 'c_1': 1, 'c_2': 1}))
    vals = res.recipe_facing_args['vals']
    assert vals == (20, 30, 40)  # 20; then {10,30,40}->30; then {10,40}->40
    assert len(set(vals)) == 3


def test_constraint_still_rejects_irreducible_predicates() -> None:
    exp = Experiment()
    exp.add_in_param('n', (1, 10), constraint=lambda v: v % 2 == 1)
    with _raises(InfeasibleParamError):
        exp.suggest_params(_ScriptedTrial({'n': 4}))


def test_condition_and_domains_see_values_not_coords() -> None:
    # A dynamic union followed by a condition that reads the mapped value.
    exp = Experiment()
    exp.add_in_param('a', Union([(0, 1), (10, 11)]))
    seen: dict[str, Any] = {}

    def cond(realized: dict[str, int | float]) -> bool:
        seen['a'] = realized.get('a')
        return True

    exp.add_in_param('b', (0, 1), condition=cond)
    # coord 2 -> value 10 (index 2 of [0,1,10,11]); the condition must see 10.
    exp.suggest_params(_ScriptedTrial({'a': 2, 'b': 0}))
    assert seen['a'] == 10


def test_reconstruct_coords_round_trips_dynamic_domain() -> None:
    exp = Experiment()
    exp.add_distinct_sorted('p', 3, (1, 6))
    exp.add_in_param('w', Union([(0.0, 1.0), (10.0, 11.0)]))
    live = exp.suggest_params(_ScriptedTrial({'p_0': 2, 'p_1': 4, 'p_2': 5, 'w': 1.5}))
    coords, specs = exp.reconstruct_coords(dict(live.realized_params))
    assert dict(coords) == dict(live.realized_params)
    # Every reconstructed coordinate lies within its declared distribution.
    for name, spec in specs.items():
        assert spec.lo <= coords[name] <= spec.hi


def test_reconstruct_values_maps_coords_to_values() -> None:
    # For plotting: stored coordinates -> human-meaningful domain values.
    exp = Experiment()
    exp.add_in_param('g', Union([(0, 3), (5, 6)]))
    exp.add_in_param('w', Union([(0.0, 1.0), (10.0, 11.0)]))
    # coord g=4 -> value 5 (across the gap); coord w=1.5 -> value 10.5.
    stored = dict(exp.suggest_params(_ScriptedTrial({'g': 4, 'w': 1.5})).realized_params)
    assert stored == {'g': 4, 'w': 1.5}  # coordinates are what get checkpointed
    values = exp.reconstruct_values(stored)
    assert values['g'] == 5
    assert _approx(values['w'], 10.5)


if __name__ == '__main__':
    # Runnable without pytest; pytest still collects the test_* functions.
    for _name, _fn in sorted(dict(globals()).items()):
        if _name.startswith('test_') and callable(_fn):
            _fn()
            print(f'PASS {_name}')
    print('all tests passed')
