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

from dataclasses import dataclass
from typing import Protocol, Any, Callable, OrderedDict, NamedTuple, Optional, Sequence, runtime_checkable

from multimethod import multimethod

ParamType = int|float

# A raw search coordinate (what Optuna/TPE samples over) and the domain value
# the recipe consumes. For a plain Interval the two are identical; a Union maps a
# contiguous coordinate onto a gapped value space (see Domain, below).
Coord = int|float
Value = int|float

@runtime_checkable
class InParamCondition(Protocol):
    def __call__(self, realized_args: dict[str, ParamType], /) -> bool:
        ...

@runtime_checkable
class InParamConstraint(Protocol):
    def __call__(self, value: ParamType, /) -> bool:
        ...

class InfeasibleParamError(Exception):
    """A suggested value violated its InParam's constraint.

    Optimizer frontends driving a live trial should translate this into their
    framework's rejection mechanism (e.g. optuna.TrialPruned).
    """
    def __init__(self, param_name: str, value: ParamType):
        super().__init__(f"Value {value!r} for parameter '{param_name}' violates its constraint")
        self.param_name = param_name
        self.value = value

@runtime_checkable
class ParamMapping[O](Protocol):
    def __call__(self, realized_args: dict[str, ParamType], /) -> Optional[O]:
        ...

class TrialObj(Protocol):
    def suggest_int(self, name: str, lo: int, hi: int, /) -> int:
        ...

    def suggest_float(self, name: str, lo: float, hi: float, /) -> float:
        ...


class CoordSpec(NamedTuple):
    """The raw-coordinate range a Domain wants sampled for one parameter."""
    kind: type  # int or float
    lo: Coord
    hi: Coord


@runtime_checkable
class Domain(Protocol):
    """A reparameterized search space for a single parameter.

    ``coord_distribution`` gives the contiguous coordinate range Optuna should
    sample (possibly narrowed using already-realized *values*, for dynamic
    domains like uniqueness).  ``value_of`` deterministically maps a sampled
    coordinate to the domain value the recipe consumes.  Reparameterizing this
    way means every coordinate Optuna proposes is automatically valid -- no
    reject-and-retry -- and the coordinate space stays dense for TPE.
    """
    def coord_distribution(self, realized: dict[str, Value], /) -> CoordSpec:
        ...

    def value_of(self, coord: Coord, realized: dict[str, Value], /) -> Value:
        ...


@dataclass
class Interval:
    """A single contiguous ``[lo, hi]`` range; coordinate == value."""
    lo: ParamType
    hi: ParamType

    def coord_distribution(self, realized: dict[str, Value], /) -> CoordSpec:
        kind = int if isinstance(self.lo, int) and isinstance(self.hi, int) else float
        return CoordSpec(kind, self.lo, self.hi)

    def value_of(self, coord: Coord, realized: dict[str, Value], /) -> Value:
        return coord


@dataclass(init=False)
class Discrete:
    """An explicit set of allowed values; coordinate is an index into them.

    Values are sorted and de-duplicated so the index space is stable and
    order-preserving (adjacent indices map to adjacent values).
    """
    values: list[ParamType]

    def __init__(self, values: Sequence[ParamType]):
        self.values = sorted(set(values))
        if not self.values:
            raise ValueError("Discrete domain needs at least one value")

    def coord_distribution(self, realized: dict[str, Value], /) -> CoordSpec:
        return CoordSpec(int, 0, len(self.values) - 1)

    def value_of(self, coord: Coord, realized: dict[str, Value], /) -> Value:
        return self.values[int(coord)]


class Union:
    """A discontinuous domain built from a list of inclusive ``(lo, hi)`` intervals.

    Integer unions (all-int endpoints) enumerate every value and index them like
    a ``Discrete``.  Float unions sample a single coordinate ``u`` in
    ``[0, total_length]`` and place it via a piecewise cumulative-length map
    (inverse-CDF over a uniform-on-union).  Both are order-preserving with zero
    rejection.  Intervals are assumed disjoint.
    """
    def __init__(self, intervals: Sequence[tuple[ParamType, ParamType]]):
        if not intervals:
            raise ValueError("Union domain needs at least one interval")
        self.intervals = [(lo, hi) for lo, hi in intervals]
        self._is_int = all(isinstance(lo, int) and isinstance(hi, int) for lo, hi in self.intervals)
        if self._is_int:
            values: list[ParamType] = []
            for lo, hi in self.intervals:
                values.extend(range(int(lo), int(hi) + 1))
            self._discrete = Discrete(values)
        else:
            # (cumulative_offset, segment_lo, segment_length) per interval.
            self._segments: list[tuple[float, float, float]] = []
            total = 0.0
            for lo, hi in self.intervals:
                length = float(hi) - float(lo)
                self._segments.append((total, float(lo), length))
                total += length
            self._total = total

    def coord_distribution(self, realized: dict[str, Value], /) -> CoordSpec:
        if self._is_int:
            return self._discrete.coord_distribution(realized)
        return CoordSpec(float, 0.0, self._total)

    def value_of(self, coord: Coord, realized: dict[str, Value], /) -> Value:
        if self._is_int:
            return self._discrete.value_of(coord, realized)
        u = float(coord)
        for offset, lo, length in self._segments:
            if u <= offset + length:
                return lo + (u - offset)
        # Floating-point overshoot of the final endpoint: clamp to it.
        offset, lo, length = self._segments[-1]
        return lo + length


Always: InParamCondition = lambda _: True
Unconstrained: InParamConstraint = lambda _: True

# What add_in_param / InParam accept for the search space: a bare (lo, hi) tuple
# (auto-wrapped as an Interval), a ready-made Domain, or -- for dynamic domains
# such as uniqueness -- a callable that returns one of those given the values
# realized so far.
StaticDomain = Domain | tuple[ParamType, ParamType]
DomainResolver = Callable[[dict[str, Value]], StaticDomain]
DomainSpec = StaticDomain | DomainResolver


def _as_domain(spec: StaticDomain) -> Domain:
    if isinstance(spec, Domain):
        return spec
    if isinstance(spec, tuple):
        return Interval(spec[0], spec[1])
    raise TypeError(f"Expected a Domain or (lo, hi) tuple, got {spec!r}")


class InParam[T: ParamType]:
    name: str
    condition: InParamCondition
    constraint: InParamConstraint

    def __init__(self, name: str, domain: DomainSpec, condition: InParamCondition = Always,
                 constraint: InParamConstraint = Unconstrained):
        self.name = name
        self.condition = condition
        self.constraint = constraint
        # A resolver is a plain callable that is not itself a Domain instance.
        if callable(domain) and not isinstance(domain, Domain):
            self._resolver: Optional[DomainResolver] = domain
            self._static: Optional[Domain] = None
        else:
            self._resolver = None
            self._static = _as_domain(domain)

    def resolve_domain(self, realized: dict[str, Value]) -> Domain:
        """The Domain governing this param for a trial with the given realized values."""
        if self._resolver is not None:
            return _as_domain(self._resolver(realized))
        assert self._static is not None
        return self._static

@dataclass
class OutParam[T]:
    name: str
    mapping: ParamMapping[T]

class SuggestParamsResult(NamedTuple):
    realized_params: dict[str, ParamType]
    recipe_facing_args: dict[str, Any]

class Experiment:
    in_params: OrderedDict[str, InParam[int] | InParam[float]]
    out_params: OrderedDict[str, OutParam[Any]]

    def __init__(self) -> None:
        self.in_params = OrderedDict()
        self.out_params = OrderedDict()

    @multimethod
    def add_in_param(self, param: InParam[Any]) -> None:
        self.in_params[param.name] = param

    @add_in_param.register
    def _(self, name: str, domain: Any, condition: InParamCondition = Always,
          constraint: InParamConstraint = Unconstrained) -> None:
        self.add_in_param(InParam(name, domain, condition, constraint))

    @multimethod
    def add_out_param(self, param: OutParam[Any]) -> None:
        self.out_params[param.name] = param

    @add_out_param.register
    def _[T: ParamType](self, name: str, mapping: ParamMapping[T]) -> None:
        self.add_out_param(OutParam(name, mapping))

    def add_distinct_sorted(self, name_prefix: str, k: int, bounds: tuple[int, int]) -> list[str]:
        """Add ``k`` integer params that are strictly increasing (hence distinct).

        Each param samples from a dynamically narrowed Interval, so every trial
        is valid with no rejection; the monotonic ordering also collapses the
        ``k!`` permutation symmetry.  Use this when the params are
        interchangeable.  Returns the generated parameter names.
        """
        lo, hi = bounds
        if hi - lo + 1 < k:
            raise ValueError(f"Cannot pick {k} distinct ints from [{lo}, {hi}]")
        names = [f'{name_prefix}_{i}' for i in range(k)]
        for i, name in enumerate(names):
            prev = names[i - 1] if i > 0 else None
            def resolve(realized: dict[str, Value], i: int = i, prev: Optional[str] = prev) -> Interval:
                low = lo if prev is None else int(realized[prev]) + 1
                high = hi - (k - 1 - i)
                return Interval(low, high)
            self.add_in_param(name, resolve)
        return names

    def add_distinct_choice(self, name_prefix: str, k: int, pool: Sequence[ParamType]) -> list[str]:
        """Add ``k`` params selecting distinct values from ``pool`` without replacement.

        Each param samples an index into the values not yet chosen, so order is
        preserved (the params are distinguishable roles) and every trial is a
        valid partial permutation with no rejection.  Returns the generated names.
        """
        pool_sorted = sorted(set(pool))
        if len(pool_sorted) < k:
            raise ValueError(f"Cannot pick {k} distinct values from a pool of {len(pool_sorted)}")
        names = [f'{name_prefix}_{i}' for i in range(k)]
        for i, name in enumerate(names):
            earlier = tuple(names[:i])
            def resolve(realized: dict[str, Value], earlier: tuple[str, ...] = earlier) -> Discrete:
                used = {realized[e] for e in earlier}
                return Discrete([v for v in pool_sorted if v not in used])
            self.add_in_param(name, resolve)
        return names

    def _walk_in_params(
        self,
        get_coord: Callable[[str, CoordSpec], Coord],
        present: Optional[Callable[[str], bool]] = None,
    ) -> tuple[OrderedDict[str, Value], OrderedDict[str, Coord], OrderedDict[str, CoordSpec]]:
        """Resolve in_params in order, threading realized *values* through
        conditions and dynamic domains.

        ``get_coord(name, spec)`` supplies each active param's raw coordinate --
        from a live trial when sampling, or from a stored entry when replaying a
        checkpoint.  ``present`` optionally skips params absent from a (partial)
        stored entry.  Returns the realized values (what conditions, domains and
        out_param mappings see), the raw coordinates (what gets checkpointed),
        and the per-param CoordSpecs (used to rebuild Optuna distributions).
        """
        realized_values: OrderedDict[str, Value] = OrderedDict()
        realized_coords: OrderedDict[str, Coord] = OrderedDict()
        coord_specs: OrderedDict[str, CoordSpec] = OrderedDict()
        for name, in_param in self.in_params.items():
            if not in_param.condition(realized_values):
                continue
            if present is not None and not present(name):
                continue
            domain = in_param.resolve_domain(realized_values)
            spec = domain.coord_distribution(realized_values)
            coord = get_coord(name, spec)
            value = domain.value_of(coord, realized_values)
            if not in_param.constraint(value):
                raise InfeasibleParamError(name, value)
            realized_values[name] = value
            realized_coords[name] = coord
            coord_specs[name] = spec
        return realized_values, realized_coords, coord_specs

    def suggest_params(self, trial: TrialObj) -> SuggestParamsResult:
        def get_coord(name: str, spec: CoordSpec) -> Coord:
            if spec.kind is int:
                return trial.suggest_int(name, int(spec.lo), int(spec.hi))
            return trial.suggest_float(name, float(spec.lo), float(spec.hi))

        realized_values, realized_coords, _ = self._walk_in_params(get_coord)

        recipe_facing_args: OrderedDict[str, Any] = OrderedDict()
        for out_param_name, out_param in self.out_params.items():
            if (val := out_param.mapping(realized_values)) is not None:
                recipe_facing_args[out_param_name] = val

        # The checkpoint stores raw coordinates (== values for plain Intervals),
        # since those are what TPE searched and what exact replay needs.
        return SuggestParamsResult(realized_coords, recipe_facing_args)

    def reconstruct_coords(
        self, stored: dict[str, ParamType]
    ) -> tuple[OrderedDict[str, Coord], OrderedDict[str, CoordSpec]]:
        """Rebuild a stored trial's raw coordinates and their CoordSpecs.

        Used by the checkpoint loader: replays the stored coordinates through the
        experiment in order so dynamic domains recover their exact per-trial
        coordinate ranges.  Params absent from ``stored`` (gated out by a
        condition when the trial was recorded) are skipped.
        """
        def get_coord(name: str, spec: CoordSpec) -> Coord:
            raw = stored[name]
            return int(raw) if spec.kind is int else float(raw)

        _, coords, specs = self._walk_in_params(get_coord, present=lambda name: name in stored)
        return coords, specs

    def reconstruct_values(self, stored: dict[str, ParamType]) -> OrderedDict[str, Value]:
        """Map a stored trial's raw coordinates back to their domain values.

        For plain ``Interval`` params a value equals its coordinate; for ``Union``
        and other reparameterized domains this recovers the human-meaningful value
        (e.g. the actual number on either side of a union's gap).  Used for
        plotting.  Params absent from ``stored`` are skipped.
        """
        def get_coord(name: str, spec: CoordSpec) -> Coord:
            raw = stored[name]
            return int(raw) if spec.kind is int else float(raw)

        values, _, _ = self._walk_in_params(get_coord, present=lambda name: name in stored)
        return values
