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

import typing
from dataclasses import dataclass
from typing import Protocol, Any, OrderedDict, NamedTuple, Optional, runtime_checkable

from multimethod import multimethod

ParamType = int|float

@runtime_checkable
class InParamCondition(Protocol):
    def __call__(self, realized_args: dict[str, ParamType], /) -> bool:
        ...

@runtime_checkable
class ParamMapping[O](Protocol):
    def __call__(self, realized_args: dict[str, ParamType], /) -> Optional[O]:
        ...

class TrialObj(Protocol):
    def suggest_int(self, name: str, lo: int, hi: int, /) -> int:
        ...

    def suggest_float(self, name: str, lo: float, hi: float, /) -> float:
        ...

Always: InParamCondition = lambda _: True

@dataclass
class InParam[T: ParamType]:
    name: str
    typ: type[T]
    bounds: tuple[T, T]  # closed (inclusive) range
    condition: InParamCondition

    def __init__(self, name: str, bounds: tuple[T, T], condition: InParamCondition = Always):
        self.name = name
        self.bounds = bounds
        self.condition = condition
        self.typ = type(bounds[0])

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
    def _[T: ParamType](self, name: str, bounds: tuple[T, T], condition: InParamCondition = Always) -> None:
        self.add_in_param(InParam(name, bounds, condition))

    @multimethod
    def add_out_param(self, param: OutParam[Any]) -> None:
        self.out_params[param.name] = param

    @add_out_param.register
    def _[T: ParamType](self, name: str, mapping: ParamMapping[T]) -> None:
        self.add_out_param(OutParam(name, mapping))

    def suggest_params(self, trial: TrialObj) -> SuggestParamsResult:
        realized_params: OrderedDict[str, ParamType] = OrderedDict()
        for in_param_name, in_param in self.in_params.items():
            if in_param.condition(realized_params):
                value: int|float
                if (typ := in_param.typ) is int:
                    value = trial.suggest_int(in_param_name, typing.cast(int, in_param.bounds[0]), typing.cast(int, in_param.bounds[1]))
                elif typ is float:
                    value = trial.suggest_float(in_param_name, in_param.bounds[0], in_param.bounds[1])
                else:
                    raise Exception(f"Unsupported type {typ}")
                realized_params[in_param_name] = value

        recipe_facing_args: OrderedDict[str, Any] = OrderedDict()
        for out_param_name, out_param in self.out_params.items():
            if (val := out_param.mapping(realized_params)) is not None:
                recipe_facing_args[out_param_name] = val

        return SuggestParamsResult(realized_params, recipe_facing_args)
